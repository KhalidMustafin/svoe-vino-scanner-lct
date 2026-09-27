"""Движок сочетаний «Лозы» — только оценка пары «вино + блюдо», для сборки `data/somm`.

Источник — `Code/backend/app/recommend/pairing.py` (f277717). Перенесены без изменений логики:
`PairingRule`, `_CompiledRule.applies`, `_FLAVOR_TO_DESCRIPTOR`, `_DishView`, `_WineView`,
`PairingEngine.from_file` (справочник + накладка «Лозы»), `score`, `_fit`,
`_meets_requirements`, `_normalize`, константы `_SCORE_SCALE`, `MAX_DISPLAY_SCORE`,
`_FIT_WEIGHT`, `_REQUIREMENT_WEIGHT`.

Не перенесено — и это не упущение, а решение 24.09:

- портальный приор (`_matches_portal_tags`, `_PORTAL_MATCH_BONUS`, `portal_dish_mapping`,
  правило `portal_editorial_prior` в накладке «Лозы» и так выключено): у вина нет портальных блюд;
- `pair_wines` с развязкой ничьих по `expert_score` и `_selection_traits` с превосходными
  степенями («самое свежее»): подборку вин к блюду сервис строит сам, по фактам организатора;
- поиск блюда в тексте (`DishRegistry`, отрицания): вопросы гостя разбирает сервис;
- pydantic-модели: вино и блюдо приходят словарями, поля — те же, что читают правила.

Добавлено:

- **накладка сканера** `overlay.json` поверх накладки «Лозы»: у правил «Лозы» меняет только
  условия, а вес, имя и объяснение оставляет как есть (иначе сборка падает); свои правила
  сканера — отдельным разделом `added`, с новым `id`, которого у «Лозы» нет, — см. `load_rules`;
- `PairScore.total` — сумма весов до сигмоиды: по ней сборка упорядочивает блюда без потолка
  0,96, на котором сверху сходятся десятки пар;
- `WineView.sugar_by_name` — сладость профиля взята по названию («возможно сладкое»), а не из
  карточки: по нему накладка отличает догадку от факта (`sweet_wine_on_fish`,
  `maybe_sweet_wine_on_fish`, `sweet_wine_on_savoury_main` у рыбных блюд);
- кэш условий блюда: условие, которое не читает `wine`, вычисляется один раз на блюдо. Логика та
  же, что у `_CompiledRule.applies`: сначала условие блюда, потом вина.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .requirements import Limits, parse, satisfies
from .safe_eval import ExpressionError, SafeExpression

#: Масштаб сигмоиды, переводящей сумму весов в оценку 0..1 (у «Лозы» подобран по разборам).
_SCORE_SCALE = 5.0

#: Потолок оценки «Лозы»: сочетание — вопрос вкуса, «100 %» обещало бы больше, чем знают правила.
MAX_DISPLAY_SCORE = 0.96

#: Вес непрерывной поправки: насколько точно вино попало в интенсивность блюда.
_FIT_WEIGHT = 1.2

#: Вес соответствия требованиям блюда (строка `wine_requirements` справочника).
_REQUIREMENT_WEIGHT = 0.8

#: Вкусовой тег блюда на языке ароматики вина — для правила «ароматический мост».
_FLAVOR_TO_DESCRIPTOR: dict[str, str] = {
    "mushroom": "earth",
    "truffle": "earth",
    "beet": "earth",
    "root_vegetable": "earth",
    "black_pepper": "pepper",
    "pepper": "pepper",
    "smoked": "smoke",
    "smoke": "smoke",
    "grilled": "smoke",
    "honey": "honey",
    "berry": "red_fruit",
    "sour_cream": "butter",
    "butter": "butter",
    "cream": "butter",
    "toast": "toast",
    "bread": "toast",
    "walnut": "nutty",
    "nut": "nutty",
    "dill": "herbal",
    "herbal": "herbal",
    "herbs": "herbal",
    "apple": "green_apple",
    "citrus": "citrus",
    "lemon": "citrus",
    "marine": "mineral",
    "seafood": "mineral",
    "fish": "mineral",
    "chocolate": "chocolate",
    "caramel": "chocolate",
    "dried_fruit": "dried_fruit",
    "spice": "spice",
    "cinnamon": "spice",
    "floral": "floral",
}

#: Поля правила «Лозы», которые накладка сканера вправе менять.
OVERLAY_FIELDS = frozenset({"dish_condition", "wine_condition"})

#: Поля своего правила сканера (раздел `added` накладки): всё правило целиком и причина.
ADDED_FIELDS = frozenset(
    {"reason", "name", "dish_condition", "wine_condition", "weight", "explanation_ru"}
)


@dataclass(frozen=True, slots=True)
class PairingRule:
    """Правило сочетаемости с весом и объяснением (`PairingRule` «Лозы» без pydantic)."""

    id: str
    name: str
    dish_condition: str
    wine_condition: str
    weight: float
    explanation_ru: str
    principle: str = ""
    source: str | None = None

    @classmethod
    def from_json(cls, entry: Mapping[str, Any]) -> PairingRule:
        return cls(
            id=str(entry["id"]),
            name=str(entry["name"]),
            dish_condition=str(entry["dish_condition"]),
            wine_condition=str(entry["wine_condition"]),
            weight=float(entry["weight"]),
            explanation_ru=str(entry["explanation_ru"]),
            principle=str(entry.get("principle") or ""),
            source=entry.get("source"),
        )


class _CompiledRule:
    """Правило с разобранными выражениями — готово к многократной проверке."""

    __slots__ = ("dish_expr", "dish_only", "rule", "wine_expr")

    def __init__(self, rule: PairingRule) -> None:
        self.rule = rule
        self.dish_expr = SafeExpression(rule.dish_condition)
        self.wine_expr = SafeExpression(rule.wine_condition)
        # Условие блюда без `wine` одинаково для всех вин — его можно вычислить один раз.
        self.dish_only = "wine" not in self.dish_expr.names

    def dish_applies(self, dish: DishView, wine: WineView) -> bool:
        return self.dish_expr.matches({"dish": dish, "wine": wine})

    def wine_applies(self, dish: DishView, wine: WineView) -> bool:
        return self.wine_expr.matches({"dish": dish, "wine": wine})


class DishView:
    """Плоское представление блюда для выражений правил (`_DishView` «Лозы»)."""

    __slots__ = (
        "acidity",
        "category",
        "descriptors",
        "fat",
        "flavor_tags",
        "id",
        "intensity",
        "requirements",
        "salt",
        "spice",
        "sweetness",
        "umami",
    )

    def __init__(self, dish: Mapping[str, Any]) -> None:
        # Умолчания — как у модели `Dish` «Лозы».
        self.id = str(dish["id"])
        self.fat = float(dish.get("fat", 2.0))
        self.acidity = float(dish.get("acidity", 1.0))
        self.spice = float(dish.get("spice", 0.0))
        self.sweetness = float(dish.get("sweetness", 0.0))
        self.salt = float(dish.get("salt", 2.0))
        self.umami = float(dish.get("umami", 2.0))
        self.intensity = float(dish.get("intensity", 3.0))
        self.flavor_tags: set[str] = set(dish.get("flavor_tags") or ())
        self.category = str(dish.get("category") or "")
        # Ароматика блюда в тех же кодах, что и ароматика вина.
        self.descriptors: set[str] = {
            _FLAVOR_TO_DESCRIPTOR[tag] for tag in self.flavor_tags if tag in _FLAVOR_TO_DESCRIPTOR
        }
        self.requirements: Limits = parse(dish.get("wine_requirements"))


class _Profile:
    """Оси профиля для `satisfies`: восемь чисел по шкале 0..5."""

    __slots__ = (
        "acidity",
        "alcohol",
        "aroma_intensity",
        "body",
        "effervescence",
        "oak",
        "sweetness",
        "tannin",
    )

    def __init__(self, profile: Mapping[str, Any]) -> None:
        for axis in self.__slots__:
            setattr(self, axis, float(profile[axis]))


class WineView:
    """Плоское представление вина для выражений правил (`_WineView` «Лозы»).

    Вход — словарь сборки: `profile` (8 осей), `descriptors` (коды ароматики приоров), `color`
    (`red` | `white` | `rose` | `orange`), `kind` (`still` | `sparkling`), `region` (код ЗГУ
    «Лозы»), `grapes` (коды сортов), `serve_temp_c` (`[от, до]`) и `sugar_by_name` (bool, нет —
    `false`): сладость профиля — догадка по названию, а не сахар карточки (поле сканера, у «Лозы»
    его нет; читает накладка). Портальных блюд нет: `dishes` всегда пусто.
    """

    __slots__ = (
        "acidity",
        "alcohol",
        "aroma_intensity",
        "body",
        "color",
        "descriptors",
        "dishes",
        "effervescence",
        "grapes",
        "id",
        "intensity",
        "kind",
        "oak",
        "profile",
        "region",
        "serve_temp_max",
        "serve_temp_min",
        "sugar_by_name",
        "sweetness",
        "tannin",
    )

    def __init__(self, wine: Mapping[str, Any]) -> None:
        self.id = str(wine["id"])
        self.profile = _Profile(wine["profile"])
        profile = self.profile
        self.sweetness = profile.sweetness
        self.acidity = profile.acidity
        self.tannin = profile.tannin
        self.body = profile.body
        self.alcohol = profile.alcohol
        self.oak = profile.oak
        self.aroma_intensity = profile.aroma_intensity
        self.effervescence = profile.effervescence
        # «Интенсивность вина» правил — совокупность тела, ароматики и градуса.
        self.intensity = (
            0.45 * profile.body + 0.35 * profile.aroma_intensity + 0.20 * profile.alcohol
        )
        self.descriptors: set[str] = set(wine.get("descriptors") or ())
        self.color = str(wine["color"])
        self.kind = str(wine.get("kind") or "still")
        self.region = str(wine.get("region") or "other")
        self.grapes: set[str] = set(wine.get("grapes") or ())
        self.dishes: set[str] = set()
        low, high = wine.get("serve_temp_c") or (8.0, 18.0)
        self.serve_temp_min, self.serve_temp_max = float(low), float(high)
        self.sugar_by_name = bool(wine.get("sugar_by_name", False))


@dataclass(frozen=True, slots=True)
class PairScore:
    """Оценка пары: сумма весов, оценка 0..1 и сработавшие правила «за» и «против»."""

    total: float
    score: float
    plus: tuple[str, ...]
    minus: tuple[str, ...]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(
    pairing_path: Path, fixes_path: Path | None = None, overlay_path: Path | None = None
) -> list[PairingRule]:
    """Правила справочника → накладка «Лозы» → накладка сканера.

    Первые два шага — `PairingEngine.from_file` «Лозы»: накладка переопределяет правила по `id`,
    выключает перечисленные в `disabled` и добавляет недостающие. Третий шаг — накладка сканера:
    у правила «Лозы» она вправе менять только условия (`OVERLAY_FIELDS`), а свои правила пишет
    целиком в разделе `added` (`ADDED_FIELDS`) под новым `id` — так правило «Лозы» не подменить
    чужим весом или объяснением. Битое правило здесь — ошибка сборки, а не тихий пропуск, как в
    сервисе «Лозы»: данные собираются заранее.
    """
    payload = _read_json(pairing_path)
    overrides: dict[str, Mapping[str, Any]] = {}
    disabled: set[str] = set()
    if fixes_path is not None and fixes_path.exists():
        fixes = _read_json(fixes_path)
        overrides = {entry["id"]: entry for entry in fixes.get("rules", []) if "id" in entry}
        disabled = set(fixes.get("disabled", []))

    entries: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for entry in payload.get("rules", []):
        rule_id = entry.get("id", "")
        if rule_id in disabled:
            continue
        entries.append(overrides.get(rule_id, entry))
        seen.add(rule_id)
    for rule_id, entry in overrides.items():
        if rule_id not in seen and rule_id not in disabled:
            entries.append(entry)

    if overlay_path is not None:
        overlay = _read_json(overlay_path)
        patches = overlay.get("rules", {})
        by_id = {entry["id"]: dict(entry) for entry in entries}
        for rule_id, patch in patches.items():
            if rule_id not in by_id:
                raise ValueError(f"накладка сканера: правила {rule_id!r} нет в справочнике")
            extra = set(patch) - OVERLAY_FIELDS - {"reason"}
            if extra:
                raise ValueError(f"накладка сканера: {rule_id} меняет {sorted(extra)}")
            for field in OVERLAY_FIELDS & set(patch):
                by_id[rule_id][field] = patch[field]
        entries = [by_id[entry["id"]] for entry in entries]
        for rule_id, added in overlay.get("added", {}).items():
            if rule_id in by_id or rule_id in disabled or rule_id in seen:
                raise ValueError(f"накладка сканера: правило {rule_id!r} уже есть у «Лозы»")
            if set(added) != ADDED_FIELDS:
                raise ValueError(
                    f"накладка сканера: у своего правила {rule_id} поля {sorted(added)}, "
                    f"нужны {sorted(ADDED_FIELDS)}"
                )
            if not isinstance(added["weight"], int | float) or not added["weight"]:
                raise ValueError(f"накладка сканера: у своего правила {rule_id} нет веса")
            entries.append({"id": rule_id, **{k: v for k, v in added.items() if k != "reason"}})

    rules = [PairingRule.from_json(entry) for entry in entries]
    for rule in rules:
        _CompiledRule(rule)  # битое выражение — ошибка сборки здесь, а не позже
    return rules


class PairingEngine:
    """Оценивает сочетаемость вина и блюда по правилам (`PairingEngine` «Лозы» без портала)."""

    def __init__(self, rules: Sequence[PairingRule]) -> None:
        self._rules = [_CompiledRule(rule) for rule in rules]

    @property
    def rules(self) -> list[PairingRule]:
        return [compiled.rule for compiled in self._rules]

    def score(self, wine: WineView, dish: DishView) -> PairScore:
        """Оценка пары в 0..1, сумма весов и сработавшие правила: за и против."""
        return self._score(wine, dish, None)

    def matrix(
        self, wines: Iterable[WineView], dishes: Sequence[DishView]
    ) -> dict[str, dict[str, PairScore]]:
        """Все пары «вино × блюдо». Условия блюда без `wine` считаются один раз на блюдо."""
        # Пустое вино-заглушка: условию «только блюдо» вино не нужно, но окружению — имя.
        cache: dict[str, list[bool | None]] = {
            dish.id: [
                compiled.dish_applies(dish, None) if compiled.dish_only else None  # type: ignore[arg-type]
                for compiled in self._rules
            ]
            for dish in dishes
        }
        return {
            wine.id: {dish.id: self._score(wine, dish, cache[dish.id]) for dish in dishes}
            for wine in wines
        }

    def _score(
        self, wine: WineView, dish: DishView, dish_cache: Sequence[bool | None] | None
    ) -> PairScore:
        plus: list[PairingRule] = []
        minus: list[PairingRule] = []
        total = 0.0
        for index, compiled in enumerate(self._rules):
            cached = dish_cache[index] if dish_cache is not None else None
            try:
                dish_ok = cached if cached is not None else compiled.dish_applies(dish, wine)
                if not dish_ok or not compiled.wine_applies(dish, wine):
                    continue
            except ExpressionError as exc:
                raise ValueError(f"правило {compiled.rule.id} не вычислилось: {exc}") from exc
            total += compiled.rule.weight
            (plus if compiled.rule.weight > 0 else minus).append(compiled.rule)

        total += _FIT_WEIGHT * self._fit(dish, wine)
        total += _REQUIREMENT_WEIGHT * satisfies(wine.profile, dish.requirements)
        return PairScore(
            total=total,
            score=self._normalize(total),
            plus=tuple(rule.id for rule in plus),
            minus=tuple(rule.id for rule in minus),
        )

    @staticmethod
    def _fit(dish: DishView, wine: WineView) -> float:
        """Насколько точно вино попадает в блюдо, в долях от единицы (`_fit` «Лозы»)."""
        intensity_gap = abs(wine.intensity - dish.intensity) / 5.0
        # Жирному блюду нужна кислотность, лёгкому она безразлична.
        acid_need = max(0.0, (dish.fat - 2.0) / 3.0)
        acid_gap = max(0.0, acid_need - wine.acidity / 5.0)
        # Танины уместны ровно настолько, насколько в блюде есть белок и жир.
        tannin_need = max(0.0, (dish.umami + dish.fat - 4.0) / 6.0)
        tannin_gap = abs(tannin_need - wine.tannin / 5.0)
        penalty = 0.5 * intensity_gap + 0.3 * acid_gap + 0.2 * tannin_gap
        return max(0.0, 1.0 - penalty)

    @staticmethod
    def _normalize(total: float) -> float:
        """Сумма весов → оценка 0..1: сигмоида с потолком `MAX_DISPLAY_SCORE`."""
        return min(1.0 / (1.0 + math.exp(-total / _SCORE_SCALE)), MAX_DISPLAY_SCORE)
