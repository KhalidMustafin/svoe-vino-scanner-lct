"""Пост-фильтр исходящего текста: рамки закона о рекламе алкоголя.

Перенос `Code/backend/app/core/content_filter.py` «Лозы» (eed9c2e). Поменялись аннотации типов
под Python 3.12 и один символ: в `_BENEFIT_WEAK` оригинала после `польз[ауые]` стоит не граница
слова `\\b`, а литеральный backspace (байт 0x08). Такая ветка не совпадает ни с каким текстом, и
«вино — польза для сердца» проходила фильтр. Здесь стоит `\\b`, как и задумано в комментарии.
Тесты «Лозы» перенесены без изменений, к ним добавлен тест на эту ветку
(`tests/unit/test_recommend_content_filter.py`).

Ст. 21 ФЗ-38 запрещает рекламу алкоголя, и информационному сервису нельзя
скатываться в неё даже случайно: утверждение о пользе, цена рядом с
названием, призыв купить — любое из этого превращает справку в рекламу.

Объяснения «похожих» у сканера собираются шаблонами из фактов каталога, поэтому фильтр —
страховочная сетка: шаблоны написаны людьми, а фильтр проверяет, что в них не просочилось
ничего из запретного и в сочетании с названиями вин (названия пишут винодельни).

Устройство после проверки первой версии на обход:

- текст НОРМАЛИЗУЕТСЯ до проверки: NFKC, вычистка мягких переносов и
  невидимых символов, схлопывание латинских гомоглифов в кириллицу —
  «Винo пoлезнo» с латинскими «o» больше не лазейка;
- запрещённое ищется парами «субъект + предикат» внутри предложения,
  без жёсткого порядка слов: «вино укрепляет иммунитет» и «иммунитет
  укрепляется вином» — одно и то же нарушение;
- у паттернов есть охранные исключения: «не существует безопасной дозы» —
  обязательная законная оговорка, а не нарушение; «пользователи каталога»
  и «акционерное общество» — не польза и не акция.

Фильтр детерминированный и грубый намеренно: ложное срабатывание стоит
одной перефразированной фразы, пропуск — нарушения закона.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import NamedTuple

logger = logging.getLogger(__name__)


class FilterVerdict(NamedTuple):
    """Результат проверки: чистый текст либо причины отклонения."""

    text: str | None
    violations: list[str]

    @property
    def clean(self) -> bool:
        return not self.violations


# ---------------------------------------------------------------------------
# Нормализация
# ---------------------------------------------------------------------------

#: Невидимые и разрывающие символы, которыми прячут слова от фильтров.
_INVISIBLE_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")

#: Латинские двойники кириллических букв.
_HOMOGLYPHS = str.maketrans("aoecxypkbmtnAOECXYPKBMTH", "аоесхурквмтпАОЕСХУРКВМТН")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_RE.sub("", text)
    # Точка в сокращении «тыс. руб.» — не конец предложения: без этого
    # цена разрезалась бы на два «предложения» и ускользала от паттерна.
    text = re.sub(r"\b(тыс|руб|р)\.", r"\1 ", text, flags=re.IGNORECASE)
    return text.translate(_HOMOGLYPHS).lower()


# ---------------------------------------------------------------------------
# Словари нарушений
# ---------------------------------------------------------------------------

#: Субъект «алкоголь» — включая анафору: «Это вино особенное. Оно лечит.»
_SUBJECT = r"(?:вин[оаеу]\b|вином\b|алкогол\w*|напит\w*|бокал\w*|игрист\w*|оно\b|он\b)"

#: Сильные предикаты пользы: срабатывают рядом с субъектом в любом порядке.
_BENEFIT_STRONG = (
    r"(?:целебн\w*|лечебн\w*|лечит\b|лечат\b|исцеля\w*|оздоров\w*|благотворн\w*"
    r"|продлева\w*\s+жизнь)"
)

#: Слабые предикаты: требуют медицинского дополнения, иначе ложные
#: срабатывания («укрепляет позиции», «полезные фильтры каталога»).
# Падеж «пользу» отсутствовал, и самая частая формулировка нарушения —
# «вино приносит пользу здоровью» — проходила фильтр насквозь. Это ровно
# та фраза, которую ст. 21 ФЗ-38 запрещает дословно.
_BENEFIT_WEAK = (
    r"(?:полезн\w*|полезен|польз[ауые]\b|укрепля\w*|улучша\w*|снижа\w*|понижа\w*|нормализу\w*"
    r"|защища\w*|помога\w*|принос\w*)"
)
_MEDICAL_OBJECT = (
    r"(?:здоровь\w*|сердц\w*|сосуд\w*|давлен\w*|иммунитет\w*|холестерин\w*|пищеварен\w*"
    r"|печен[ьи]\w*|сон\b|сну\b|нерв\w*|простуд\w*|болезн\w*)"
)

#: Призыв к покупке — любые формы, включая «стоит купить».
_PURCHASE = re.compile(
    r"\b(?:купи(?:те)?\b|покупай\w*|закаж\w*|заказыва\w*|приобрет\w*|приобрес\w*)"
    r"|\b(?:стоит|советую|рекомендую|надо|нужно)\s+(?:купить|приобрести|заказать)"
    r"|\bне\s+упустите\b|\bторопитесь\b|\bспешите\b|\bуспейте\b"
)

#: Скидки и акции — по словоформам, чтобы «акционерное общество» жило.
_PROMO = re.compile(r"\b(?:скидк\w*|распродаж\w*|промокод\w*|акци(?:я|и|ю|ей|ях|ями)\b)")

#: Отрицание рядом с promo: «скидок и акций у нас нет» — честная оговорка.
_PROMO_NEGATION = re.compile(r"\b(?:нет|не\s+бывает|не\s+делаем|без)\b")

#: Цена: цифры или числительные прописью рядом с деньгами.
_PRICE_DIGITS = re.compile(
    r"\d[\d\s]{0,7}(?:тыс\w*[.\s]*)?(?:₽|руб\w*|р[.\s]|ru[bв]\b|rur\b)", re.IGNORECASE
)
_PRICE_WORDS = re.compile(
    r"(?:цен[аыеу]\b|стоит\b|стоил\w*|обойд\w*)[^.!?\n]{0,40}"
    r"(?:тысяч\w*|полтор\w*|сотн\w*|сто\b|пятьсот|двести|триста|четыреста|рубл\w*)"
)

#: Опьянение и руль.
_DRIVING = re.compile(r"\bпромилле\b|\bза\s+руль\b|\bза\s+рулём\b|\bза\s+рулем\b")
_DRIVING_OK = re.compile(r"\b(?:нельзя|запрещен\w*|не\s+садитесь|не\s+сто\w+т)\b")

#: «Безопасная доза».
_SAFE_DOSE = re.compile(
    r"(?:безопасн|безвредн)\w*[^.!?\n]{0,40}(?:доз\w*|количеств\w*|бокал\w*|употреблен\w*"
    r"|пить|выпить)"
    r"|(?:доз\w*|количеств\w*)[^.!?\n]{0,40}(?:безопасн|безвредн)\w*"
)
_SAFE_DOSE_NEGATION = re.compile(r"\b(?:не\s+существует|не\s+бывает|нет\b|не\s+назв\w*)")

#: Регулярная доза: «бокал в день», «ежедневно». Рядом с похвалой это
#: утверждение о пользе регулярного употребления, даже когда слова
#: «здоровье» в фразе нет.
_DAILY_DOSE = re.compile(r"(?:бокал\w*|стакан\w*|порци\w*)\s+(?:вина\s+)?в\s+день|ежедневн\w*")

#: Ложные контексты пользы: слова, в которых «польза» — не про алкоголь.
_BENEFIT_FALSE_CONTEXT = re.compile(
    r"\bпользовател\w*|\bполезн\w*\s+(?:фильтр\w*|совет\w*|функц\w*|ссылк\w*|подборк\w*)"
)


def _claims_benefit(sentence: str) -> bool:
    """Утверждение о пользе алкоголя — те же четыре ветки, что у «Лозы», по порядку."""
    cleaned = _BENEFIT_FALSE_CONTEXT.sub(" ", sentence)
    has_subject = re.search(_SUBJECT, cleaned) is not None
    if has_subject and re.search(_BENEFIT_STRONG, cleaned):
        return True
    if has_subject and re.search(_BENEFIT_WEAK, cleaned) and re.search(_MEDICAL_OBJECT, cleaned):
        return True
    if re.search(_BENEFIT_STRONG, cleaned) and re.search(_MEDICAL_OBJECT, cleaned):
        # «Этот напиток лечит простуду» без слова «вино» — субъект шире.
        return True
    # «Бокал вина в день полезен» — медицинского объекта нет, но это
    # утверждение о пользе регулярного употребления.
    return bool(has_subject and _DAILY_DOSE.search(cleaned) and re.search(_BENEFIT_WEAK, cleaned))


def _sentence_violations(sentence: str) -> list[str]:
    """Классы нарушений в одном предложении (уже нормализованном)."""
    hits: list[str] = []

    if _claims_benefit(sentence):
        hits.append("польза алкоголя")

    if _PURCHASE.search(sentence):
        hits.append("призыв к покупке")
    if _PROMO.search(sentence) and not _PROMO_NEGATION.search(sentence):
        hits.append("призыв к покупке")

    if _PRICE_DIGITS.search(sentence) or _PRICE_WORDS.search(sentence):
        hits.append("цена рядом с товаром")

    if _DRIVING.search(sentence) and not _DRIVING_OK.search(sentence):
        hits.append("опьянение и вождение")

    if _SAFE_DOSE.search(sentence) and not _SAFE_DOSE_NEGATION.search(sentence):
        hits.append("безопасные дозы")

    return hits


#: Чем заменяется вычеркнутое: пустая замена оставила бы рваный текст.
_REPLACEMENT = "Об этом я не рассказываю."

#: Границы предложений: точки, восклицания, вопросы И переводы строк —
#: маркированный список без точек тоже состоит из отдельных высказываний.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def check(text: str) -> FilterVerdict:
    """Проверяет текст, не меняя его."""
    violations: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(_normalize(text or "")):
        violations.extend(_sentence_violations(sentence))
    unique = sorted(set(violations))
    return FilterVerdict(text=text if not unique else None, violations=unique)


def sanitize(text: str) -> FilterVerdict:
    """Вычёркивает высказывания с нарушениями, остальное оставляет.

    Работает по предложениям и строкам, а не по всему тексту: одна плохая
    фраза не должна уничтожать весь ответ.
    """
    if not text:
        return FilterVerdict(text=text, violations=[])

    # Резать нужно исходный текст, а проверять — нормализованный;
    # предложения режутся одинаково, потому что нормализация не
    # добавляет и не убирает границ.
    original_parts = _SENTENCE_SPLIT.split(text)
    normalized_parts = _SENTENCE_SPLIT.split(_normalize(text))
    if len(original_parts) != len(normalized_parts):
        # Крайне маловероятно, но честнее проверить целиком, чем разъехаться.
        verdict = check(text)
        return verdict if verdict.clean else FilterVerdict(_REPLACEMENT, verdict.violations)

    kept: list[str] = []
    violations: list[str] = []
    for original, normalized in zip(original_parts, normalized_parts, strict=True):
        hits = _sentence_violations(normalized)
        if hits:
            violations.extend(hits)
            logger.warning(
                "Фильтр контента вычеркнул фразу (%s): %r", ", ".join(hits), original[:120]
            )
            continue
        if original.strip():
            kept.append(original.strip())

    if violations and not kept:
        kept = [_REPLACEMENT]

    return FilterVerdict(text=" ".join(kept), violations=sorted(set(violations)))
