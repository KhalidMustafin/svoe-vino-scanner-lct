# Пул обучения и разработки фундаментального трека (`trainpool.*`)

Собран `research/2026-09-26_fund/build_trainpool.py` (ветка `fund`) на снимке данных
`fund/frozen_data` (индекс `ccd3a01f`, gt `899d5db3`, словарь `c2c7befd`), только CPU. Роли
наборов, запреты и приёмка — [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md).
**kr-test в пуле нет**: `protocol.assert_no_test` проверил каждый id и каждую картинку при сборке
и при слиянии.

## Состав и база продукта

| набор (`set`) | кадры | группы вин | «то же вино» продукта | строго | CV top-1 | верный в top-20 | кадров вин kr-test |
|---|---|---|---|---|---|---|---|
| `catalog_v2` | 353 | 226 | 315 (89,2 %) | 307 | 264 | 344 | 57 |
| `kr_dev` | 308 | 245 | **266 (86,4 %)** | 256 | 249 | 299 | 0 |
| `kr_dev_sp` (same_packshot вин kr-dev) | 156 | 148 | 149 | 140 | 141 | 156 | 0 |
| `ooc_v2` (вне каталога, отрицательные) | 408 | 376 | — | — | — | — | 0 |
| `pairs` (студия Роскачества) | 358 | 355 | 315 (88,0 %); dev 133/149, test 182/209 | 315 | 306 | 356 | 102 |
| `pairs_phone` (их «телефонные» копии) | 358 | 355 (те же) | 284 (79,3 %); dev 119/149, test 165/209 | 284 | 275 | 349 | 102 |
| **всего** | **1 941** | **1 170** | | | | | |

- Ответы v2, kr_dev и kr_dev_sp совпали с итоговой сборкой 25.09 (`goal26/final_*.json`) кадр в
  кадр: 353/353, 308/308, 156/156. Слой выбора по записи (`resolve_row`) = ответ сервиса на всех
  кадрах пула (`replay_equal`).
- Студийные пары: у 2 кадров `pairs` и 1 кадра `pairs_phone` чтение `garbage`/`loop` — ответ по
  Э2 (CV top-1), как в сервисе.
- `ooc_v2`: 409 в meta, M608 пропущен — нет чтения в кэше стенда (как и у Э6: 408).
- `kr_test_wine` — кадр показывает вино kr-test (другой снимок, не из магазина). Такие кадры
  разрешены (EVAL_PROTOCOL §3), флаг — для разбора «seen / unseen».
- Студийные пары: 149 из 358 — половина `dev` прежнего стенда; на 298 кадрах `dev` (`pairs` +
  `pairs_phone`, 291 с верным в top-20) обучен ранкер `-goal` (`goal_split` = `dev`). На них
  признаки и ответы продукта — в выборке обучения. `goal_split` = `test` (209 + 209) ранкер не
  видел; их прежний однократный замер (85,2 / 74,6 %, 17.09) к нынешнему индексу не относится.

## Файлы

- `trainpool.jsonl` — строка на кадр:
  - `id`, `set`, `source`, `photo`, `image` (путь к картинке), `wine_source` (id вина набора);
  - метка: `slug`, `acceptable`, `in_catalog`, `same_packshot`, `group` (компонента «того же
    вина» по всему пулу: slug ∪ acceptable ∪ wine_id; вне каталога — `ooc:<вино>`), `kr_test_wine`,
    у пар `goal_split`;
  - CV: `cv` — top-20 `[slug, счёт zmax, вид эталона, ранг]`, `margin` — отрыв по всему каталогу;
  - чтение: `read` = `{fields: LabelFields JSON, raw_text}` после разбора кодом сервиса (Э6),
    `vlm_status`, `vlm_lines`;
  - слой выбора: `rank_slugs`, `rank_scores` (счёт ранкера `-goal` по убыванию), `p_top1`,
    `resolve` (доказательства сервиса: `ambiguous_rerank` — H5, `bonus_flip_blocked` — P1,
    `fallback` — Э2), `answer`, `outcome`, `p_answer`;
  - счёт: `correct` («то же вино»), `strict`, `cv_correct`, `true_rank_cv` (ранг верного в CV
    top-20 или `null`), `replay_equal`, `row` — номер строки в `trainpool.npz`.
- `trainpool.npz`:
  - `ids` (N);
  - `features` (N, 20, 34) float32 — 34 признака `-goal` (`feature_names`, `resolve-features/3`)
    для кандидатов в порядке CV; `feat_slugs` (N, 20); строки без кандидата — NaN и `""`;
  - `qemb` (N, 4, 1152) float16 и `qemb_names` = `bottle, label, band, full`.
- `trainpool_summary.json` — таблица выше; `trainpool_parts/` — части по наборам.

## Что лежит в векторах запроса

`qemb` — ровно то, что сервис считает на кадре: `decode_on_backgrounds(байты, [BACKGROUND])` →
`from_query(кадр, None)` → четыре окна `bottle`, `label`, `band`, `full` → SigLIP2 so400m
(`google/siglip2-so400m-patch14-384`, pooler, CPU float32) → единичная норма, хранение float16.
TTA-видов и детектора нет. Индекс: 6 312 строк = 2 103 slug × виды эталона `bottle, label,
band` (у `merlo-litavshhuk` два эталона — 6 строк), float16, единичная норма.

Счёт CV сервиса (`per_slug="zmax"`, `app/features/index.py: align_pairs`): косинусы 6 312 × 4 →
для каждой пары «окно запроса × вид эталона» стандартизация по всем строкам вида и возврат на шкалу
запроса (`M + s·(c − mu)/sd`) → максимум по окнам для строки → максимум по строкам slug → top-20
и отрыв top-1 − top-2 по всему каталогу. Параметров нет.

Векторы v2, kr и ooc — `acc_plan/retrieval/qemb_*.npz` (25.09, CPU); студийных пар —
`fund/protocol/qemb_pairs.npz` (`qemb_pairs.py`, 25.09): прежний CV пар (`runs/goal/iter20`) шёл
по индексу до правки эталонов 23.09. «Телефонная» копия — сохранённый JPEG
`runs/pairs-phone-images` (те же байты, что читал OCR), а не порча на лету.

## Как повторить конвейер с другим CV или другим слоем выбора (CPU)

```python
import sys; sys.path.insert(0, "research/2026-09-26_fund")
import numpy as np, protocol as P, replay as R
from app.features.embedder import unit_rows

rows = P.jsonl(P.PROTOCOL / "trainpool.jsonl"); z = np.load(P.PROTOCOL / "trainpool.npz")
P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])   # обязательно
rp = R.FundReplay()                       # сервис на снимке; model_path=... — другая модель выбора

r = rows[0]
rp.resolve_row(r)                         # ответ продукта по записи (= r["answer"])

# другой счёт CV: вектор счёта по всем slug в порядке rp.svc.index.slug_order
Q = unit_rows(z["qemb"][r["row"]].astype(np.float32))
scores, best_row = rp.cv_scores(Q)        # счёт сервиса (zmax) — точка отсчёта
scores = scores + my_delta                # свой счёт: другая модель, слияние, SIFT-проверка…
cands, margin = rp.candidates_from_scores(scores, best_row)
v = R.VisualResult(candidates=cands, margin=margin)
rp.resolve(v, R.read_of(r["read"]), vlm_status=r["vlm_status"], vlm_lines=r["vlm_lines"])["answer"]
```

- **Другая модель выбора** того же формата (`LogisticRanker`, `app/resolve/learned.py`, свои
  веса на тех же 34 признаках) — `R.FundReplay(model_path=путь)`: ранкер, H5 и P1 берут её.
- **Свои признаки или свой слой выбора** — `app.resolve.features.query_features(visual,
  {"vlm35": read}, rp.svc.attrs, top_k=20)` даёт `QueryFeatures` (признаки 34 столбцов по
  кандидатам), дальше — своя модель; `app.resolve.ambiguous.rerank_ambiguous` и
  `block_bonus_flip` — правила H5 и P1 поверх любой `LogisticRanker`. Готовые 34 признака уже
  лежат в `trainpool.npz` (`features`).
- **Кадр целиком** (новые векторы запроса, картинка): `rp.scan(байты, qvec)`; при
  `rp.cv_fn = функция(Q) -> (кандидаты, отрыв)` поиск идёт через неё. Чтение — из кэша стенда
  по sha1 кадра; студийным кадрам записанное чтение подаётся через `rp.forced_reading`.
- Проверка и пример — `replay_demo.py`: на v2 + kr-dev (661 кадр) `resolve_row` = продукт 661/661,
  `cv_scores → candidates_from_scores` = выдача сервиса 661/661 бит в бит, пример «CV только по
  окну full» — 580 против 581 «то же вино», 38 сменившихся ответов; ~30 мс на кадр.
- GroupKFold — по полю `group` (sklearn нет; свой разрез по группам). kr-dev и kr_dev_sp одного
  вина — одна группа.
