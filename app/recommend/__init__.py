"""Слой «после поиска»: справочник для рекомендаций, похожие вина по фактам, фильтр текста.

Чистый Python: без torch, FastAPI и `app.api`. Пакет читает только выгрузку организатора
(`data/gt/gt_tokens.jsonl`), наш словарь групп (`data/catalog/wines.jsonl`, `wine_groups.json`)
и данные сомелье (`data/somm/`); снимка портала в продукте нет (решение 24.09). Пакет ничего
не решает про кадр: slug выбирает сканер, а здесь — что показать рядом с ним.
Импорты пакета проверяет `tests/unit/test_recommend.py::test_package_imports_no_torch_fastapi_or_api`.
"""
