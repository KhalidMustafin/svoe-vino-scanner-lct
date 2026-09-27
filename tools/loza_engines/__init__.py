"""Движки «Лозы» для сборки данных сомелье — только на сборке, только в подпроцессе.

Пакет «Лозы» называется `app`, как и пакет сканера, поэтому её код сюда перенесён под своим
именем и запускается отдельным процессом (`run.py`, `python -I -S`: в `sys.path` нет корня
репозитория, site-packages и `app` сканера). Сервис этот пакет не импортирует: в работе
`safe_eval` нет, сервис читает готовые `data/somm/*.json` (`app/recommend/somm_data.py`).

Происхождение (репозиторий «Лозы», вне этого репозитория; коммит файла и начало sha256 исходника):

    safe_eval.py      backend/app/core/safe_eval.py             451848e  e322fe5b6284e2fa
    requirements.py   backend/app/recommend/requirements.py     f2b4b66  e00161ea372317f6
    pairing.py        backend/app/recommend/pairing.py          f277717  11c1cacea528e9c6
                      (только оценка пары, см. docstring модуля)
    reference/pairing.json              data/reference/pairing.json   6c999c2  77a8087128001fda
                      (ключи rules и dishes как есть; portal_dish_mapping убран — это таблица
                      связи с категориями портала, а данных портала в продукте нет)
    reference/pairing_rules_fixes.json  data/reference/…             8a83b77  712f9c22b0785890
    reference/dish_dative.json          data/reference/…             d9305cd  70ee202a455ef805
    reference/knowledge.json            data/reference/…             87ce6fe  40a5bad4667c9376
                      (как есть; чистку по 38-ФЗ делает сборка, `scripts/build_somm.py`)

`overlay.json` — наша накладка поверх правил «Лозы»: только условия правил, причины — в файле.
"""
