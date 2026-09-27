"""Фильтр контента: рамки ФЗ-38 для любого исходящего текста.

Перенос `Code/backend/tests/test_content_filter.py` «Лозы» без изменений в проверках.
"""

from __future__ import annotations

from app.recommend.content_filter import check, sanitize


class TestForbidden:
    def test_польза_алкоголя_режется(self):
        verdict = sanitize("Красное вино полезно для сердца. Подойдёт к стейку.")
        assert "полезно" not in verdict.text
        assert "Подойдёт к стейку." in verdict.text
        assert verdict.violations

    def test_медицинские_обещания(self):
        assert not check("Бокал вина снижает давление и укрепляет сосуды.").clean

    def test_призыв_к_покупке(self):
        assert not check("Купите это вино сегодня!").clean
        assert not check("Сейчас на него действует скидка.").clean

    def test_цена_с_валютой(self):
        assert not check("Отличный выбор за 1 500 ₽.").clean
        assert not check("Стоит около 2000 рублей.").clean

    def test_безопасная_доза(self):
        assert not check("Один бокал — безвредная доза.").clean


class TestAllowed:
    def test_обычный_ответ_проходит(self):
        text = (
            "Ближе всего к стилю «Шабли» — Ркацители Рислинг: совпадение 88%. "
            "Кислотность выше обычной для белых, дуба нет."
        )
        verdict = sanitize(text)
        assert verdict.clean
        assert verdict.text == text

    def test_отказ_о_здоровье_проходит(self):
        # Сам отказ упоминает тему, но не делает утверждений о пользе.
        text = "Я не отвечаю о влиянии алкоголя на здоровье."
        assert sanitize(text).clean

    def test_балл_дегустации_не_цена(self):
        assert sanitize("Оценка дегустационной комиссии 91 балл из 100.").clean

    def test_пустой_текст(self):
        assert sanitize("").clean
        assert sanitize("").text == ""


class TestWholeTextRemoved:
    def test_целиком_запрещённый_текст_заменяется_оговоркой(self):
        verdict = sanitize("Вино лечит простуду.")
        assert verdict.text == "Об этом я не рассказываю."
        assert verdict.violations


class TestBypassesClosed:
    """Обходы, которые фильтр раньше пропускал: каждый — точный вход, на котором он ошибался."""

    def test_синонимы_пользы(self):
        assert not check("Вино благотворно влияет на организм.").clean
        assert not check("Вино укрепляет здоровье.").clean
        assert not check("Бокал вина укрепляет иммунитет.").clean

    def test_анафора_и_напиток(self):
        assert not check("Это вино особенное. Оно лечит простуду и укрепляет сосуды.").clean
        assert not check("Этот напиток лечит простуду.").clean

    def test_гомоглифы_и_мягкий_перенос(self):
        assert not check("Винo пoлезнo для сердца.").clean  # латинские «o»
        assert not check("Вино по\u00adлезно для сердца.").clean  # мягкий перенос

    def test_цена_во_всех_обличьях(self):
        assert not check("Бутылка стоит 1500 р.").clean
        assert not check("Цена: полторы тысячи рублей.").clean
        assert not check("Обойдётся примерно в 2 тыс. руб.").clean
        assert not check("Стоит 1500 rub.").clean

    def test_промилле_и_руль(self):
        assert not check("Два бокала дадут около 0,6 промилле, потом можно садиться за руль.").clean
        assert not check("Пару бокалов можно выпить и спокойно сесть за руль.").clean

    def test_призыв_в_любых_формах(self):
        assert not check("Обязательно купи это вино.").clean
        assert not check("Это вино стоит купить на праздник.").clean
        assert not check("Рекомендую приобрести бутылку к ужину.").clean
        assert not check("Не упустите шанс попробовать — торопитесь!").clean

    def test_список_без_точек_режется_построчно(self):
        verdict = sanitize("- Аромат вишни\n- Вино снижает давление\n- Танины мягкие")
        assert verdict.violations
        assert "Аромат вишни" in verdict.text
        assert "Танины мягкие" in verdict.text
        assert "давление" not in verdict.text


class TestFalsePositivesClosed:
    """Честные фразы, которые первая версия резала."""

    def test_пользователи_и_полезные_фильтры(self):
        assert sanitize("Пользователи каталога высоко оценили это вино.").clean
        assert sanitize("Полезные фильтры каталога помогут выбрать напиток по сладости.").clean

    def test_акционерное_общество(self):
        assert sanitize("«Абрау-Дюрсо» — акционерное общество из Краснодарского края.").clean

    def test_отказ_от_скидок_не_реклама(self):
        assert sanitize("Скидок и акций у нас нет — мы информационный сервис.").clean

    def test_законная_оговорка_о_дозе_живёт(self):
        assert sanitize("Не существует безопасной дозы алкоголя.").clean
        assert sanitize("Безопасного количества алкоголя не бывает.").clean

    def test_запрет_садиться_за_руль_живёт(self):
        assert sanitize("После вина нельзя садиться за руль.").clean

    def test_укрепляет_позиции_не_медицина(self):
        assert sanitize("Хозяйство укрепляет позиции среди российских виноделен.").clean


class TestBenefitCasesClosed:
    """Формулировки пользы, которые проходили фильтр насквозь.

    Ст. 21 ФЗ-38 запрещает утверждать, что алкоголь полезен. Проверка
    шла по словам «польза/пользы/пользе», а самая частая формулировка
    стоит в винительном падеже — «приносит пользу здоровью» — и не
    ловилась вовсе.
    """

    def test_приносит_пользу(self):
        assert not sanitize("Вино приносит пользу здоровью.").clean
        assert not sanitize("Красное вино приносит пользу сердцу.").clean

    def test_беглая_гласная_в_кратком_прилагательном(self):
        """«Полезен» не подходит под основу «полезн»: гласная беглая."""
        assert not sanitize("Вино полезен для сердца.").clean

    def test_регулярная_доза(self):
        """«Бокал вина в день полезен» — польза без слова «здоровье»."""
        assert not sanitize("Бокал вина в день полезен.").clean
        assert not sanitize("Вино ежедневно улучшает самочувствие.").clean

    def test_польза_без_алкоголя_живёт(self):
        assert sanitize("Совет полезен для выбора закуски.").clean
        assert sanitize("Подавайте бокал вина в день праздника.").clean
        assert sanitize("Вино приносит радость за столом.").clean


class TestPortFix:
    """Отличие переноса от «Лозы»: ветка «польза/пользу/пользы/пользе» снова работает.

    В оригинале после `польз[ауые]` стоял литеральный backspace (0x08) вместо `\\b`, и без
    другого слова пользы («приносит», «полезно») фраза проходила.
    """

    def test_польза_без_глагола_режется(self):
        assert not check("Вино — польза для сердца.").clean
        assert not check("Красное вино: пользы для сосудов больше.").clean

    def test_пользователи_по_прежнему_живут(self):
        assert check("Пользователи каталога любят это вино за сердцевину вкуса.").clean
