"""Безопасное вычисление логических выражений из данных — перенос «Лозы».

Источник — `Code/backend/app/core/safe_eval.py` (451848e) без изменений логики: поменялись только
аннотации типов под Python 3.12 и добавлено свойство `names`. Правила сочетаний хранятся в
справочнике строками вида ``dish.fat >= 3.5 and wine.acidity >= 3.5``. `eval` здесь недопустим:
справочник — данные, а данные не исполняются как код. Выражение разбирается в AST и обходится
интерпретатором с белым списком узлов: сравнения, логика, поля двух известных объектов,
литералы, операции над множествами и шесть чистых функций. Узел вне списка — явная ошибка, а не
тихое выполнение.

Модуль работает только в подпроцессе сборки `scripts/build_somm.py`; сервис его не импортирует.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any


class ExpressionError(ValueError):
    """Выражение не разобралось или содержит запрещённую конструкцию."""


#: Узлы, разрешённые в выражении. Всё остальное отвергается.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.BinOp,
    ast.BitAnd,
    ast.BitOr,
    ast.Sub,
    ast.Add,
    ast.Mult,
    ast.Div,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Set,
    ast.Tuple,
    ast.List,
    ast.Call,
)

#: Функции, доступные выражениям: закрытый список чистых вычислений над числами и коллекциями.
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "len": len,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
}


class SafeExpression:
    """Скомпилированное выражение, готовое к многократному вычислению."""

    __slots__ = ("_names", "_source", "_tree")

    def __init__(self, source: str) -> None:
        self._source = source
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"Не разобрать выражение {source!r}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ExpressionError(
                    f"Недопустимая конструкция {type(node).__name__} в выражении {source!r}"
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise ExpressionError(f"Запрещено обращение к {node.attr!r}")
            if isinstance(node, ast.Call):
                # Вызвать можно только функцию из белого списка и только по прямому имени.
                if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                    name = getattr(node.func, "id", type(node.func).__name__)
                    raise ExpressionError(f"Вызов {name!r} запрещён в выражении {source!r}")
                if node.keywords:
                    raise ExpressionError("Именованные аргументы в выражениях не поддерживаются")

        self._tree = tree
        # Имена окружения, которые читает выражение: сборка по ним понимает, что условие блюда
        # не смотрит на вино и его можно вычислить один раз на блюдо.
        self._names = frozenset(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCTIONS
        )

    @property
    def source(self) -> str:
        return self._source

    @property
    def names(self) -> frozenset[str]:
        """Имена окружения в выражении (`dish`, `wine`), без функций белого списка."""
        return self._names

    def __repr__(self) -> str:  # pragma: no cover - отладочное
        return f"SafeExpression({self._source!r})"

    def evaluate(self, context: Mapping[str, Any]) -> Any:
        """Вычисляет выражение в окружении вида ``{"dish": dish, "wine": wine}``."""
        return _Evaluator(context).visit(self._tree.body)

    def matches(self, context: Mapping[str, Any]) -> bool:
        """Истинно ли выражение."""
        return bool(self.evaluate(context))


class _Evaluator:
    """Обходчик разрешённого подмножества AST."""

    __slots__ = ("_context",)

    def __init__(self, context: Mapping[str, Any]) -> None:
        self._context = context

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, f"_visit_{type(node).__name__}", None)
        if method is None:
            raise ExpressionError(f"Узел {type(node).__name__} не поддерживается")
        return method(node)

    # --- литералы и имена ------------------------------------------------
    def _visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def _visit_Name(self, node: ast.Name) -> Any:
        if node.id not in self._context:
            raise ExpressionError(f"Неизвестное имя {node.id!r}")
        return self._context[node.id]

    def _visit_Attribute(self, node: ast.Attribute) -> Any:
        target = self.visit(node.value)
        try:
            return getattr(target, node.attr)
        except AttributeError as exc:
            raise ExpressionError(f"У объекта нет поля {node.attr!r}") from exc

    def _visit_Set(self, node: ast.Set) -> set[Any]:
        return {self.visit(item) for item in node.elts}

    def _visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        return tuple(self.visit(item) for item in node.elts)

    def _visit_List(self, node: ast.List) -> list[Any]:
        return [self.visit(item) for item in node.elts]

    # --- операции --------------------------------------------------------
    def _visit_BoolOp(self, node: ast.BoolOp) -> Any:
        # Вычисление ленивое, как в Python: правая часть не трогается, если левая всё решила.
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = self.visit(value)
            if result:
                return result
        return result

    def _visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ExpressionError(f"Унарная операция {type(node.op).__name__} запрещена")

    def _visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.BitAnd):
            return self._as_set(left) & self._as_set(right)
        if isinstance(node.op, ast.BitOr):
            return self._as_set(left) | self._as_set(right)
        if isinstance(node.op, ast.Sub):
            if isinstance(left, set | frozenset) or isinstance(right, set | frozenset):
                return self._as_set(left) - self._as_set(right)
            return left - right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ExpressionError("Деление на ноль в выражении правила")
            return left / right
        raise ExpressionError(f"Операция {type(node.op).__name__} запрещена")

    def _visit_Call(self, node: ast.Call) -> Any:
        assert isinstance(node.func, ast.Name)  # проверено при компиляции
        function = _ALLOWED_FUNCTIONS[node.func.id]
        return function(*(self.visit(arg) for arg in node.args))

    def _visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            if not self._compare(operator, left, right):
                return False
            left = right
        return True

    # --- служебное -------------------------------------------------------
    @staticmethod
    def _compare(operator: ast.cmpop, left: Any, right: Any) -> bool:
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
        if isinstance(operator, ast.In):
            return left in right
        if isinstance(operator, ast.NotIn):
            return left not in right
        raise ExpressionError(f"Сравнение {type(operator).__name__} запрещено")

    @staticmethod
    def _as_set(value: Any) -> frozenset[Any]:
        """Приводит значение к множеству: теги блюда пишутся ``dish.flavor_tags & {'beef'}``."""
        if isinstance(value, set | frozenset):
            return frozenset(value)
        if isinstance(value, list | tuple):
            return frozenset(value)
        raise ExpressionError(f"Значение {value!r} не привести к множеству")
