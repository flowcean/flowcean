import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import reduce
from typing import Literal, override

import polars as pl
import sympy

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)

type FilterMode = Literal["and", "or"]


class FilterExpr(ABC):
    @abstractmethod
    def get(self) -> pl.Expr:
        """Get the polars expression for this filter."""

    def __call__(self) -> pl.Expr:
        return self.get()


class CollectionExpr(FilterExpr):
    expr_collection: Iterable[pl.Expr]

    def __init__(
        self, expressions: str | FilterExpr | Iterable[str | FilterExpr]
    ) -> None:
        if not isinstance(expressions, Iterable):
            expressions = [expressions]
        self.expr_collection = (
            expression()
            if isinstance(expression, FilterExpr)
            else str_to_pl(expression)
            for expression in expressions
        )


class And(CollectionExpr):
    def get(self) -> pl.Expr:
        return reduce(
            lambda x, y: x.and_(y),
            self.expr_collection,
        )


class Or(CollectionExpr):
    def get(self) -> pl.Expr:
        return reduce(
            lambda x, y: x.or_(y),
            self.expr_collection,
        )


class Not(FilterExpr):
    expression: pl.Expr

    def __init__(self, expression: str | FilterExpr) -> None:
        self.expression = (
            expression()
            if isinstance(expression, FilterExpr)
            else str_to_pl(expression)
        )

    def get(self) -> pl.Expr:
        return self.expression.not_()


class Filter(Transform):
    """Filter an environment based on one or multiple expressions."""

    predicate: pl.Expr

    def __init__(
        self,
        expression: str | FilterExpr,
    ) -> None:
        """Initializes the Filter transform.

        Args:
            expression: String or filter expression used to filter the
                environment. Records that do not match the expression are
                discarded. Standard comparison and mathematical operations are
                supported within the expressions. Features can be accessed by
                there name.
        """
        if isinstance(expression, str):
            self.predicate = str_to_pl(expression)
        else:
            self.predicate = expression()

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(self.predicate)


def str_to_pl(expression: str) -> pl.Expr:
    sym_expr = sympy.parse_expr(expression, evaluate=False)
    symbols = list(sym_expr.free_symbols)
    lambda_expr = sympy.lambdify(
        symbols,
        sym_expr,
        "math",
        docstring_limit=0,
    )
    return lambda_expr(*[pl.col(str(symbol)) for symbol in symbols])
