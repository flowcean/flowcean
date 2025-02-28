import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import reduce

import polars as pl
import sympy
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class FilterExpr(ABC):
    """Expression to be used in a Filter transform."""

    @abstractmethod
    def get(self) -> pl.Expr:
        """Get the polars expression for this filter."""

    def __call__(self) -> pl.Expr:
        return self.get()


class Filter(Transform):
    """Filter an environment based on one or multiple expressions.

    This transform allows to filter an environment based on or multiple boolean
    expressions.
    Assuming the input environment is given by

     t | N  | x
    ---|----|---
     1 | 10 | 0
     2 | 12 | 1
     3 |  5 | 2
     4 | 15 | 1
     5 | 17 | 0

    The following transformation can be used to filter the environment so that
    the result contains only records where `x=1`:

    ```python
        Filter("x == 1")
    ```

    The result dataset after applying the transform will be

     t | N  | x
    ---|----|---
     2 | 15 | 1
     4 | 12 | 1

    To only get records where `x=1` *and* `t > 3` the filter expression

    ```python
    Filter(And(["x == 1", "t > 3"]))
    ```

    can be used.

    To filter all records where `x=1` *and* `t > 3` *or* `N < 15` use

    ```python
    Filter(And(["x == 1", Or(["t > 3", "N < 15"])]))
    ```

    """

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
            self.predicate = _str_to_pl(expression)
        else:
            self.predicate = expression()

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.filter(self.predicate)


def _str_to_pl(expression: str) -> pl.Expr:
    sym_expr = sympy.parse_expr(expression, evaluate=False)
    symbols = list(sym_expr.free_symbols)
    lambda_expr = sympy.lambdify(
        symbols,
        sym_expr,
        "math",
        docstring_limit=0,
    )
    return lambda_expr(*[pl.col(str(symbol)) for symbol in symbols])


class CollectionExpr(FilterExpr):
    expr_collection: Iterable[pl.Expr]

    def __init__(
        self,
        expressions: str | FilterExpr | Iterable[str | FilterExpr],
    ) -> None:
        if not isinstance(expressions, Iterable):
            expressions = [expressions]
        self.expr_collection = (
            expression()
            if isinstance(expression, FilterExpr)
            else _str_to_pl(expression)
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
            else _str_to_pl(expression)
        )

    def get(self) -> pl.Expr:
        return self.expression.not_()
