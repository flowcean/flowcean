import logging
from collections.abc import Iterable
from functools import reduce
from typing import Literal, override

import polars as pl
import sympy

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)

type FilterMode = Literal["and", "or"]


class Filter(Transform):
    """Filter an environment based on one or multiple expressions."""

    predicates: list[pl.Expr]

    def __init__(
        self,
        expressions: str | Iterable[str],
        *,
        filter_mode: FilterMode = "and",
    ) -> None:
        """Initializes the Filter transform.

        Args:
            expressions: String expressions used to filter the environment.
                Records that do not match the expression are discarded.
                Standard comparison and mathematical operations are supported
                within the expressions. Features can be accessed by there name.
                If multiple expressions are provided, `filter_mode' decides
                whether one or all must be fulfilled.
            filter_mode: Define how multiple filter expressions are to be
                processed. If set to `and` all expressions need to be fulfilled
                    by a record. If set to `or` one or more expressions need to
                    be fulfilled.
        """
        if isinstance(expressions, str):
            self.predicates = [str_to_pl(expressions)]
        else:
            self.predicates = [str_to_pl(expr) for expr in expressions]
        self.filter_mode = filter_mode

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.filter(
            self.predicates
            if len(self.predicates) == 1
            else reduce(
                (lambda x, y: x & y)
                if self.filter_mode == "and"
                else (lambda x, y: x | y),
                self.predicates,
            ),
        )


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
