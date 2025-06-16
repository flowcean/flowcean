from typing import cast

import polars as pl
import sympy  # type:ignore[reportMissingTypeStubs]

from .predicate import Predicate


class PolarsPredicate(Predicate):
    """Predicate for Polars DataFrame.

    This predicate allows for two different ways to provide the predicate
    expression:

    1. As a Polars expression. This expression is used directly and must
    evaluate to a single boolean value. For example, the following expression
    checks if the values in the "feature_a" column are in the list [1, 2, 3]
    and if the values in the "feature_b" column are greater than 0:
    ```python
        import polars as pl

        PolarsPredicate(
            pl.col("feature_a").is_in([1, 2, 3]).and_(pl.col("feature_b") > 0),
        )
    ```

    2. As a string. The string is parsed as a Polars expression.
    Any string identifier are replace with the respective feature during
    evaluation. The string expression must evaluate to a single boolean value
    as well. For example, the following expression checks if "feature_a" is
    always greater than "feature_b":
    ```python
        import polars as pl

        PolarsPredicate(
            "feature_a > feature_b",
        )
    ```
    Boolean expressions like `and`, `or`, and `not` are *not* supported by this
    syntax. See `AndPredicate`, `OrPredicate` and `NotPredicate` for combined
    predicates or use the polars expression syntax above.
    """

    def __init__(self, expr: pl.Expr | str) -> None:
        """Initialize the predicate from a polars expression or a string."""
        self.expr = _str_to_pl(expr) if isinstance(expr, str) else expr

    def __call__(
        self,
        input_data: pl.DataFrame | pl.LazyFrame,
        prediction: pl.DataFrame | pl.LazyFrame,
    ) -> bool:
        return cast(
            "bool",
            pl.concat(
                [input_data.lazy(), prediction.lazy()],
                how="horizontal",
            )
            .select(
                self.expr.cast(pl.Boolean).alias("predicate"),
            )
            .lazy()
            .collect()
            .item(),
        )


def _str_to_pl(expression: str) -> pl.Expr:
    sym_expr = sympy.parse_expr(expression, evaluate=False)
    symbols = list(sym_expr.free_symbols)
    lambda_expr = sympy.lambdify(  # type:ignore[reportUnknownVariableType, reportUnknownMemberType]
        symbols,
        sym_expr,
        "math",
        docstring_limit=0,
    )
    return cast(
        "pl.Expr",
        lambda_expr(*[pl.col(str(symbol)) for symbol in symbols]),
    )
