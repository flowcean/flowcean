import logging

import polars as pl
from sympy import lambdify
from sympy.core import Expr
from sympy.parsing.sympy_parser import parse_expr

from agenc.core import Transform

logger = logging.getLogger(__name__)


class ComputeFeature(Transform):
    """Computes a new feature based on existing features.

    Computes a new feature based on existing features in the DataFrame.
    The values of the features are available under their respective names.
    Furthermore, all common mathematical operations and many functions can
    be used.

    Args:
        expression (str): Mathematical expression used to compute the new
            features value.

        output_feature_name (str): Name of the newly computed feature.

    The below example shows the usage of a `ComputeFeature` transform in an
    `experiment.yaml` file. Assuming the loaded data is represented by the
    table:

    .. list-table:: Original data
        :header-rows: 1

        *   - x_a
            - y_a
            - x_b
            - y_b
        *   - 0
            - 0
            - 1
            - 2
        *   - 1
            - 1
            - 4
            - 7

    The following transform can be used to calculate the euclidean
    distance between the points `a` with `x_a` and `y_a` and `b` and
    store the result in a new feature named `distance`.

    .. highlight:: yaml
    .. code-block:: yaml

        transforms:
            - classpath: agenc.transforms.ComputeFeature
                arguments:
                    expression: "sqrt((x_a - x_b)^2 + (y_a - y_b)^2)"
                    output_feature_name: distance
            - ...

    The resulting Dataframe after the transform is:

    .. list-table:: Transformed data
        :header-rows: 1

        *   - x_a
            - y_a
            - x_b
            - y_b
            - distance
        *   - 0
            - 0
            - 1
            - 2
            - 2.236
        *   - 1
            - 1
            - 4
            - 7
            - 6.708

    Note that the used features (`x_a`, `y_a`, `x_b` and `y_b`) are still
    present in the Dataframe. To remove them use :class:`.SelectTransform`.
    """

    def __init__(self, expression: str, output_feature_name: str) -> None:
        self.expression_string = expression
        self.output_feature_name = output_feature_name

        sym_expression: Expr = parse_expr(self.expression_string)
        free_symbols = list(sym_expression.free_symbols)

        self.column_fnc = lambdify(free_symbols, sym_expression)(
            *[pl.col(str(symbol)) for symbol in free_symbols]
        )

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug(
            f"Computing new feature {self.output_feature_name}",
            f" = {self.expression_string}",
        )

        return (
            data.lazy()
            .with_columns(self.column_fnc.alias(self.output_feature_name))
            .collect()
        )
