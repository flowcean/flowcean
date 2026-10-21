import polars as pl
import pytest
from sklearn.tree import DecisionTreeRegressor

from flowcean.core import Lambda
from flowcean.sklearn import SciKitModel
from flowcean.testing import generator
from flowcean.testing.domain import Discrete


@pytest.fixture(params=["combination", "stochastic", "ddtig"])
def cases(request, monkeypatch) -> generator.TestcaseGenerator:
    if request.param == "combination":
        return generator.CombinationGenerator(Discrete("x", [1.0, 2.0]))
    if request.param == "stochastic":
        return generator.StochasticGenerator(
            [Discrete("x", [1.0, 2.0])], test_case_count=2, seed=1
        )

    # Generation is independent of resetting and replaying the resulting rows.
    inputs = pl.DataFrame({"x": [1.0, 2.0]})
    monkeypatch.setattr(generator.TestPipeline, "execute", lambda self: inputs)
    estimator = DecisionTreeRegressor().fit(inputs, [1.0, 2.0])
    return generator.DDTIGenerator(
        SciKitModel(estimator, output_names=["y"]),
        n_testinputs=2,
        test_coverage_criterium="dtc",
        dataset=inputs.with_columns(y=pl.col("x")),
    )


def test_reset_reopens_source_and_existing_views(
    cases: generator.TestcaseGenerator,
) -> None:
    transform = Lambda[pl.LazyFrame, pl.LazyFrame](lambda frame: frame)
    first = cases | transform
    sibling = cases | transform
    nested = first | transform

    assert len(list(first)) == 2
    assert list(cases) == []
    assert list(sibling) == []
    assert list(nested) == []

    for stream in (cases, first, sibling, nested):
        cases.reset()
        assert len(list(stream)) == 2
        assert list(stream) == []
