from flowcean.testing import generator
from flowcean.testing.generator.ddtig import (
    ModelHandler,
)
from flowcean.testing.generator.ddtig import (
    TestPipeline as DDTIGTestPipeline,
)


def test_generator_exports_ddtig_support_types() -> None:
    assert generator.ModelHandler is ModelHandler
    assert generator.TestPipeline is DDTIGTestPipeline
