import subprocess
import sys
import textwrap


def test_generator_package_import_does_not_import_ddti_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import sys

        importlib.import_module("flowcean.testing")
        before_modules = set(sys.modules)

        module = importlib.import_module("flowcean.testing.generator")

        imported_top_level_modules = {
            name.split(".", maxsplit=1)[0]
            for name in set(sys.modules) - before_modules
        }
        assert "river" not in imported_top_level_modules, (
            imported_top_level_modules
        )
        assert "torch" not in imported_top_level_modules, (
            imported_top_level_modules
        )
        assert module.CombinationGenerator is not None
        assert module.StochasticGenerator is not None
        assert module.TestcaseGenerator is not None
        """,
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
