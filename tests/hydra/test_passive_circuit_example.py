import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_passive_circuit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "passive_circuit_run",
        Path("examples/passive_circuit/run.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("svg_error", "expected_message"),
    [
        (RuntimeError("graphviz missing"), "graphviz missing"),
        (OSError("disk full"), "disk full"),
    ],
)
def test_print_selector_outputs_keeps_text_output_when_svg_write_fails(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    svg_error: RuntimeError | OSError,
    expected_message: str,
) -> None:
    module = _load_passive_circuit_module()

    class FailingSelector:
        def __init__(self) -> None:
            self.saved_path: Path | None = None

        def summary_text(self) -> str:
            return "summary"

        def mode_summary_text(self) -> str:
            return "mode-summary"

        def leaf_summary_text(self) -> str:
            return "leaf-summary"

        def tree_text(self) -> str:
            return "tree"

        def save_svg(self, path: Path) -> None:
            self.saved_path = path
            raise svg_error

    selector = FailingSelector()

    module.print_selector_outputs(selector, output_dir=tmp_path)

    assert selector.saved_path == tmp_path / "selector_tree.svg"
    output = capsys.readouterr().out
    assert "selector_summary" in output
    assert "summary" in output
    assert "selector_mode_summary" in output
    assert "mode-summary" in output
    assert "selector_leaf_summary" in output
    assert "leaf-summary" in output
    assert "selector_tree" in output
    assert "tree" in output
    assert "selector_svg_skipped" in output
    assert expected_message in output


def test_parse_args_defaults_live_plot_to_false() -> None:
    module = _load_passive_circuit_module()

    args = module.parse_args([])

    assert args.live_plot is False


def test_parse_args_enables_live_plot_flag() -> None:
    module = _load_passive_circuit_module()

    args = module.parse_args(["--live-plot"])

    assert args.live_plot is True
