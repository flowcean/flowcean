import shutil
import subprocess

from flowcean.hydra.selector.inspection import SelectorInspection


def _escape_dot_label(label: str) -> str:
    return (
        label.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace(
            "\n",
            "\\n",
        )
    )


def _compact_flow_summary(flow_summary: str) -> str:
    parts = [part.strip() for part in flow_summary.splitlines()]
    return " | ".join(part for part in parts if part)


def _build_split_label(
    feature_name: str,
    threshold: float,
    sample_count: int,
) -> str:
    return f"{feature_name} <= {threshold:.6g}\nraw_samples={sample_count}"


def _build_leaf_label(mode_id: int, flow_summary: str) -> str:
    return f"mode={mode_id}\nflow={_compact_flow_summary(flow_summary)}"


def _invalid_inspection(message: str) -> ValueError:
    return ValueError(f"invalid selector inspection: {message}")


def _validate_child_reference(
    node_id: int,
    child_id: int | None,
    known_node_ids: set[int],
    side: str,
) -> None:
    if child_id is not None and child_id not in known_node_ids:
        message = f"node {node_id} {side}_child_id={child_id} is missing"
        raise _invalid_inspection(message)


def build_selector_dot(inspection: SelectorInspection) -> str:
    leaves_by_id = {leaf.node_id: leaf for leaf in inspection.leaves}
    known_node_ids = {node.node_id for node in inspection.nodes}
    lines = ["digraph Selector {", "  node [shape=box];"]

    for node in inspection.nodes:
        if node.is_leaf:
            leaf = leaves_by_id.get(node.node_id)
            if leaf is None:
                message = f"leaf node {node.node_id} is missing leaf summary"
                raise _invalid_inspection(message)
            label = _build_leaf_label(leaf.mode_id, leaf.flow_summary)
        else:
            if node.feature_name is None or node.threshold is None:
                message = (
                    f"split node {node.node_id} is missing split metadata"
                )
                raise _invalid_inspection(message)
            _validate_child_reference(
                node.node_id,
                node.left_child_id,
                known_node_ids,
                "left",
            )
            _validate_child_reference(
                node.node_id,
                node.right_child_id,
                known_node_ids,
                "right",
            )
            label = _build_split_label(
                node.feature_name,
                node.threshold,
                node.sample_count,
            )

        lines.append(
            f'  node_{node.node_id} [label="{_escape_dot_label(label)}"];',
        )

        if node.left_child_id is not None:
            lines.append(
                f"  node_{node.node_id} -> node_{node.left_child_id};",
            )
        if node.right_child_id is not None:
            lines.append(
                f"  node_{node.node_id} -> node_{node.right_child_id};",
            )

    lines.append("}")
    return "\n".join(lines)


def render_dot_svg(dot_source: str) -> str:
    dot_path = shutil.which("dot")
    if dot_path is None:
        message = (
            "Graphviz 'dot' executable not found; install Graphviz "
            "to render selector SVG output."
        )
        raise RuntimeError(message)

    result = subprocess.run(  # noqa: S603
        [dot_path, "-Tsvg"],
        input=dot_source,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = "Graphviz failed to render selector SVG"
        if stderr:
            message = f"{message}: {stderr}"
        raise RuntimeError(message)

    return result.stdout
