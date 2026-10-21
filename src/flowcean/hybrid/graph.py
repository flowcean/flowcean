"""Deterministic DOT export for declared hybrid-system structure."""

from collections.abc import Mapping

from matplotlib.colors import to_hex

from ._graphviz import _dot_style, _escape_dot_label
from ._graphviz import render_dot_svg as render_dot_svg
from ._plot_style import INK, location_color_map, pale_color
from .hybrid_system import (
    HybridSystem,
    Location,
    _callback_label,
    display_label,
)


def build_hybrid_system_dot(
    system: HybridSystem,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_event_labels: bool = True,
    show_reset_labels: bool = True,
    show_direction: bool = False,
    show_entry_policy: bool = False,
) -> str:
    """Export the complete declared automaton as deterministic DOT text.

    Locations and transitions retain declaration order, including isolated
    locations, parallel transitions, and self-loops. A point and incoming arrow
    mark the initial location. Node IDs are positional, independent of display
    labels; reordering locations changes their IDs.

    Labels follow the model's display-label precedence: explicit labels, then
    callback names, then positional fallbacks for unnamed objects. Labels are
    treated as literal text, not DOT markup or mathematical expressions.
    No simulation or callback evaluation is performed, and Graphviz need not
    be installed.
    Positive fixed delays are annotated in model time units; delay callbacks
    are annotated by name, without evaluation.

    Args:
        system: Constructed hybrid system to visualize.
        location_colors: Optional colors keyed by Location identity, using
            Matplotlib color names or hex values, as in native plots. Defaults
            follow this system's declaration order; matching colors across
            different artifacts do not establish corresponding locations.
        show_event_labels: Annotate transitions with event-surface labels.
        show_reset_labels: Annotate transitions that have resets with labels.
        show_direction: Include the event surface's zero-crossing direction.
        show_entry_policy: Include each transition's surface-entry policy.

    Returns:
        DOT source suitable for saving or passing to ``render_dot_svg``.
    """
    location_indices = {
        id(location): index for index, location in enumerate(system.locations)
    }
    colors = location_color_map(system.locations, location_colors)
    lines = [
        "digraph HybridSystem {",
        *_dot_style(),
        "  rankdir=LR;",
        f'  start [shape=point, fillcolor="{INK}"];',
    ]

    for index, location in enumerate(system.locations):
        label = _escape_dot_label(
            display_label(location, fallback=f"location_{index}"),
        )
        color = to_hex(colors[location])
        lines.append(
            f'  location_{index} [label="{label}", color="{color}", '
            f'fillcolor="{pale_color(color)}"];'
        )

    initial_index = location_indices[id(system.initial_location)]
    lines.append(f"  start -> location_{initial_index};")

    for index, transition in enumerate(system.transitions):
        source = location_indices[id(transition.source)]
        target = location_indices[id(transition.target)]
        annotations: list[str] = []
        if show_event_labels:
            annotations.append(
                "event: "
                + display_label(
                    transition.event_surface, fallback=f"event_{index}"
                ),
            )
        if show_reset_labels and transition.reset is not None:
            annotations.append(
                "reset: "
                + display_label(transition.reset, fallback=f"reset_{index}"),
            )
        if show_direction:
            annotations.append(
                f"direction: {transition.event_surface.direction.name.lower()}"
            )
        if show_entry_policy:
            annotations.append(
                f"entry_policy: {transition.entry_policy.value}"
            )
        if callable(transition.delay):
            name = _callback_label(transition.delay) or f"delay_{index}"
            annotations.append(f"delay: {name}")
        elif transition.delay > 0:
            annotations.append(f"delay: {transition.delay:g}")
        edge = f"  location_{source} -> location_{target}"
        if annotations:
            label = _escape_dot_label("\n".join(annotations))
            edge += f' [label="{label}"]'
        lines.append(f"{edge};")

    lines.append("}")
    return "\n".join(lines)
