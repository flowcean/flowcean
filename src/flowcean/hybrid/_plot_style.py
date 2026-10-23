"""Private visual baseline for hybrid plots and diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb

if TYPE_CHECKING:
    from .hybrid_system import Location

PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
FONT_FAMILY = "DejaVu Sans"
INK = "#17212B"
BORDER = "#697586"
LINE_WIDTH = 2.0
ANNOTATION_SIZE = 8
BAND_ALPHA = 0.22
_NODE_FILL_ALPHA = 0.36
LEGEND_FONT = {"family": FONT_FAMILY, "size": 9}


def location_color_map(
    locations: Sequence[Location],
    overrides: Mapping[Location, str] | None,
) -> dict[Location, str]:
    """Colors belong to location identities within one declared artifact."""
    colors = {
        location: PALETTE[index % len(PALETTE)]
        for index, location in enumerate(locations)
    }
    if overrides is not None:
        colors.update(overrides)
    return colors


def pale_color(color: str) -> str:
    """Blend diagram colors with white, stronger than plot background bands."""
    red, green, blue = (
        1 - _NODE_FILL_ALPHA + _NODE_FILL_ALPHA * value
        for value in to_rgb(color)
    )
    return to_hex((red, green, blue))


def style_axes(ax: Axes) -> None:
    """Style axes furniture only, leaving data artists and global rc alone."""
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#E5E7EB", linewidth=0.6, linestyle="-", alpha=1)
    ax.tick_params(colors=INK, width=0.8, labelfontfamily=FONT_FAMILY)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.9)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_color(INK)
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontweight("normal")
