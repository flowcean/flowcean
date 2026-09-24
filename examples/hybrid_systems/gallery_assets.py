"""Generate static figures for the hybrid-system gallery.

Run directly with ``python examples/hybrid_systems/gallery_assets.py``. Graphviz's
``dot`` is required for automata; Matplotlib's noninteractive Agg backend is used
for traces. Layout of Graphviz SVGs depends on the installed Graphviz version.
"""

import argparse
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scenarios import SCENARIOS, WIND_TURBINE, Scenario

from flowcean.hybrid import (
    HybridSystem,
    Trace,
    build_hybrid_system_dot,
    plot_locations,
    plot_trace,
    render_dot_svg,
    simulate,
)
from flowcean.hybrid.benchmarks import wind_turbine_power
from flowcean.hybrid.hybrid_system import display_label


@dataclass(frozen=True)
class Mode:
    """A declared location's display name and shared figure color."""

    name: str
    label: str
    color: str


@dataclass(frozen=True)
class Profile:
    """Presentation settings for a configured gallery scenario."""

    slug: str
    scenario: Scenario
    sample_dt: float
    modes: tuple[Mode, ...]

    @property
    def colors(self) -> dict[str, str]:
        """Map model location names to the same colors in both figures."""
        return {mode.name: mode.color for mode in self.modes}


@dataclass(frozen=True)
class Palette:
    """Theme-wide figure colors; mode colors do not change with theme."""

    background: str
    foreground: str
    grid: str
    line: str


PALETTES = {
    "light": Palette("#f8fbfc", "#18323d", "#b5cbd3", "#164b65"),
    "dark": Palette("#102832", "#edf7fa", "#52717c", "#a9e1f2"),
}

PROFILES = (
    Profile(
        "thermostat",
        next(s for s in SCENARIOS if s.name == "Thermostat"),
        0.02,
        (
            Mode("heating", "Heating", "#b75c12"),
            Mode("cooling", "Cooling", "#167f92"),
        ),
    ),
    Profile(
        "bouncing_ball",
        next(s for s in SCENARIOS if s.name == "Bouncing Ball"),
        0.005,
        (Mode("flight", "Flight", "#4763c7"),),
    ),
    Profile(
        "wind_turbine",
        WIND_TURBINE,
        0.1,
        (
            Mode("no_generation", "No generation", "#9a4b21"),
            Mode("gradual_generation", "Gradual generation", "#9a6c04"),
            Mode("below_rated_power", "Below rated power", "#237d72"),
            Mode(
                "approaching_rated_power", "Approaching rated power", "#7253ad"
            ),
            Mode("rated_power", "Rated power", "#b03860"),
        ),
    ),
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "docs/assets/hybrid_systems"
)


def profile_for(slug: str) -> Profile:
    """Select a named gallery profile."""
    try:
        return next(profile for profile in PROFILES if profile.slug == slug)
    except StopIteration as error:
        raise ValueError(f"Unknown profile: {slug}") from error


def _checked_modes(profile: Profile, system: HybridSystem) -> None:
    declared = [display_label(location) for location in system.locations]
    configured = [mode.name for mode in profile.modes]
    if declared != configured or len(set(configured)) != len(configured):
        raise ValueError(
            f"{profile.slug}: configured modes {configured!r} do not match "
            f"declared locations {declared!r}"
        )


def styled_dot(profile: Profile, system: HybridSystem, theme: str) -> str:
    """Style the library's declared DOT graph without changing its topology."""
    _checked_modes(profile, system)
    palette = PALETTES[theme]
    original = build_hybrid_system_dot(system, show_direction=True)
    lines = []
    for line in original.splitlines():
        compact_line = line
        if " -> " in line:
            # Retain event names/directions without crowded metadata prefixes.
            compact_line = line.replace('label="event: ', 'label="', 1)
            compact_line = compact_line.replace(r"\ndirection: ", r"\n", 1)
            if '[label="' in compact_line:
                # Leave room between opposing edge labels, not only their boxes.
                compact_line = compact_line.replace('label="', 'label="  ', 1)
                compact_line = compact_line.replace(r"\n", r"  \n  ")
                compact_line = compact_line.replace('"];', '  "];')
        lines.append(compact_line)
    compact = "\n".join(lines)
    # Graphviz default attributes apply only to nodes/edges created afterwards.
    # Insert them before the library's first node rather than at the end.
    defaults = (
        f'  graph [bgcolor="{palette.background}", pad="0.22", '
        f'ranksep="0.65", nodesep="0.35", '
        f'fontname="DejaVu Sans", fontcolor="{palette.foreground}"];\n'
        f'  node [shape=box, style="rounded,filled", penwidth=1.5, '
        f'fontname="DejaVu Sans", fontsize=13, margin="0.18,0.12"];\n'
        f'  edge [color="{palette.foreground}", '
        f'fontcolor="{palette.foreground}", fontname="DejaVu Sans", '
        f"fontsize=10, arrowsize=0.7];\n"
        f'  start [shape=point, color="{palette.foreground}", '
        f'fillcolor="{palette.foreground}"];'
    )
    dot = compact.replace("  start [shape=point];", defaults, 1)
    if profile.slug == "wind_turbine":
        dot = dot.replace("  rankdir=LR;", "  rankdir=TB;", 1)
    overrides = []
    for index, mode in enumerate(profile.modes):
        label = mode.label.replace(" ", r"\n")
        overrides.append(
            f'  location_{index} [label="{label}", '
            f'color="{mode.color}", fillcolor="{mode.color}", '
            'fontcolor="#ffffff"];'
        )
    return dot[:-1] + "\n" + "\n".join(overrides) + "\n}"


def _axes_style(
    ax: Axes, palette: Palette, *, time_span: tuple[float, float]
) -> None:
    ax.set_facecolor(palette.background)
    ax.tick_params(colors=palette.foreground, labelsize=10)
    ax.xaxis.label.set_color(palette.foreground)
    ax.yaxis.label.set_color(palette.foreground)
    for spine in ax.spines.values():
        spine.set_color(palette.grid)
    ax.grid(color=palette.grid, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(*time_span)


def _mode_legend(fig: Figure, profile: Profile, palette: Palette) -> None:
    handles = [
        Patch(facecolor=mode.color, label=mode.label) for mode in profile.modes
    ]
    legend = fig.legend(
        handles=handles,
        loc="outside lower center",
        ncols=2 if len(handles) < 5 else 3,
        frameon=False,
        fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(palette.foreground)


def _thermostat_trace(
    fig: Figure,
    profile: Profile,
    system: HybridSystem,
    trace: Trace,
    palette: Palette,
) -> None:
    ax = fig.subplots()
    plot_trace(
        trace,
        dims=[0],
        location_colors=profile.colors,
        show_events=False,
        ax=ax,
    )
    # plot_trace uses the library's exact event snapshots and NaN reset breaks.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.lines[0].set(color=palette.line, linewidth=1.8, label="Temperature")
    assert trace.u is not None
    target = trace.u[:, 0]
    half_band = system.parameters["hysteresis"] / 2
    ax.plot(
        trace.t,
        target,
        color=palette.foreground,
        ls=":",
        lw=1.5,
        label="Target",
    )
    ax.plot(
        trace.t,
        target + half_band,
        color=profile.colors["heating"],
        ls="--",
        lw=1.2,
        label="Switch off heating",
    )
    ax.plot(
        trace.t,
        target - half_band,
        color=profile.colors["cooling"],
        ls="--",
        lw=1.2,
        label="Switch on heating",
    )
    ax.set_ylabel("Temperature")
    ax.set_xlabel("Time")
    legend = ax.legend(
        handles=ax.lines[:4],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
        frameon=False,
        fontsize=10,
    )
    _axes_style(ax, palette, time_span=profile.scenario.t_span)
    for text in legend.get_texts():
        text.set_color(palette.foreground)


def _ball_trace(
    fig: Figure, profile: Profile, trace: Trace, palette: Palette
) -> None:
    axes = fig.subplots(2, 1, sharex=True)
    for ax, dim, label in zip(
        axes, (0, 1), ("Height (m)", "Velocity (m/s)"), strict=True
    ):
        plot_trace(
            trace,
            dims=[dim],
            location_colors=profile.colors,
            show_events=False,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.lines[0].set(color=palette.line, linewidth=1.7, label="_nolegend_")
        ax.set_ylabel(label)
        _axes_style(ax, palette, time_span=profile.scenario.t_span)
    velocity = axes[1]
    resets = [event for event in trace.events if event.reset is not None]
    for index, event in enumerate(resets):
        velocity.plot(
            [event.time, event.time],
            [event.state_before[1], event.state_after[1]],
            linestyle="--",
            color=profile.colors["flight"],
            lw=1.2,
            marker="o",
            markersize=3,
            label="Bounce reset" if index == 0 else "_nolegend_",
        )
    if resets:
        legend = velocity.legend(
            handles=[velocity.lines[1]],
            loc="lower right",
            frameon=False,
            fontsize=10,
        )
        for text in legend.get_texts():
            text.set_color(palette.foreground)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("Time (s)")


def _wind_trace(
    fig: Figure,
    profile: Profile,
    system: HybridSystem,
    trace: Trace,
    palette: Palette,
) -> None:
    assert trace.u is not None
    signals = (
        ("Wind speed (m/s)", trace.u[:, 0]),
        ("Rotor speed (rpm)", trace.x[:, 0] * 60 / (2 * np.pi)),
        ("Blade pitch (degrees)", np.rad2deg(trace.x[:, 3])),
        ("Tower displacement (m)", trace.x[:, 1]),
        (
            "Generator mechanical\npower (MW)",
            wind_turbine_power(trace, parameters=system.parameters) / 1e6,
        ),
    )
    axes = fig.subplots(5, 1, sharex=True)
    for ax, (label, values) in zip(axes, signals, strict=True):
        ax.plot(trace.t, values, color=palette.line, lw=1.6)
        plot_locations(
            trace, ax=ax, location_colors=profile.colors, alpha=0.20
        )
        ax.set_ylabel(label)
        _axes_style(ax, palette, time_span=profile.scenario.t_span)
    rated_mw = system.parameters["rated_mechanical_power"] / 1e6
    rated = axes[-1].axhline(
        rated_mw,
        color=palette.foreground,
        ls="--",
        lw=1.1,
        label=f"Rated shaft power ({rated_mw:.2f} MW)",
    )
    legend = axes[-1].legend(
        handles=[rated], loc="lower right", frameon=False, fontsize=9
    )
    for text in legend.get_texts():
        text.set_color(palette.foreground)
    axes[-1].set_xlabel("Time (s)")


def trace_svg(
    profile: Profile, system: HybridSystem, trace: Trace, theme: str
) -> str:
    """Render one static trace SVG from the already simulated scenario."""
    palette = PALETTES[theme]
    size = {
        "thermostat": (9, 4.8),
        "bouncing_ball": (9, 6),
        "wind_turbine": (9, 10.5),
    }[profile.slug]
    with mpl.rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": f"hybrid-{profile.slug}-trace-{theme}",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": palette.background,
            "axes.facecolor": palette.background,
            "text.color": palette.foreground,
            "savefig.facecolor": palette.background,
        }
    ):
        fig = plt.figure(figsize=size, layout="constrained")
        try:
            if profile.slug == "thermostat":
                _thermostat_trace(fig, profile, system, trace, palette)
            elif profile.slug == "bouncing_ball":
                _ball_trace(fig, profile, trace, palette)
            else:
                _wind_trace(fig, profile, system, trace, palette)
            _mode_legend(fig, profile, palette)
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg", metadata={"Date": None})
            # Match the repository's whitespace checks without changing paths.
            return "".join(
                f"{line.rstrip()}\n" for line in buffer.getvalue().splitlines()
            )
        finally:
            plt.close(fig)


def render_profile(profile: Profile) -> tuple[dict[str, str], str]:
    """Simulate once and render both themes of each figure."""
    system = profile.scenario.factory()
    _checked_modes(profile, system)
    trace = simulate(
        system,
        t_span=profile.scenario.t_span,
        input_stream=profile.scenario.input_stream,
        sample_dt=profile.sample_dt,
        rtol=1e-7,
        atol=1e-9,
    )
    assets = {}
    for theme in PALETTES:
        stem = profile.slug
        assets[f"{stem}-automaton-{theme}.svg"] = render_dot_svg(
            styled_dot(profile, system, theme)
        )
        assets[f"{stem}-trace-{theme}.svg"] = trace_svg(
            profile, system, trace, theme
        )
    visited = ", ".join(dict.fromkeys(str(mode) for mode in trace.location))
    result = (
        f"{profile.slug}: dt={profile.sample_dt:g}, span={profile.scenario.t_span}, "
        f"states={trace.x.shape[1]}, declared modes={len(system.locations)}, "
        f"transitions={len(system.transitions)}, samples={len(trace.t)}, "
        f"events={len(trace.events)}, visited={visited}"
    )
    return assets, result


def main(argv: list[str] | None = None) -> int:
    """Write or check selected SVG files, without altering files in check mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[p.slug for p in PROFILES])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check", action="store_true", help="compare without writing"
    )
    args = parser.parse_args(argv)
    selected = (profile_for(args.profile),) if args.profile else PROFILES
    stale = []
    try:
        for profile in selected:
            assets, result = render_profile(profile)
            print(result)
            for name, content in assets.items():
                path = args.output_dir / name
                if args.check:
                    if (
                        not path.is_file()
                        or path.read_bytes() != content.encode("utf-8")
                    ):
                        stale.append(str(path))
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    print(f"wrote {path}")
    except RuntimeError as error:
        parser.exit(1, f"Figure generation failed: {error}\n")
    if stale:
        for path in stale:
            print(f"stale asset (missing or different): {path}")
        return 1
    if args.check:
        print("All selected SVG assets are current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
