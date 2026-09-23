"""Export benchmark automata without running their simulations."""

import argparse
from pathlib import Path

from flowcean.hybrid import build_hybrid_system_dot, render_dot_svg
from flowcean.hybrid.benchmarks import all_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg",
        action="store_true",
        help="also render SVG files (requires Graphviz's dot executable)",
    )
    args = parser.parse_args()
    output_dir = Path("outputs") / "automata"
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in all_specs():
        system = spec.factory()
        dot = build_hybrid_system_dot(system)
        stem = spec.name.lower().replace(" ", "_")
        dot_path = output_dir / f"{stem}.dot"
        dot_path.write_text(dot, encoding="utf-8")
        print(f"wrote {dot_path}")
        if args.svg:
            try:
                svg = render_dot_svg(dot)
            except RuntimeError as error:
                parser.exit(status=1, message=f"{error}\n")
            svg_path = output_dir / f"{stem}.svg"
            svg_path.write_text(svg, encoding="utf-8")
            print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
