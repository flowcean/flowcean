"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files.nav import Nav

PROJECT_ROOT: Path = Path(__file__).parent.parent
"""The root directory of the project."""

SRC_DIR: Path = PROJECT_ROOT / "src"
"""The source directory to index for Python files."""

OUTPUT_DIR: Path = Path("reference")
"""The output directory for the generated documentation."""


def process_python_file(nav: Nav, py_file: Path) -> None:
    """Process a Python file to generate documentation.

    Args:
        nav: The navigation object to update.
        py_file: The path to the Python file.
    """
    relative_path = py_file.relative_to(SRC_DIR)
    doc_path = relative_path.with_suffix(".md")
    module_parts = relative_path.with_suffix("").parts

    if module_parts[-1] == "__main__":
        return

    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif module_parts[-1].startswith("_"):
        return

    nav[module_parts] = doc_path.as_posix()
    output_path = OUTPUT_DIR / doc_path
    generate_doc_page(output_path, module_parts)
    mkdocs_gen_files.set_edit_path(
        output_path,
        py_file.relative_to(PROJECT_ROOT),
    )


def generate_doc_page(doc_path: Path, module_parts: tuple[str, ...]) -> None:
    """Generate a documentation page for a module.

    Args:
        doc_path: Path to save the documentation file.
        module_parts: Parts of the module path.
    """
    with mkdocs_gen_files.open(doc_path, "w") as file:
        ident = ".".join(module_parts)
        file.write(f"::: {ident}")


def write_navigation(nav: Nav) -> None:
    """Write the navigation summary to a file."""
    with mkdocs_gen_files.open(OUTPUT_DIR / "SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


nav = Nav()

for py_file in sorted(SRC_DIR.rglob("*.py")):
    process_python_file(nav, py_file)

write_navigation(nav)
