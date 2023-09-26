import nox


@nox.session(python=["3.10", "3.11"])
def test(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    try:
        session.run(
            "coverage",
            "run",
            "--source",
            "src/agenc/",
            "--module",
            "pytest",
        )
    finally:
        session.run("coverage", "report")


@nox.session()
def ruff(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session()
def black(session: nox.Session) -> None:
    session.install("black")
    session.run("black", "--diff", "--check", "src/", "tests/")


@nox.session(python="3.11")
def mypy(session: nox.Session) -> None:
    session.install(".[typecheck]")
    session.run("mypy", "src", "tests")


@nox.session(python="3.11")
def docs(session: nox.Session) -> None:
    session.install(".[docs]")
    session.run(
        "sphinx-build",
        "-b",
        "spelling",
        "-W",
        "docs/",
        "docs/_build/html",
    )
    session.run(
        "sphinx-build",
        "-b",
        "html",
        "-W",
        "docs/",
        "docs/_build/html",
    )
