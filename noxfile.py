import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "venv"


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
        session.run("coverage", "xml")


@nox.session()
def ruff(session: nox.Session) -> None:
    session.install("ruff==0.2.*")
    session.run("ruff", "check", ".")


@nox.session(python="3.11")
def mypy(session: nox.Session) -> None:
    session.install(".[typecheck]")
    session.run("mypy", "src", "tests")


@nox.session(python="3.11")
def boiler(session: nox.Session) -> None:
    session.install(".[sklearn]")
    session.chdir("examples/boiler/")
    session.run("python", "run.py")


@nox.session(python="3.11")
def failure_time_prediction(session: nox.Session) -> None:
    session.install(".[lightning,sklearn]")
    session.chdir("examples/failure_time_prediction/")
    session.run("agenc", "--experiment", "experiment.yaml")


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
