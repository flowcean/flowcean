import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "venv"


@nox.session(python=["3.10", "3.11", "3.12"])
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
            *session.posargs,
        )
    finally:
        session.run("coverage", "report")
        session.run("coverage", "xml")


@nox.session()
def ruff(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "check", ".")


@nox.session()
def mypy(session: nox.Session) -> None:
    session.install("mypy")
    session.install(".[typecheck]")
    session.run("mypy", "src", "tests")
    session.run("mypy", "examples/automatic_lashing_platform")
    session.run("mypy", "examples/boiler")
    session.run("mypy", "examples/coffee_machine")
    session.run("mypy", "examples/failure_time_prediction")


@nox.session()
def pyright(session: nox.Session) -> None:
    session.install("pyright")
    session.install(".[sklearn,lightning,grpc,test]")
    session.run("pyright", "src", "tests", "examples")


@nox.session()
def boiler(session: nox.Session) -> None:
    session.install(".[sklearn]")
    session.chdir("examples/boiler/")
    session.run("python", "run.py")


@nox.session()
def failure_time_prediction(session: nox.Session) -> None:
    session.install(".[lightning,sklearn]")
    session.chdir("examples/failure_time_prediction/")
    session.run("python", "run.py")


@nox.session()
def automatic_lashing_platform(session: nox.Session) -> None:
    session.install(".[sklearn]")
    session.chdir("examples/automatic_lashing_platform/")
    session.run("python", "run.py")


@nox.session()
def linear_data(session: nox.Session) -> None:
    session.install(".[lightning,sklearn]")
    session.chdir("examples/linear_data/")
    session.run("python", "run.py")


@nox.session()
def docs(session: nox.Session) -> None:
    session.install(".[docs]")
    session.run(
        "sphinx-build",
        "-E",
        "-a",
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
