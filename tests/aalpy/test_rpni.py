from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from runpy import run_path

import polars as pl
import pytest
from aalpy.automata import MealyMachine, MooreMachine
from polars.testing import assert_frame_equal

from flowcean.aalpy import (
    RPNIMealyLearner,
    RPNIMealyModel,
    RPNIMooreLearner,
    RPNIMooreModel,
)
from flowcean.core import Model, learn_offline
from flowcean.polars import DataFrame


def frame(name, words, dtype: pl.DataType | type[pl.DataType] = pl.String):
    return pl.DataFrame(pl.Series(name, words, dtype=pl.List(dtype)))


def labels(word, *, moore=False):
    state = 0
    result = [state] if moore else []
    for symbol in word:
        state ^= symbol == "a"
        result.append(state)
    return result


@pytest.fixture(params=[False, True], ids=["mealy", "moore"])
def trained(request):
    moore = request.param
    words = [list(word) for word in product("ab", repeat=3)]
    inputs = frame("commands", words)
    outputs = frame(
        "responses", [labels(w, moore=moore) for w in words], pl.Int8
    )
    learner = RPNIMooreLearner() if moore else RPNIMealyLearner()
    model = learner.learn(inputs.lazy(), outputs.lazy())
    return model, inputs, outputs, moore


def test_fit_and_full_word_prediction(trained):
    model, inputs, outputs, moore = trained
    assert isinstance(model, RPNIMooreModel if moore else RPNIMealyModel)
    assert isinstance(model.automaton, MooreMachine if moore else MealyMachine)
    assert isinstance(model, Model)
    assert not isinstance(model, MealyMachine | MooreMachine)
    assert_frame_equal(model.predict(inputs).collect(), outputs)
    with pytest.raises(AttributeError):
        model.automaton = model.automaton  # pyright: ignore[reportAttributeAccessIssue]


def test_learn_offline(trained):
    _, inputs, outputs, moore = trained
    learner = RPNIMooreLearner() if moore else RPNIMealyLearner()
    model = learn_offline(
        DataFrame(inputs.hstack(outputs)),
        learner,
        ["commands"],
        ["responses"],
    )
    assert_frame_equal(model.predict(inputs).collect(), outputs)


def test_empty_words_and_empty_batch(trained):
    model, inputs, outputs, moore = trained
    assert model.predict(frame("commands", [[]])).collect().to_dicts() == [
        {"responses": [0] if moore else []},
    ]
    assert_frame_equal(
        model.predict(inputs.clear()).collect(), outputs.clear()
    )


def test_independent_repeated_and_concurrent_predictions(trained):
    model, _, _, moore = trained
    model.automaton.current_state = model.automaton.states[-1]
    saved_state = model.automaton.current_state
    words = [["a"], ["a", "b"], ["b"], ["a"], []]
    inputs = frame("commands", words)
    expected = frame(
        "responses", [labels(w, moore=moore) for w in words], pl.Int8
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: model.predict(inputs).collect(), range(8))
        )
    for result in results:
        assert_frame_equal(result, expected)
    assert model.automaton.current_state is saved_state


def test_undefined_input_and_failure_do_not_change_state(trained):
    model, inputs, outputs, _ = trained
    saved_state = model.automaton.current_state
    with pytest.raises(ValueError, match=r"Undefined transition.*position 1"):
        model.predict(frame("commands", [["a", "unknown"]]))
    assert model.automaton.current_state is saved_state
    assert_frame_equal(model.predict(inputs).collect(), outputs)


def test_missing_transition_for_known_alphabet_symbol():
    model = RPNIMooreLearner().learn(
        frame("commands", [["a", "b"]]),
        frame("responses", [[0, 1, 2]], pl.Int64),
    )
    assert not model.automaton.is_input_complete()
    with pytest.raises(ValueError, match=r"Undefined transition.*'b'"):
        model.predict(frame("commands", [["b"]]))


def test_persistence(trained, tmp_path):
    model, inputs, outputs, _ = trained
    path = tmp_path / "automaton.fml"
    model.save(path)
    restored = Model.load(path)
    assert type(restored) is type(model)
    assert_frame_equal(restored.predict(inputs).collect(), outputs)


@pytest.mark.parametrize(
    ("symbols", "dtype"),
    [
        (["tea", "coffee"], pl.String),
        ([False, True], pl.Boolean),
        ([1, 2], pl.UInt32),
        ([1.5, 2.5], pl.Float32),
    ],
)
def test_symbol_types_and_silent_training(symbols, dtype, capsys):
    inputs = frame("a", [symbols], dtype)
    outputs = frame("b", [symbols], dtype)
    model = RPNIMealyLearner().learn(inputs, outputs)
    assert_frame_equal(model.predict(inputs).collect(), outputs)
    assert capsys.readouterr().out == ""


def test_prediction_rejects_wrong_column(trained):
    model, inputs, _, _ = trained
    with pytest.raises(ValueError, match="Expected input column 'commands'"):
        model.predict(inputs.rename({"commands": "other"}))


@pytest.mark.parametrize(
    ("learner", "outputs"),
    [
        (RPNIMealyLearner(), [[True]]),
        (RPNIMooreLearner(), [[False, True]]),
    ],
)
def test_prediction_rejects_different_symbol_dtype(learner, outputs):
    model = learner.learn(
        frame("commands", [[True]], pl.Boolean),
        frame("responses", outputs, pl.Boolean),
    )
    with pytest.raises(
        ValueError,
        match="Expected input symbols with dtype Boolean, got Int64",
    ):
        model.predict(frame("commands", [[1]], pl.Int64))


@pytest.mark.parametrize("learner", [RPNIMealyLearner, RPNIMooreLearner])
@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (pl.DataFrame(), "exactly one"),
        (pl.DataFrame({"a": [[1]], "b": [[1]]}), "exactly one"),
        (pl.DataFrame({"a": [1]}), "lists"),
        (pl.DataFrame({"a": [[[1]]]}), "Symbols must"),
        (pl.DataFrame({"a": [[{"value": 1}]]}), "exactly numeric"),
        (
            pl.DataFrame({"a": [[{"time": "now", "value": 1}]]}),
            "exactly numeric",
        ),
        (
            pl.DataFrame({"a": [[{"time": 0, "value": 1, "extra": 2}]]}),
            "exactly numeric",
        ),
        (frame("a", [None]), "null traces"),
        (frame("a", [[None]]), "symbols must"),
        (frame("a", [[float("nan")]], pl.Float64), "symbols must"),
        (frame("a", [[float("inf")]], pl.Float64), "symbols must"),
        (
            pl.DataFrame({"a": [[{"time": float("nan"), "value": 1}]]}),
            "timestamps must",
        ),
        (
            pl.DataFrame(
                {"a": [[{"time": None, "value": 1}]]},
                schema={
                    "a": pl.List(
                        pl.Struct({"time": pl.Float64, "value": pl.Int64})
                    )
                },
            ),
            "timestamps must",
        ),
        (
            pl.DataFrame(
                {"a": [[None]]},
                schema={
                    "a": pl.List(
                        pl.Struct({"time": pl.Float64, "value": pl.Int64})
                    )
                },
            ),
            "timestamps must",
        ),
    ],
)
def test_malformed_shapes(learner, side, invalid, message):
    valid = frame("a", [["x"]])
    with pytest.raises(ValueError, match=message):
        learner().learn(
            invalid if side == "input" else valid,
            invalid if side == "output" else valid,
        )


@pytest.mark.parametrize("learner", [RPNIMealyLearner, RPNIMooreLearner])
def test_row_count_mismatch_and_no_rows(learner):
    with pytest.raises(ValueError, match="same nonzero number"):
        learner().learn(frame("a", [["x"]]), frame("b", [["x"], ["x"]]))
    with pytest.raises(ValueError, match="same nonzero number"):
        learner().learn(frame("a", []), frame("b", []))


@pytest.mark.parametrize(
    ("learner", "outputs", "message"),
    [
        (RPNIMealyLearner, [["x", "y"]], "Mealy trace 0 requires"),
        (RPNIMooreLearner, [["x"]], "including the initial-state output"),
    ],
)
def test_malformed_lengths(learner, outputs, message):
    with pytest.raises(ValueError, match=message):
        learner().learn(frame("a", [["x"]]), frame("b", outputs))


def test_mealy_training_requires_observations():
    with pytest.raises(ValueError, match="at least one nonempty"):
        RPNIMealyLearner().learn(frame("a", [[]]), frame("b", [[]]))


def test_moore_can_learn_only_initial_output():
    model = RPNIMooreLearner().learn(frame("a", [[]]), frame("b", [["ready"]]))
    assert model.predict(frame("a", [[]])).collect().to_dicts() == [
        {"b": ["ready"]}
    ]


@pytest.mark.parametrize(
    ("learner", "words", "outputs"),
    [
        (RPNIMealyLearner, [["a", "b"], ["a"]], [[0, 1], [1]]),
        (RPNIMooreLearner, [["a"], ["b"]], [[0, 1], [1, 1]]),
    ],
)
def test_conflicting_prefixes(learner, words, outputs, capsys):
    with pytest.raises(ValueError, match="Conflicting outputs"):
        learner().learn(frame("a", words), frame("b", outputs, pl.Int64))
    assert capsys.readouterr().out == ""


def test_timestamp_ordering_is_stable_and_independent():
    inputs = pl.DataFrame(
        {
            "commands": [
                [
                    {"time": 2, "value": "a"},
                    {"time": 0, "value": "a"},
                    {"time": 1, "value": "b"},
                    {"time": 1, "value": "a"},
                ]
            ]
        }
    )
    outputs = pl.DataFrame(
        {
            "responses": [
                [
                    {"time": 40, "value": 1},
                    {"time": 10, "value": 1},
                    {"time": 30, "value": 0},
                    {"time": 20, "value": 1},
                ]
            ]
        }
    )
    model = RPNIMealyLearner().learn(inputs, outputs)
    expected = frame("responses", [[1, 1, 0, 1]], pl.Int64)
    assert_frame_equal(model.predict(inputs).collect(), expected)
    assert_frame_equal(
        model.predict(frame("commands", [["a", "b", "a", "a"]])).collect(),
        expected,
    )
    assert model.predict(
        frame("commands", [["a", "b"]])
    ).collect().to_dicts() == [{"responses": [1, 1]}]


def test_moore_timestamped_initial_output():
    inputs = pl.DataFrame({"a": [[{"time": 10, "value": "toggle"}]]})
    outputs = pl.DataFrame(
        {
            "b": [
                [
                    {"time": 10, "value": "on"},
                    {"time": 0, "value": "off"},
                ]
            ]
        }
    )
    model = RPNIMooreLearner().learn(inputs, outputs)
    assert_frame_equal(
        model.predict(inputs).collect(), frame("b", [["off", "on"]])
    )
    assert_frame_equal(
        model.predict(frame("a", [[]])).collect(), frame("b", [["off"]])
    )


def test_coffee_machine_pipeline_without_external_data(
    tmp_path, monkeypatch, capsys
):
    import flowcean.cli

    example = (
        Path(__file__).resolve().parents[2] / "examples/coffee_machine/run.py"
    )
    module = run_path(str(example))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for index in range(5):
        pl.DataFrame(
            {"t": [2, 0, 1], "input": [0, 0, 1], "output": [1, 1, 0]}
        ).write_csv(data_dir / f"trace{index}.csv")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(flowcean.cli, "initialize", lambda: None)
    module["main"]()
    assert "1.0" in capsys.readouterr().out
