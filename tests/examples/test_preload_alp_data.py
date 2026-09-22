from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def test_preloader_preserves_scalar_sensor_series_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preloader_path = (
        Path(__file__).resolve().parents[2]
        / "examples/automatic_lashing_platform/utils/preload_alp_data.py"
    )
    spec = spec_from_file_location("preload_alp_data", preloader_path)
    assert spec is not None
    assert spec.loader is not None
    preload_alp_data = module_from_spec(spec)
    spec.loader.exec_module(preload_alp_data)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pl.DataFrame(
        {
            "t": [0.0, 0.5, 1.0],
            "p_accumulator": [10.0, 11.0, 12.0],
            "p_sensor": pl.Series([2, 3, 4], dtype=pl.Int32),
        },
    ).write_parquet(data_dir / "simulation.parquet")
    metadata = pl.DataFrame(
        {"container_weight": [1000.0], "active_valve_count": [2], "T": [20.0]},
    )
    metadata.write_json(data_dir / "simulation.json")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        preload_alp_data.flowcean.cli, "initialize", lambda: None
    )

    preload_alp_data.main()

    expected = pl.DataFrame(
        {
            "p_accumulator": [
                [
                    {"time": 0.0, "value": 10.0},
                    {"time": 0.5, "value": 11.0},
                    {"time": 1.0, "value": 12.0},
                ],
            ],
            "p_sensor": [
                [
                    {"time": 0.0, "value": 2},
                    {"time": 0.5, "value": 3},
                    {"time": 1.0, "value": 4},
                ],
            ],
        },
        schema={
            "p_accumulator": pl.List(
                pl.Struct({"time": pl.Float64, "value": pl.Float64}),
            ),
            "p_sensor": pl.List(
                pl.Struct({"time": pl.Float64, "value": pl.Int32}),
            ),
        },
    ).hstack(metadata)
    assert_frame_equal(
        pl.read_parquet(tmp_path / "alp_sim_data.parquet"), expected
    )
