import logging
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from midas_palaestrai import ArlDefenderObjective
from typing_extensions import override

import flowcean.cli
from flowcean.core import Action, ActiveInterface, Observation
from flowcean.core.metric import ActiveMetric
from flowcean.core.report import Report
from flowcean.core.strategies.active import (
    evaluate_active,
    learn_active,
)
from flowcean.mosaik.energy_system import (
    EnergySystemActive,
)
from flowcean.palaestrai.sac_learner import SACLearner, SACModel

logger = logging.getLogger("energy_example_sac")
END = 5 * 24 * 60 * 60
TEST_END = 10 * 24 * 60 * 60


def run_active(end: int = END, test_end: int = TEST_END) -> None:
    flowcean.cli.initialize()

    # Prepare the required paths
    data_path = (Path(__file__) / ".." / "data").resolve()
    scenario_file = data_path / "midas_scenario.yml"
    output_path = Path.cwd() / "_outputs"

    # Get sensor and actuator IDs
    with (data_path / "actuators.txt").open("r") as f:
        actuator_ids = f.read().splitlines()
    with (data_path / "sensors.txt").open("r") as f:
        sensor_ids = f.read().splitlines()

    learning_params = [
        {
            "update_after": 2000,
            "update_every": 200,
            "batch_size": 100,
            "fc_dims": (256, 256, 256),
        },
        {
            "update_after": 1000,
            "update_every": 100,
            "batch_size": 100,
            "fc_dims": (128, 128),
        },
    ]
    all_learner = []
    training_objectives = []
    test_sim_result_files = []
    reports = []

    for i in range(len(learning_params)):
        # Setup the environment
        environment = EnergySystemActive(
            "agenc_demo_training",
            str(output_path / f"training_results_{i:02d}.csv"),
            scenario_file=str(scenario_file),
            reward_func=calculate_reward,
            end=end,
        )
        try:
            learner = SACLearner(
                actuator_ids,
                sensor_ids,
                ArlDefenderObjective(),
                **learning_params[i],
            )
            learner.setup(environment.action, environment.observation)
            all_learner.append(learner)
        except Exception:
            logger.exception("Failed to load learner")
            environment.shutdown()
            continue

        logger.warning("Starting simulation #%02d ...", (i + 1))
        try:
            model = learn_active(environment, learner)
            training_objectives.append(learner.objective_values)
        except Exception:
            logger.exception("Error during environment operation.")
            continue

        model.save(str(output_path / f"sac_model_{i:02d}"))

        # Prepare evaluation run
        test_results_file = output_path / f"test_results_{i:02d}.csv"
        environment = EnergySystemActive(
            "agenc_demo",
            str(test_results_file),
            scenario_file=str(scenario_file),
            reward_func=calculate_reward,
            end=test_end,
        )
        test_sim_result_files.append(test_results_file)

        logger.warning("Starting evaluation #%02d ...", (i + 1))
        cast("SACModel", model).eval()
        try:
            reports.append(
                evaluate_active(
                    environment,
                    model,
                    [
                        VoltageDeviation(),
                        VoltagePeaks(),
                        VoltageDrops(),
                        LineLoadings(),
                    ],
                ),
            )
        except Exception:
            logger.exception("Error during environment operation.")
            continue

    # Report the training results
    plot_training_results(training_objectives, output_path)

    # Report the evaluation results
    plot_evaluation_results(test_sim_result_files, output_path)

    plot_report(reports, output_path)
    logger.info("Finished!")


def plot_training_results(
    training_objectives: list[list[float]],
    output_path: Path,
) -> None:
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim(0.0, 1.1)
    ax.grid(visible=True)
    for i, vals in enumerate(training_objectives):
        mobj = np.array(vals).mean()
        ax.scatter(
            np.arange(len(vals)),
            vals,
            label=f"Model {i + 1} ({mobj:.5f})",
            s=3,
        )

    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Objective Value")
    plt.title("Objective Comparison of Models")
    plt.savefig(
        str(output_path / "objectives.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_evaluation_results(
    test_sim_result_files: list[str],
    output_path: Path,
) -> None:
    _, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylim(0.8, 1.1)
    ax.grid(visible=True)
    colors = ["blue", "orange", "green", "purple", "red", "black", "yellow"]
    for i, tf in enumerate(test_sim_result_files):
        data = pd.read_csv(tf, index_col=False)

        vdata = data[
            [c for c in data.columns if ("bus" in c and "vm_pu" in c)]
        ]
        n = 100
        vmeans = cast("pd.Series", vdata.mean(axis=1))
        mvmeans = vmeans.rolling(window=n, min_periods=1).mean()
        index = np.arange(len(mvmeans))
        vmins = cast("pd.Series", vdata.min(axis=1))
        vmaxs = cast("pd.Series", vdata.max(axis=1))
        ax.fill_between(
            index,
            vmins,
            vmaxs,
            color=colors[i],
            alpha=0.2,
        )
        ax.plot(
            mvmeans,
            linewidth=1,
            color=colors[i],
            label=f"Model {i + 1}: {n}-step Moving Avg",
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Voltage magnitude p.u.")
    ax.legend()
    plt.title("Average Voltage Comparison of Models")
    plt.savefig(
        str(output_path / "grid_health.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def plot_report(reports: list[Report], output_path: Path) -> None:
    for i, report in enumerate(reports):
        print(f"Model {i + 1}\n{report}")

    header = (
        "| Model | VoltageDeviation | VoltagePeaks "
        "| VoltageDrops | LineLoadings |"
    )
    header_line = (
        "| ----- | ---------------- | ------------ "
        "| ------------ | ------------ |"
    )
    print(header)
    print(header_line)
    lines = []
    for i, report in enumerate(reports):
        lines.append(
            f"| Model {i} | {report['SACModel']['VoltageDeviation']:.5f} "
            f"| {report['SACModel']['VoltagePeaks']:.5f} | "
            f"{report['SACModel']['VoltageDrops']:.5f} | "
            f"{report['SACModel']['LineLoadings']:.5f} |",
        )

        print(lines[-1])
    with (Path(output_path) / "report.md").open("w") as f:
        f.write(f"{header}\n")
        f.write(f"{header_line}\n")
        for line in lines:
            f.write(f"{line}\n")


def calculate_reward(sensors: list) -> list:
    voltages = sorted([s.value for s in sensors if "vm_pu" in s.uid])
    voltage_rewards = [
        ActiveInterface(
            value=voltages[0],
            uid="vm_pu-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=voltages[-1],
            uid="vm_pu-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=median(voltages),
            uid="vm_pu-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=mean(voltages),
            uid="vm_pu-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
        ActiveInterface(
            value=stdev(voltages),
            uid="vm_pu-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=1.5,
        ),
    ]

    lineloads = sorted(
        [s.value for s in sensors if ".loading_percent" in s.uid],
    )

    lineload_rewards = [
        ActiveInterface(
            value=lineloads[0],
            uid="lineload-min",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=lineloads[-1],
            uid="lineload-max",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=median(lineloads),
            uid="lineload-median",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=mean(lineloads),
            uid="lineload-mean",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
        ActiveInterface(
            value=stdev(lineloads),
            uid="lineload-std",
            shape=(),
            dtype=np.float32,
            value_min=0.0,
            value_max=200.0,
        ),
    ]

    return voltage_rewards + lineload_rewards


class VoltageDeviation(ActiveMetric):
    @override
    def __call__(
        self,
        observations: list[Observation],
        action: list[Action],
    ) -> Any:
        values = []
        for observation in observations[1:]:
            vm_pu_mse = [
                (1.0 - cast("float", sensor.value)) ** 2
                for sensor in observation.sensors
                if "vm_pu" in sensor.uid
            ]
            values.append(math.sqrt(sum(vm_pu_mse) / len(vm_pu_mse)))
        return sum(values) / len(values)


class VoltagePeaks(ActiveMetric):
    @override
    def __call__(
        self,
        observations: list[Observation],
        action: list[Action],
    ) -> Any:
        values = []
        for observation in observations[1:]:
            vm_pu_mse = [
                cast("float", sensor.value)
                for sensor in observation.sensors
                if "vm_pu" in sensor.uid
            ]
            values.append(max(vm_pu_mse))
        return max(values) - 1.0


class VoltageDrops(ActiveMetric):
    @override
    def __call__(
        self,
        observations: list[Observation],
        action: list[Action],
    ) -> Any:
        values = []
        for observation in observations[1:]:
            vm_pu_mse = [
                cast("float", sensor.value)
                for sensor in observation.sensors
                if "vm_pu" in sensor.uid
            ]
            values.append(min(vm_pu_mse))
        return 1.0 - min(values)


class LineLoadings(ActiveMetric):
    @override
    def __call__(
        self,
        observations: list[Observation],
        action: list[Action],
    ) -> Any:
        values = []
        for observation in observations[1:]:
            loadings = [
                cast("float", sensor.value)
                for sensor in observation.sensors
                if ("loading_percent" in sensor.uid and "line" in sensor.uid)
            ]
            values.append(sum(loadings) / len(loadings))
        return sum(values) / len(values)


if __name__ == "__main__":
    run_active()
