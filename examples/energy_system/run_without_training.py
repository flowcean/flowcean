import logging
from pathlib import Path

from run_with_sac import calculate_reward

import flowcean.cli
from flowcean.core.strategies.active import (
    StopLearning,
)
from flowcean.mosaik.energy_system import (
    EnergySystemActive,
)
from flowcean.palaestrai.sac_model import SACModel

logger = logging.getLogger("energy_example_sac")


def run_active() -> None:
    flowcean.cli.initialize()
    environment = EnergySystemActive(
        "midasmv_der",
        "my_results.csv",
        reward_func=calculate_reward,
    )

    model = SACModel.load(str(Path.cwd() / "_outputs" / "sac_model_only"))

    try:
        while True:
            observations = environment.observe()
            action = model.predict(observations)
            environment.act(action)
            environment.step()
    except StopLearning:
        pass

    environment.shutdown()

    logger.info("Finished!")


if __name__ == "__main__":
    run_active()
