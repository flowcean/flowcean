import environment
from sac_learner import SACLearner

import flowcean.cli
from flowcean.core import ActiveEnvironment, ActiveLearner, Model
from flowcean.strategies.active import StopLearning, learn_active


def main() -> None:
    flowcean.cli.initialize_logging()

    env = environment.MosaikEnvironment(
        start_date="2017-01-01 00:00:00+0100",
        end=1 * 6 * 60 * 60,
        seed=None,
        params={"name": "midasmv_der"},
    )

    learner = SACLearner()

    model = learn_active(
        env,
        learner,
    )
    print(model)
    print(learner.rewards)


if __name__ == "__main__":
    main()
