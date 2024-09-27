import environment
import polars as pl
from midas_palaestrai import ArlDefenderObjective
from sac_learner import SACLearner

import flowcean.cli
from flowcean.strategies.active import learn_active


def main() -> None:
    flowcean.cli.initialize_logging(log_level="INFO")

    env = environment.MosaikEnvironment(
        start_date="2017-01-01 00:00:00+0100",
        end=1 * 24 * 60 * 60,
        seed=None,
        params={"name": "midasmv_der"},
    )
    actuator_ids = [
        "Pysimmods-0.Photovoltaic-0.q_set_mvar",
        "Pysimmods-0.Photovoltaic-1.q_set_mvar",
        "Pysimmods-0.Photovoltaic-2.q_set_mvar",
    ]
    sensor_ids = [
        "Powergrid-0.0-bus-1.vm_pu",
        "Powergrid-0.0-bus-1.in_service",
        "Powergrid-0.0-bus-2.vm_pu",
        "Powergrid-0.0-bus-3.vm_pu",
        "Powergrid-0.0-bus-4.vm_pu",
        "Powergrid-0.0-bus-5.vm_pu",
        "Powergrid-0.0-bus-6.vm_pu",
    ]
    learner = SACLearner(actuator_ids, sensor_ids, ArlDefenderObjective())

    model = learn_active(env, learner)
    print(model.summary())

    data = {"objective": learner.objectives}
    for row in learner.rewards:
        for item in row:
            data.setdefault(item.uid, [])
            data[item.uid].append(item.value)

    data = pl.DataFrame(data)
    print(pl.DataFrame(data))
    data.write_csv("energy_system_results.csv")


if __name__ == "__main__":
    main()
