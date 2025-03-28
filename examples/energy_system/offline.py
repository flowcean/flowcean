from flowcean.mosaik.energy_system import EnergySystemOffline
from flowcean.polars import Select, TrainTestSplit


def run_offline() -> None:
    filename = "midasmv_der_my.csv"
    environment = EnergySystemOffline("midasmv_der", filename) | Select(
        [
            "Powergrid-0.0-bus-2.p_mw",
            "Powergrid-0.0-bus-3.p_mw",
            "Powergrid-0.0-bus-2.vm_pu",
            "Powergrid-0.0-bus-3.vm_pu",
        ],
    )
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(environment)


if __name__ == "__main__":
    run_offline()
