import matplotlib.pyplot as plt
import polars as pl


def plot_alp_pressures() -> None:
    data = (
        pl.scan_parquet("data/alp_sim_data.parquet")
        .filter(pl.col("active_valve_count") > 0)
        .filter(pl.col("container_weight").is_close(45000, abs_tol=100))
        .head(1)
        .collect(engine="streaming")
        .lazy()
    )
    p_accumulator = data.select(
        pl.col("p_accumulator").list.explode().struct.unnest(),
    ).collect()
    p_accumulator.write_csv("pressure_curve.csv")
    print(f"container_weight: {data.select('container_weight').collect()}")
    print(f"T: {data.select('T').collect()}")
    print(f"Active Valve Count: {data.select('active_valve_count').collect()}")
    print(
        f"p_init: {data.select(pl.col('p_accumulator').list.first()).collect()}"
    )

    p_cylinder1 = data.select(
        pl.col("p_cylinder1").list.explode().struct.unnest(),
    ).collect()
    p_cylinder2 = data.select(
        pl.col("p_cylinder2").list.explode().struct.unnest(),
    ).collect()
    p_cylinder3 = data.select(
        pl.col("p_cylinder3").list.explode().struct.unnest(),
    ).collect()
    p_cylinder4 = data.select(
        pl.col("p_cylinder4").list.explode().struct.unnest(),
    ).collect()

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(
        p_accumulator["time"], p_accumulator["value"], label="p_accumulator"
    )
    # ax.plot(p_cylinder1["time"], p_cylinder1["value"], label="p_cylinder1")
    # ax.plot(p_cylinder2["time"], p_cylinder2["value"], label="p_cylinder2")
    # ax.plot(p_cylinder3["time"], p_cylinder3["value"], label="p_cylinder3")
    # ax.plot(p_cylinder4["time"], p_cylinder4["value"], label="p_cylinder4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure")
    ax.legend()
    plt.show()


def plot_performances() -> None:
    results = pl.read_csv("results_regression_tree.csv")
    fig, ax = plt.subplots(layout="constrained")
    for features in results["features"].unique():
        subset = results.filter(pl.col("features") == features)
        ax.plot(
            subset["depth"],
            subset["MAE"] / 1000.0,
            marker="o",
            label=f"Features: {features}",
        )
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Mean Absolute Error (MAE) [t]")
    ax.set_title("Regression Tree Performance on ALP Data")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot_alp_pressures()
    # plot_performances()
