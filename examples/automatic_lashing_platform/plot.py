import matplotlib.pyplot as plt
import polars as pl

from flowcean.environments.parquet import ParquetDataLoader

data = ParquetDataLoader("./alp_sim_data.parquet").observe()
selected = (
    data.sort("containerWeight")
    .select(
        [
            pl.col("p_cylinder1")
            .list.eval(pl.element().struct.field("value"))
            .list.to_struct(),
            "containerWeight",
        ]
    )
    .unnest("p_cylinder1")
)

print(selected)

# for row in range(10):
#     plt.plot(selected.row(row))
pressure = selected.select(pl.exclude("containerWeight"))
plt.plot(pressure.row(0), label=selected.select("containerWeight").row(0)[0])
plt.plot(
    pressure.row(8001), label=selected.select("containerWeight").row(8001)[0]
)
plt.plot(
    pressure.row(16000), label=selected.select("containerWeight").row(16000)[0]
)

plt.legend()
plt.show()
