import jax.numpy as jnp
from boiler import BoilerNoTime

import flowcean.cli
import flowcean.utils
from flowcean.ode import rollout

config = flowcean.cli.initialize()
flowcean.utils.initialize_random(config.seed)

mode0 = "heating"
t0 = 0.0
x0 = jnp.array([15.785017])
t1 = 10.0
dt0 = 0.01

system = BoilerNoTime(**config.boilernotime.system)
traces = system.simulate(
    mode0=mode0,
    x0=x0,
    t0=t0,
    t1=t1,
    dt0=dt0,
)
data = rollout(traces, dt=0.1)
data.write_csv("out.csv")
