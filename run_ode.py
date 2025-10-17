from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController

vector_field = lambda t, y, args: -y
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(dense=True)
stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(
    term,
    solver,
    t0=0,
    t1=3,
    dt0=0.1,
    y0=1,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
)

print(sol.evaluate(t0=1.3))
print(sol.ts)  # DeviceArray([0.   , 1.   , 2.   , 3.    ])
print(sol.ys)  # DeviceArray([1.   , 0.368, 0.135, 0.0498])
