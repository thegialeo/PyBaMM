import time

import pybamm

# Using Modelica-style reg_sqrt for j0 calculation (implemented in Chen2020.py)
# The tolerance sets the regularization width: delta = tol * c_max
# reg_sqrt(x, delta) returns 0 for x <= 0, smoothly transitions to sqrt(x) for x >= delta
# pybamm.settings.tolerances['j0__c_s'] = 0.001  # default

model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Chen2020")
experiment = pybamm.Experiment(
    [
        "Discharge at 1C for 100 hours or until 2.5 V",
        "Rest for 1 hours",
        # "Charge at 3C until 4.2 V",
        "Hold at 4.2 V until C/1000",
        "Rest for 1 hours",
    ]
    * 100
)

# Add termination events for concentration bounds
for domain in ["negative", "positive"]:
    Domain = domain.capitalize()
    c_s = model.variables[f"{Domain} particle concentration [mol.m-3]"]
    c_s_max = pybamm.Parameter(f"Maximum concentration in {domain} electrode [mol.m-3]")

    # Track c_s_diff = c_max - c_s (for monitoring)
    c_s_diff = c_s_max - c_s
    model.variables[
        f"{Domain} particle concentration difference from maximum [mol.m-3]"
    ] = c_s_diff
    model.variables[
        f"{Domain} particle surface concentration difference from maximum [mol.m-3]"
    ] = pybamm.surf(c_s_diff)

    # Minimum concentration (for monitoring)
    min_value = pybamm.minimum(pybamm.min(c_s_diff), pybamm.min(c_s))
    model.variables[f"{Domain} particle minimum concentration"] = min_value / c_s_max

# Electrolyte concentration events
c_e = model.variables["Electrolyte concentration [mol.m-3]"]
c_e_init = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]")
model.variables["Electrolyte minimum concentration"] = pybamm.min(c_e / c_e_init)

sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=param,
    solver=pybamm.IDAKLUSolver(
        # rtol=1e-6,
        # atol=1e-12,
        options={
            # "print_stats": True,
            "max_num_steps": 5000,
            "t_no_progress": 1.0,
            "num_steps_no_progress": 1000,
            "diagnose_on_failure": True,
        },
        on_failure="ignore",
    ),
)

timer = time.time()
sol = sim.solve(showprogress=True)
failure = sol.termination == "failure"
print()
if failure:
    print("FAILURE!")
else:
    print("SUCCESS!")
print(f"Time taken: {time.time() - timer} seconds")

print()

output_variables = [
    "Voltage [V]",
    "C-rate",
]
min_variables = [
    "Electrolyte minimum concentration",
    "Negative particle minimum concentration",
    "Positive particle minimum concentration",
    "Negative particle concentration",
    "Positive particle concentration",
]
output_variables.extend(min_variables)

for variable in min_variables:
    v = sol[variable].data
    print(variable, v.min(), v.max())

# sol.plot(output_variables=output_variables)
