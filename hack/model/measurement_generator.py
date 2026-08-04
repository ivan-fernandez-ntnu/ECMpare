#* Import the libraries
import numpy as np
import pandas as pd
import os
import pybamm
import matplotlib.pyplot as plt

from pathlib import Path

from hack.model.functions.model_functions import generate_battery_model
from hack.model.functions.protocol_functions import generate_experiment
from hack.model.functions.solver_functions import generate_solver

# ====================================================================
#* Execution configuration options
model_type = "MP-DFN"
param_set = "Chen2020" # Chen2020 - "ORegan2022" - "MSMR_Example" - 
model_options = {
    "MP-DFN": {
        "particle size": "distribution",
        "thermal": "lumped",
        "surface temperature": "ambient",
        "current collector": "uniform",
        "dimensionality": 0,
    },
    "MSMR": {
        "number of MSMR reactions": ("6", "4"),
        "thermal": "lumped",
        # "surface temperature": "lumped",
        # "current collector": "uniform",
        # "dimensionality": 0,
    },
}

test_protocol = "ICA_short_rest_100soc"  # "HPPC" - "short" - "GITT" - "GITT_short_rest" - "ICI" - "ICA" - "P-OCV" - "APP" - "FUDS" - "DST" - "APP_short_rest_100soc" - "STEP_short_rest_050soc"
fixed_period_protocol = True
period_value_protocol = 0.01

intial_soc_pu = 1.00    #! Atencion
initial_cell_temp_k = 298.15 + 0
initial_room_temp_k = initial_cell_temp_k

output_vars = [
            "Voltage [V]",
            "Current [A]",
            'Battery voltage [V]',
            # "Cell temperature [K]"
            "X-averaged cell temperature [K]",
            "Negative electrode bulk open-circuit potential [V]",
            "Positive electrode bulk open-circuit potential [V]",
            "Negative particle concentration overpotential [V]",
            "Positive particle concentration overpotential [V]",
            "X-averaged negative electrode reaction overpotential [V]",
            "X-averaged positive electrode reaction overpotential [V]",
            "X-averaged concentration overpotential [V]",
            "X-averaged electrolyte ohmic losses [V]",
            "X-averaged negative electrode ohmic losses [V]",
            "X-averaged positive electrode ohmic losses [V]",
            'X-averaged battery solid phase ohmic losses [V]',
            "Discharge capacity [A.h]",
            "Average negative particle stoichiometry",
            "Average positive particle stoichiometry",
            
        ]

# output_vars = None

solver_name = "IDAKLUSolver"  # "CasadiSolver" - "solver"
max_time_step = period_value_protocol


# ====================================================================

#* Define the protocol
experiment = generate_experiment(protocol_name=test_protocol, fixed_period=fixed_period_protocol, period_value = period_value_protocol)

# Define the model
model, params = generate_battery_model(model_type=model_type, param_set=param_set, options=model_options)

# Define the initial conditions

params.update({
    "Ambient temperature [K]": initial_room_temp_k,
    "Initial temperature [K]": initial_cell_temp_k,
}, check_already_exists=False)

# Define the solver

solver = generate_solver(solver_name=solver_name, output_vars=output_vars, max_time_step=max_time_step)

# Define the simulation
if param_set == "ORegan2022":

    var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 40, "r_p": 40, "R_n": 12, "R_p": 12}

    submesh_types = model.default_submesh_types
    submesh_types["negative particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
    submesh_types["positive particle"] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, submesh_params={"side": "right"})
    
    sim = pybamm.Simulation(
        model,
        parameter_values=params,
        experiment=experiment,
        var_pts=var_pts,
        submesh_types=submesh_types,
        solver=solver,
    )
else:
    sim = pybamm.Simulation(
        model,
        parameter_values=params,
        experiment=experiment,
        solver=solver,
    )


# Run the simulation

solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)

# Make the plot

sim.plot(
    output_variables = output_vars,
    show_plot = False
)


# Export the data


max_time_step_text = str(max_time_step).replace(".","p")
initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")

plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.svg")

data_output = pd.DataFrame(solution.get_data_dict())
data_output["time [s]"] = solution.t
# data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv")
data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.parquet", compression="gzip")
