#* Import the libraries
import numpy as np
import pandas as pd
import os
import pybamm
import matplotlib.pyplot as plt

from pathlib import Path

from hack.model.functions.model_functions import generate_battery_model
from hack.model.functions.protocol_functions import generate_experiment, simulate_long_experiment
from hack.model.functions.solver_functions import generate_solver

from tqdm import tqdm

# ====================================================================
#* Execution configuration options
model_type = "MP-DFN"
param_set = "Chen2020" # Chen2020 - "ORegan2022" - "MSMR_Example" - "OKane2022"
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
    "DFN": {    # This simple DFN is meant to be used when degradation data is generated
        # Thermal configuration
        "thermal": "lumped",
        "surface temperature": "ambient",

        # Current-collector configuration
        "current collector": "uniform",
        "dimensionality": 0,

        # SEI growth
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",

        # Lithium plating
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",

        # Mechanical degradation
        # Negative: swelling and cracking
        # Positive: swelling without cracking
        "particle mechanics": (
            "swelling and cracking",
            "swelling only",
        ),

        # SEI growth over newly generated crack surfaces
        "SEI on cracks": "true",

        # Loss of active material
        "loss of active material": "stress-driven",

        # Accumulated energy variables
        "calculate discharge energy": "true",
    }
}

# test_protocol_list =            ["DST_short_rest_010soc_eval", "DST_short_rest_020soc_eval", "DST_short_rest_030soc_eval", "DST_short_rest_040soc_eval", "DST_short_rest_050soc_eval", "DST_short_rest_060soc_eval", "DST_short_rest_070soc_eval", "DST_short_rest_080soc_eval", "DST_short_rest_090soc_eval", "DST_short_rest_100soc_eval", "DST_short_rest_000soc_eval"]
# period_value_protocol_list =    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# intial_soc_pu_list =            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
# initial_cell_temp_k_list =      [298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15, 298.15]
test_protocol_list =            ["HPPC_short_rest_100soc_long_experiment"]
period_value_protocol_list =    [0.001]
intial_soc_pu_list =            [1.0]
initial_cell_temp_k_list =      [298.15+20]

fixed_period_protocol = True


for test_protocol, period_value_protocol, intial_soc_pu, initial_cell_temp_k in tqdm(zip(test_protocol_list, period_value_protocol_list, intial_soc_pu_list, initial_cell_temp_k_list), total=len(test_protocol_list)):
    try:
        if "long_experiment" in test_protocol:
            long_experiment = True
        else:
            long_experiment = False

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


        # Define the model
        model, params = generate_battery_model(model_type=model_type, param_set=param_set, options=model_options)

        # Define the initial conditions

        params.update({
            "Ambient temperature [K]": initial_room_temp_k,
            "Initial temperature [K]": initial_cell_temp_k,
        }, check_already_exists=False)

        # Define the solver

        solver = generate_solver(solver_name=solver_name, output_vars=output_vars, max_time_step=max_time_step)


        if long_experiment:
            
            solution, sim = simulate_long_experiment(intial_soc_pu, test_protocol, model, params, solver, model_type, param_set, solver_name, max_time_step, initial_cell_temp_k, output_vars, plot_interm = False)

        else:

            #* Define the protocol
            experiment = generate_experiment(protocol_name=test_protocol, fixed_period=fixed_period_protocol, period_value = period_value_protocol)
            
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

            elif param_set == "OKane2022":
                var_pts = { 
                    "x_n": 5,
                    "x_s": 5,
                    "x_p": 5,
                    "r_n": 30,
                    "r_p": 30,
                    }
                
                simulation = pybamm.Simulation(
                    model=model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                    var_pts=var_pts,
                )

                solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)

                sim.plot(
                    output_variables = output_vars,
                    show_plot = False
                )

            else:
                sim = pybamm.Simulation(
                    model,
                    parameter_values=params,
                    experiment=experiment,
                    solver=solver,
                )

            solution = sim.solve(showprogress=True, initial_soc=intial_soc_pu)

            sim.plot(
                output_variables = output_vars,
                show_plot = False
            )


        # Export the data

        if not long_experiment:

            max_time_step_text = str(max_time_step).replace(".","p")
            initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")
            #TODO Check this step because when long experiment the plot is already done.
            plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.png")
            data_output = pd.DataFrame(solution.get_data_dict())
            data_output["time [s]"] = solution.t
            # data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv")
            data_output.to_parquet(f"hack\\model\\output\\parquet\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.parquet", compression="gzip")
            data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv")


    except Exception as e:
        print(e)
        continue
