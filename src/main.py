# imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from tqdm import tqdm

from src.adapter_functions.data_to_method_functions import create_study_cases,data_to_method
from src.adapter_functions.method_to_model_functions import method_to_model
from src.method_functions.method_functions import estimate_eecm_parameters
from src.model_functions.model_generator_functions import model_generator
from src.result_functions.result_functions import append_execution_results, calculate_reference_vs_solution_errors, plot_parameter_surfaces_vs_c_rate_soc_temperature
from hack.model.functions.protocol_functions import generate_experiment
from hack.model.functions.solver_functions import RaiseOnExperimentFailure, generate_solver


# This is the pseudocode of the paper

#* ============ CONFIGURATION OF THE EXECUTION ============

# Data Selection
data_model_type = "MP-DFN"
param_set = "Chen2020" # Chen2020 - "ORegan2022" - "MSMR_Example" - 
test_protocol = "HPPC_short_rest_100soc"  # "HPPC" - "short" - "GITT" - "ICI" - "ICA" - "P-OCV" - "APP"
solver_name = "IDAKLUSolver"  # "CasadiSolver" - "solver"
max_time_step = 0.01
initial_cell_temp_k = 298.15 + 0
output_vars = [
            "Cycle",
            "Step",
            "time [s]",
            "Voltage [V]",
            "Current [A]",
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
            "Discharge capacity [A.h]",
            ]

# Study case selection
study_case_list = ["sc-4"] # ["sc-0", "sc-1", "sc-2", "sc-3", "sc-4"] # "sc-0": original, "sc-1": measurement error, "sc-2": soc error, "sc-3": ocv error, "sc-4": combined error

#* ========== ESTIMATION PROCESS CONFIGURATION ==========
# EECM configuration
model_type = "thevenin_2rc"     # "thevenin_0rc" - "thevenin_1rc" - "thevenin_2rc" ¦¦ Future: "thevenin_3rc" - "thevenin_1rc_hyst" - "thevenin_2rc_hyst" - "fractional" 
cell_capacity_nom = 5.0             # Nominal Capacity for LG M50 21700

# Method configuration
method_group_list = ["LSB"] # "LSB", "LSB", "LSB", "LSB", "LSB", 
method_type_list = ["MWLS"] # "LS", "MWLS", "NLLS", "MWNLLS", "SLSQP", 

# Recursive methods
recursive_execution = False
recursive_method_list = ["basic"]
recursive_param_list = [1]
resursive_freq_second_list = [200]

#* ========== CORE OF THE EXECUTION ==========
for method_group, method_type in tqdm(zip(method_group_list, method_type_list), desc=f"Methods", position=0, leave=True, dynamic_ncols=True):
    print(f"\nmethod={method_type}\n")
    if method_type in ["NLLS", "MWNLLS"]:
        method_solver = "sp.optimize.curve_fit.lm"   # "sp.optimize.curve_fit.lm" 
    elif method_type in ["TROLS", "MWTROLS"]:
        method_solver = "sp.optimize.curve_fit.trf"   # "sp.optimize.curve_fit.trf" - "sp.optimize.curve_fit.dogbox"
    else:
        method_solver = "cholesky"   # "cholesky" - "pinv" - "solve" - "lstsq" 

    initial_guess = [0.01,   0.001,  5,      0.01,   5      ] # [R0, R1, TAU_1, R2, TAU_2]
    lower_bound =   [0.0,    0.0,    0.0,    0.0,    0.0    ] # [R0, R1, TAU_1, R2, TAU_2]
    upper_bound =   [1,      1,      1e4,    1,      1e4    ] # [R0, R1, TAU_1, R2, TAU_2]

    # Test protocol configuration
    fixed_period_protocol = True
    period_value_protocol = 0.01
    # cell_soc_initial = 0.7              # Based on the experiment design 
    # test_protocol_eval = "FUDS"  # "HPPC" - "short" - "GITT" - "ICI" - "ICA" - "P-OCV" - "APP" - "FUDS" - "DST"
    cell_soc_initial_list = [0.5] # [0.3, 0.5, 0.7, 0.3, 0.5, 0.7, 0.3, 0.5, 0.7, 0.999]              # Based on the experiment design  0.9999, 
    test_protocol_eval_list = ["APP_short_rest_050soc_eval"] # ["FUDS_short_rest_030soc_eval","FUDS_short_rest_050soc_eval","FUDS_short_rest_070soc_eval","APP_short_rest_030soc_eval","APP_short_rest_050soc_eval","APP_short_rest_070soc_eval","DST_short_rest_030soc_eval","DST_short_rest_050soc_eval","DST_short_rest_070soc_eval"] #"DST_short_rest_100soc", "FUDS_short_rest_100soc", "APP_short_rest_100soc""ICA_short_rest_100soc", "ICA_short_rest_100soc"
    solver_name_eval = "IDAKLUSolver"  # "CasadiSolver" - "IDAKLUSolver"
    max_time_step_eval = period_value_protocol

    if model_type == "thevenin_0rc" or model_type == "rint":
        output_vars_eval = [
            # "Cycle",
            "Time [s]",
            # "Time [min]",
            # "Time [h]",
            "Current variable [A]",
            # "Total current density [A.m-2]",
            "Current [A]",
            "C-rate",
            "SoC",
            "Open-circuit voltage [V]",
            # "Element-1 overpotential [V]",
            # "Cell temperature [degC]",
            "Cell temperature [K]",
            # "Jig temperature [degC]",
            # "Jig temperature [K]",
            # "Ambient temperature [degC]",
            "Ambient temperature [K]",
            # "Heat transfer from cell to jig [W]",
            # "Heat transfer from jig to ambient [W]",
            # "Entropic change [V/K]",
            # "Reversible heat generation [W]",
            "R0 [Ohm]",
            # "Element-0 overpotential [V]",
            # "Element-0 irreversible heat generation [W]",
            # "R1 [Ohm]",
            # "C1 [F]",
            # "R2 [Ohm]",
            # "C2 [F]",
            # "tau1 [s]",
            # "tau2 [s]",
            # "Element-1 irreversible heat generation [W]",
            # "Distributed SoC",
            # "x ECMD",
            # "Diffusion overpotential [V]",
            # "Surface SoC",
            # "Irreversible heat generation [W]",
            # "Total heat generation [W]",
            "Voltage [V]",
            "Overpotential [V]",
            "Battery voltage [V]",
            "Power [W]",
            # "Resistance [Ohm]",
        ]
    elif model_type == "thevenin_1rc":
        output_vars_eval = [
            # "Cycle",
            "Time [s]",
            # "Time [min]",
            # "Time [h]",
            "Current variable [A]",
            # "Total current density [A.m-2]",
            "Current [A]",
            "C-rate",
            "SoC",
            "Open-circuit voltage [V]",
            "Element-1 overpotential [V]",
            # "Cell temperature [degC]",
            "Cell temperature [K]",
            # "Jig temperature [degC]",
            # "Jig temperature [K]",
            # "Ambient temperature [degC]",
            "Ambient temperature [K]",
            # "Heat transfer from cell to jig [W]",
            # "Heat transfer from jig to ambient [W]",
            # "Entropic change [V/K]",
            # "Reversible heat generation [W]",
            "R0 [Ohm]",
            # "Element-0 overpotential [V]",
            # "Element-0 irreversible heat generation [W]",
            "R1 [Ohm]",
            "C1 [F]",
            # "R2 [Ohm]",
            # "C2 [F]",
            "tau1 [s]",
            # "tau2 [s]",
            # "Element-1 irreversible heat generation [W]",
            # "Distributed SoC",
            # "x ECMD",
            # "Diffusion overpotential [V]",
            # "Surface SoC",
            # "Irreversible heat generation [W]",
            # "Total heat generation [W]",
            "Voltage [V]",
            "Overpotential [V]",
            "Battery voltage [V]",
            "Power [W]",
            # "Resistance [Ohm]",
        ]
    elif model_type == "thevenin_2rc":
        output_vars_eval = [
            # "Cycle",
            "Time [s]",
            # "Time [min]",
            # "Time [h]",
            "Current variable [A]",
            # "Total current density [A.m-2]",
            "Current [A]",
            "C-rate",
            "SoC",
            "Open-circuit voltage [V]",
            "Element-1 overpotential [V]",
            "Element-2 overpotential [V]",
            # "Cell temperature [degC]",
            "Cell temperature [K]",
            # "Jig temperature [degC]",
            # "Jig temperature [K]",
            # "Ambient temperature [degC]",
            "Ambient temperature [K]",
            # "Heat transfer from cell to jig [W]",
            # "Heat transfer from jig to ambient [W]",
            # "Entropic change [V/K]",
            # "Reversible heat generation [W]",
            "R0 [Ohm]",
            # "Element-0 overpotential [V]",
            # "Element-0 irreversible heat generation [W]",
            "R1 [Ohm]",
            "C1 [F]",
            "R2 [Ohm]",
            "C2 [F]",
            "tau1 [s]",
            "tau2 [s]",
            # "Element-1 irreversible heat generation [W]",
            # "Distributed SoC",
            # "x ECMD",
            # "Diffusion overpotential [V]",
            # "Surface SoC",
            # "Irreversible heat generation [W]",
            # "Total heat generation [W]",
            "Voltage [V]",
            "Overpotential [V]",
            "Battery voltage [V]",
            "Power [W]",
            # "Resistance [Ohm]",
        ]

    show_simulation_plot = False 
    reuse_existing_study_case_file = False

    # * ==========================================================
    # region - Load the ocv-curve from the param_set:

    selected_ocv_curve_file = f"hack\\model\\output\\parameters\\ocv_soc_{param_set}.csv"
    u_ocv_table = pd.read_csv(selected_ocv_curve_file, index_col="Unnamed: 0")

    # endregion

    #* This part is to avoid spending time on something that has been already done before and requires lot of time.
    initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")
    max_time_step_text = str(max_time_step).replace(".","p")    

    for study_case in tqdm(study_case_list, desc=f"Study case", position=1, leave=False, dynamic_ncols=True):
        print(f"\nsc={study_case}\n")
        try:    
            if not Path(f"src\\data\\study_case_files\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_{study_case}.parquet").exists() or not reuse_existing_study_case_file:
            # region - Load the selected data:
                
                # selected_data_file = f"hack\\model\\output\\csv\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv"
                selected_data_file = f"hack\\model\\output\\parquet\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.parquet"
                selected_data_path= os.path.join(os.getcwd(),selected_data_file)
                # data_t = pd.read_csv(selected_data_path)
                raw_data_t = pd.read_parquet(selected_data_path)

                data_t = create_study_cases(raw_data_t, study_case)
                data_t.to_parquet(f"src\\data\\study_case_files\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_{study_case}.parquet", compression="gzip")

            else:

                # selected_data_file = f"hack\\model\\output\\csv\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv"
                selected_data_file = f"src\\data\\study_case_files\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_{study_case}.parquet"
                selected_data_path= os.path.join(os.getcwd(),selected_data_file)
                # data_t = pd.read_csv(selected_data_path)
                data_t = pd.read_parquet(selected_data_path)

            # endregion

            # region - Generate data_to_method_adapter()
            """
            The idea of this section is to adapt the data from the files to the format that is required by the method function selected in this particular case.
            """
            method_input = data_to_method(data_t.copy(),output_vars, max_time_step, method_group, method_type,model_type, initial_guess, lower_bound, upper_bound, study_case)

            # endregion

            # region - Generate estimation method
            """
            This section calls the method function and feed it with the adapted data.
            """
            estimation_result = estimate_eecm_parameters(method_input, method_group, method_type, method_solver)
            # print(estimation_result)
            # endregion

            # region - Generate method_to_model_adapter()
            """
            The idea of this section is to adapt the data from the files to the format that is required by the method function selected in this particular case.
            """
            model_parameters = method_to_model(estimation_result, method_input, method_group, method_type, model_type, u_ocv_table)

            # endregion

            # region - Generate model
            """
            The idea of this section is to receive model_parameters and generate a model object. This model_generator should be able to receive parameters as values, functions and LUT.
            """
            model_object, params_object = model_generator(model_type, model_parameters, param_set)
            # endregion
        except Exception as err:
            print(f"Error during estimation process with method {method_type}, study case {study_case} and protocol {test_protocol}: {err}")
            continue
        
        # region - Simulate test protocols
        """
        The idea of this section is to generate the simulation object that contains the results.
        """

        #* Define the protocol
        max_time_step_eval_text = str(max_time_step_eval).replace(".","p")
        initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")

        for test_protocol_eval, cell_soc_initial in tqdm(zip(test_protocol_eval_list, cell_soc_initial_list), desc=f"Eval protocol", position=2, leave=False, dynamic_ncols=True):
            print(f"\neval_protocol={test_protocol_eval}\n")
            try:

                if recursive_execution:

                    experiment = generate_experiment(protocol_name=test_protocol_eval, fixed_period=fixed_period_protocol, period_value = period_value_protocol)

                    params_object.update({
                        "Ambient temperature [K]": initial_cell_temp_k,
                        "Initial temperature [K]": initial_cell_temp_k,
                        "Initial SoC": pybamm.Scalar(cell_soc_initial),
                        "Lower voltage cut-off [V]": 0, 
                        "Upper voltage cut-off [V]": 10,
                    }, check_already_exists=False)

                    solver = generate_solver(solver_name=solver_name_eval, output_vars=output_vars_eval, max_time_step=max_time_step_eval)

                    # Until this point everthing stay the same

                    # The model object should be updated probably and this restrict the methods to those ones with parameters and not functions.

                    sim = pybamm.Simulation(
                        model_object,
                        parameter_values=params_object,
                        experiment=experiment,
                        solver=solver,
                    )

                    # Calculate G[k] and P[k] every t seconds


                    True
                else:                    

                    experiment = generate_experiment(protocol_name=test_protocol_eval, fixed_period=fixed_period_protocol, period_value = period_value_protocol)

                    params_object.update({
                        "Ambient temperature [K]": initial_cell_temp_k,
                        "Initial temperature [K]": initial_cell_temp_k,
                        "Initial SoC": pybamm.Scalar(cell_soc_initial),
                        "Lower voltage cut-off [V]": 0, 
                        "Upper voltage cut-off [V]": 10,
                    }, check_already_exists=False)

                    solver = generate_solver(solver_name=solver_name_eval, output_vars=output_vars_eval, max_time_step=max_time_step_eval)

                    sim = pybamm.Simulation(
                        model_object,
                        parameter_values=params_object,
                        experiment=experiment,
                        solver=solver,
                    )

                    # Run the simulation
                    solution = sim.solve(
                        showprogress=True,
                        callbacks=[RaiseOnExperimentFailure()],
                        )

                    sim.plot(
                        output_variables = output_vars_eval,
                        show_plot = show_simulation_plot
                    )

                    # Export the data
                    plt.savefig(fname = f"src\\output\\figures\\simulation\\simulation_{model_type}_{param_set}_{method_type}_{test_protocol}__{max_time_step_text}_{test_protocol_eval}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.svg")
                    plt.close()

                    data_output = pd.DataFrame(solution.get_data_dict())
                    data_output["time [s]"] = solution.t
                    data_output.to_parquet(f"src\\output\\parquet\\simulation\\simulation_{model_type}_{param_set}_{method_type}_{test_protocol}__{max_time_step_text}_{test_protocol_eval}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.parquet", compression="gzip")


                    #* Calculate error
                    # Load reference profile
                    selected_data_file = f"src\\data\\study_case_files\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_sc-0.parquet"
                    selected_data_path= os.path.join(os.getcwd(),selected_data_file)
                    data_reference_t = pd.read_parquet(selected_data_path)

                    # Compare values from data_reference_t and solution
                    soc_bin_width = 0.01

                    results = calculate_reference_vs_solution_errors(
                        data_reference_t=data_reference_t,
                        solution=solution,
                        soc_bin_width=soc_bin_width,   # 1% SoC bins
                        ohmic_sign=-1.0,
                        plot=True,
                        save_plots=True,
                        save_dir=f"src\\output\\figures\\errors\\error_{model_type}_{param_set}_{method_type}_{test_protocol}__{max_time_step_text}_{test_protocol_eval}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}",
                    )

                    exported_paths = append_execution_results(
                        results=results,
                        export_dir=f"src\\output\\figures\\errors",
                        model_type=model_type,
                        param_set=param_set,
                        method_type=method_type,
                        solver_name_eval=solver_name_eval,
                        test_protocol=test_protocol,
                        test_protocol_eval=test_protocol_eval,
                        initial_cell_temp_k_text=initial_cell_temp_k_text,
                        max_time_step_eval_text=max_time_step_eval_text,
                        study_case = study_case,
                        extra_metadata={
                            "soc_bin_width": soc_bin_width,
                        },
                        export_soc_summary=True,
                        export_residuals_per_run=False,
                    )

                    print(exported_paths)
            
            except pybamm.SolverError as err:
                print(f"\nError during execution with study-case {study_case} and test protocol {test_protocol_eval}: {err}\n")


        # endregion

        # region - Results
        """
        The idea of this region is to generate the results based on the comparison of the EECM and the data source used.
        """

        # Example callable
        if method_type in ["MWLS", "LS", "NLLS", "SLSQP", "TROLS", "MWNLLS", "MWTROLS", "MWSLSQP"]:

            temperatures = [25.0]
            c_rate_values = np.linspace(-2.0, 2.0, 100)
            c_rate_values = c_rate_values[np.abs(c_rate_values) > 0.05]
            soc_values = np.linspace(0.0, 1.0, 100)

            if model_type in ["rint", "thevenin_0rc", "thevenin_1rc", "thevenin_2rc"]:

                r0_fun = model_parameters[model_type]["r_0"]
                _,_,_ = plot_parameter_surfaces_vs_c_rate_soc_temperature(
                    param_fun=r0_fun,
                    temperatures=temperatures,
                    c_rate_values=c_rate_values,
                    soc_values=soc_values,
                    capacity_ah=cell_capacity_nom,
                    parameter_name="R0 [Ohm]",
                    temperature_unit="°C",
                    save_path=f"src\\output\\figures\\parameters\\r_0_{model_type}_{param_set}_{method_type}_{solver_name_eval}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.png",
                    show=False,
                )

            if model_type in ["thevenin_1rc", "thevenin_2rc"]:

                r1_fun = model_parameters[model_type]["r_1"]
                _,_,_ = plot_parameter_surfaces_vs_c_rate_soc_temperature(
                    param_fun=r1_fun,
                    temperatures=temperatures,
                    c_rate_values=c_rate_values,
                    soc_values=soc_values,
                    capacity_ah=cell_capacity_nom,
                    parameter_name="R1 [Ohm]",
                    temperature_unit="°C",
                    save_path=f"src\\output\\figures\\parameters\\r_1_{model_type}_{param_set}_{method_type}_{solver_name_eval}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.png",
                    show=False,
                )
                c1_fun = model_parameters[model_type]["c_1"]
                _,_,_ = plot_parameter_surfaces_vs_c_rate_soc_temperature(
                    param_fun=c1_fun,
                    temperatures=temperatures,
                    c_rate_values=c_rate_values,
                    soc_values=soc_values,
                    capacity_ah=cell_capacity_nom,
                    parameter_name="C1 [F]",
                    temperature_unit="°C",
                    save_path=f"src\\output\\figures\\parameters\\c_1_{model_type}_{param_set}_{method_type}_{solver_name_eval}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.png",
                    show=False,
                )

            if model_type in ["thevenin_2rc"]:

                r2_fun = model_parameters[model_type]["r_2"]
                _,_,_ = plot_parameter_surfaces_vs_c_rate_soc_temperature(
                    param_fun=r2_fun,
                    temperatures=temperatures,
                    c_rate_values=c_rate_values,
                    soc_values=soc_values,
                    capacity_ah=cell_capacity_nom,
                    parameter_name="R2 [Ohm]",
                    temperature_unit="°C",
                    save_path=f"src\\output\\figures\\parameters\\r_2_{model_type}_{param_set}_{method_type}_{solver_name_eval}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.png",
                    show=False,
                )
                c2_fun = model_parameters[model_type]["c_2"]
                _,_,_ = plot_parameter_surfaces_vs_c_rate_soc_temperature(
                    param_fun=c2_fun,
                    temperatures=temperatures,
                    c_rate_values=c_rate_values,
                    soc_values=soc_values,
                    capacity_ah=cell_capacity_nom,
                    parameter_name="C2 [F]",
                    temperature_unit="°C",
                    save_path=f"src\\output\\figures\\parameters\\c_2_{model_type}_{param_set}_{method_type}_{solver_name_eval}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_eval_text}_{study_case}.png",
                    show=False,
                )


        # endregion

