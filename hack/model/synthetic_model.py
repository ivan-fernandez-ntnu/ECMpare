import pybamm
import pandas as pd
import matplotlib.pyplot as plt

model_type = "MSMR"
step_time = 0.01
solver_method = "IDAKLUSolver"

if model_type == "MP-DFN":
    options = {"particle size": "distribution"}
    model = pybamm.lithium_ion.DFN(options=options, name="MP-DFN (Chen2020 base)")

    # Base parameters
    params = pybamm.ParameterValues("Chen2020")

    # Add PSD parameters (default lognormal assumption)
    params = pybamm.get_size_distribution_parameters(params)

    # --- (Optional but recommended) sanity check ---
    required = [
        "Negative area-weighted particle-size distribution [m-1]",
        "Positive area-weighted particle-size distribution [m-1]",
        "Negative minimum particle radius [m]",
        "Negative maximum particle radius [m]",
        "Positive minimum particle radius [m]",
        "Positive maximum particle radius [m]",
    ]
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"PSD parameters not found after augmentation: {missing}")

elif model_type == "MSMR":
    model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
    parameter_values = model.default_parameter_values



# (Optional) Discretization controls (trade accuracy vs speed)
# Start coarse, then refine once everything runs.
# var_pts = {
#     "x_n": 30, "x_s": 15, "x_p": 30,   # through-cell
#     "r_n": 20, "r_p": 20,              # particle radial diffusion
#     "R_n": 10, "R_p": 10,              # particle-size distribution bins
# }

# 4) Define an experiment to "exercise" the model
# Use C-rates so the current scales with the parameter set capacity.
short_test = True
if short_test:
    experiment = pybamm.Experiment(
            [
                pybamm.step.string("Rest for 1 minutes", period="10 seconds"),
                pybamm.step.string("Discharge at 1C for 10 seconds", period="0.01 seconds"),
                pybamm.step.string("Rest for 1 minutes", period="1 seconds"),
                pybamm.step.string("Charge at 1C until 4.2 V", period="0.01 seconds"),
                pybamm.step.string("Hold at 4.2 V until C/50", period="0.01 seconds"),
                pybamm.step.string("Rest for 1 minutes", period="1 seconds"),
            ]
        )
else:
    experiment = pybamm.Experiment(
        [
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
            pybamm.step.string("Discharge at 1C until 2.5 V", period="1 seconds"),
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C until 4.2 V", period="1 seconds"),
            pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),
            pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 100% SOC
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),   
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 90% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 80% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 70% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 60% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 50% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 40% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 30% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 20% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 10% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Discharge at 0.5C for 0.2 hours", period="0.5 seconds"),      # Discharge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 0% SOC   
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 10% SOC 
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 20% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 30% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 40% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 50% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 60% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 70% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 80% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C for 0.2 hours", period="0.5 seconds"),         # Charge until 10% DOD
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 90% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
            pybamm.step.string("Charge at 1C for 15 seconds", period="0.001 seconds"),      # Charge pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),        
            pybamm.step.string("Charge at 0.5C until 4.2 V", period="1 seconds"),
            pybamm.step.string("Hold at 4.2 V until C/50", period="1 seconds"),
            pybamm.step.string("Rest for 30 minutes", period="10 seconds"),                 # Resting at 100% SOC   
            pybamm.step.string("Discharge at 1C for 15 seconds", period="0.001 seconds"),   # Dicharge Pulse
            pybamm.step.string("Rest for 15 seconds", period="0.001 seconds"),
            pybamm.step.string("Rest for 10 minutes", period="10 seconds"),  
        ],
        # period=f"{step_time} seconds",   # output sampling
    )

# 5) Solve
# IDAKLU is usually fastest for DFN-class models if installed; otherwise Casadi is fine.
# try:
#     solver = pybamm.IDAKLUSolver()
# except Exception:
#     solver = pybamm.CasadiSolver()

if solver_method == "CasadiSolver":
    if model_type == "MP-DFN":
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            experiment=experiment,
            # var_pts=var_pts,
            solver=pybamm.CasadiSolver(
                mode="safe",
                dt_max=1.0,                      # start smaller than the default 600 s 
                max_step_decrease_count=20,      # allow more halvings before error 
                return_solution_if_failed_early=True, 
                ),
        )

    elif model_type == "MSMR":
        sim = pybamm.Simulation(
            model,
            # parameter_values=params,
            experiment=experiment,
            # var_pts=var_pts,
            solver=pybamm.CasadiSolver(
                mode="safe",
                dt_max=step_time,                      # start smaller than the default 600 s 
                max_step_decrease_count=20,      # allow more halvings before error 
                return_solution_if_failed_early=True, 
                ),
        )
elif solver_method == "IDAKLUSolver":
        output_vars = [
            "Voltage [V]",
            "Current [A]",
            # add any other variables you actually need in the CSV
        ]
        sim = pybamm.Simulation(
            model,
            # parameter_values=params,
            experiment=experiment,
            # var_pts=var_pts,
            solver = pybamm.IDAKLUSolver(
                output_variables = output_vars,
                rtol=1e-6,
                atol=1e-6,
                on_failure="warn",   # "warn", "error", or "ignore" (default is raise)
                options={
                    "dt_max": step_time,        # cap internal step to 1 s (try 1.0, 0.5, 0.1 ...)
                    "dt_init": 0.0,       # 0.0 = let solver choose
                    "dt_min": 0.0,        # 0.0 = let solver choose
                    "max_num_steps": 200000,  # bump if you see "max steps" type failures
                },
            )
        )


solution = sim.solve(showprogress=True, initial_soc=0.5)


# 6) Plot a few key outputs (including PSD-resolved variables)

output_variables=[
    "Voltage [V]",
    "Current [A]",
    # "Cell temperature [K]"
    # "X-averaged negative particle surface concentration distribution [mol.m-3]",
    # "X-averaged positive particle surface concentration distribution [mol.m-3]",
    # "Negative area-weighted particle-size distribution [m-1]",
    # "Positive area-weighted particle-size distribution [m-1]",
]

sim.plot(
    output_variables = output_variables,
    # output_variables=[
    #     "Voltage [V]",
    #     "Current [A]",
    #     # "X-averaged negative particle surface concentration distribution [mol.m-3]",
    #     # "X-averaged positive particle surface concentration distribution [mol.m-3]",
    #     # "Negative area-weighted particle-size distribution [m-1]",
    #     # "Positive area-weighted particle-size distribution [m-1]",
    # ]
    show_plot = False
)


step_time_text = str(step_time).replace(".","p")

plt.savefig(fname = f"hack\\model\\output\\figures\\simulation_{model_type}_{solver_method}_HPPC_{step_time_text}.svg")

data_output = pd.DataFrame(solution.get_data_dict())
data_output["time [s]"] = solution.t
data_output.to_csv(f"hack\\model\\output\\csv\\simulation_{model_type}_{solver_method}_HPPC_{step_time_text}.csv")