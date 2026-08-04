import pandas as pd
import numpy as np
import pybamm

import re

def model_generator(model_type, model_parameters, param_set):
    
    # "rint" - "thevenin_1rc" - "thevenin_2rc" - "thevenin_3rc" - "thevenin_1rc_hyst" - "thevenin_2rc_hyst" - "fractional" 
    if model_type == "rint" or model_type == "thevenin_0rc":
        n_rc = 0
        hyst_label = None
        model_options = {
            "number of rc elements": n_rc,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }
        model_object = pybamm.equivalent_circuit.Thevenin(name=f'Thevenin Equivalent Circuit Model ({n_rc}RC)', options=model_options, build=True)
        params_refer = pybamm.ParameterValues(param_set)
        params = pybamm.ParameterValues("ECM_Example")

        # Construye OCV(SOC) desde Chen2020

        params.update({
            "R0 [Ohm]": model_parameters[model_type]["r_0"],
            # "R1 [Ohm]": model_parameters[model_type]["r_1"],
            # "C1 [F]": model_parameters[model_type]["c_1"],
            "Cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            # "Cell thermal mass [J/K]": 1000, 
            # "Cell-jig heat transfer coefficient [W/K]": 10, 
            # "Current function [A]": 100, 
            # "Element-1 initial overpotential [V]": 0, 
            # "Entropic change [V/K]": <function dUdT at 0x0000020B96F1E0C0>, 
            # "Initial SoC": 0.5, 
            # "Initial temperature [K]": 298.15, 
            # "Jig thermal mass [J/K]": 500, 
            # "Jig-air heat transfer coefficient [W/K]": 10, 
            "Lower voltage cut-off [V]": params_refer["Lower voltage cut-off [V]"], 
            "Upper voltage cut-off [V]": params_refer["Upper voltage cut-off [V]"],
            "Nominal cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            "Open-circuit voltage [V]": model_parameters[model_type]["u_ocv"], 
            # "Initial SoC": pybamm.Scalar(cell_soc_initial), 
            # "Open-circuit voltage [V]": True #TODO Check if this parameter make sense. I guess so because it comes from the Chan2020 parameter set
        })

    elif model_type == "thevenin_1rc":
        n_rc = 1
        hyst_label = None
        model_options = {
            "number of rc elements": n_rc,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }
        model_object = pybamm.equivalent_circuit.Thevenin(name=f'Thevenin Equivalent Circuit Model ({n_rc}RC)', options=model_options, build=True)
        params_refer = pybamm.ParameterValues(param_set)
        params = pybamm.ParameterValues("ECM_Example")

        # Construye OCV(SOC) desde Chen2020

        params.update({
            "R0 [Ohm]": model_parameters[model_type]["r_0"],
            "R1 [Ohm]": model_parameters[model_type]["r_1"],
            "C1 [F]": model_parameters[model_type]["c_1"],
            "Cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            # "Cell thermal mass [J/K]": 1000, 
            # "Cell-jig heat transfer coefficient [W/K]": 10, 
            # "Current function [A]": 100, 
            # "Element-1 initial overpotential [V]": 0, 
            # "Entropic change [V/K]": <function dUdT at 0x0000020B96F1E0C0>, 
            # "Initial SoC": 0.5, 
            # "Initial temperature [K]": 298.15, 
            # "Jig thermal mass [J/K]": 500, 
            # "Jig-air heat transfer coefficient [W/K]": 10, 
            "Lower voltage cut-off [V]": params_refer["Lower voltage cut-off [V]"], 
            "Upper voltage cut-off [V]": params_refer["Upper voltage cut-off [V]"],
            "Nominal cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            "Open-circuit voltage [V]": model_parameters[model_type]["u_ocv"], 
            # "Initial SoC": pybamm.Scalar(cell_soc_initial), 
            # "Open-circuit voltage [V]": True #TODO Check if this parameter make sense. I guess so because it comes from the Chan2020 parameter set
        })

    elif model_type == "thevenin_2rc":
        n_rc = 2
        hyst_label = None
        model_options = {
            "number of rc elements": n_rc,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }
        model_object = pybamm.equivalent_circuit.Thevenin(name=f'Thevenin Equivalent Circuit Model ({n_rc}RC)', options=model_options, build=True)
        params_refer = pybamm.ParameterValues(param_set)
        params = pybamm.ParameterValues("ECM_Example")

        # Construye OCV(SOC) desde Chen2020

        params.update({
            "R0 [Ohm]": model_parameters[model_type]["r_0"],
            "R1 [Ohm]": model_parameters[model_type]["r_1"],
            "C1 [F]": model_parameters[model_type]["c_1"],
            "R2 [Ohm]": model_parameters[model_type]["r_2"],
            "C2 [F]": model_parameters[model_type]["c_2"],
            "Cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            # "Cell thermal mass [J/K]": 1000, 
            # "Cell-jig heat transfer coefficient [W/K]": 10, 
            # "Current function [A]": 100, 
            "Element-1 initial overpotential [V]": 0, 
            "Element-2 initial overpotential [V]": 0, 
            # "Entropic change [V/K]": <function dUdT at 0x0000020B96F1E0C0>, 
            # "Initial SoC": 0.5, 
            # "Initial temperature [K]": 298.15, 
            # "Jig thermal mass [J/K]": 500, 
            # "Jig-air heat transfer coefficient [W/K]": 10, 
            "Lower voltage cut-off [V]": params_refer["Lower voltage cut-off [V]"], 
            "Upper voltage cut-off [V]": params_refer["Upper voltage cut-off [V]"],
            "Nominal cell capacity [A.h]": params_refer["Nominal cell capacity [A.h]"], 
            "Open-circuit voltage [V]": model_parameters[model_type]["u_ocv"], 
            # "Initial SoC": pybamm.Scalar(cell_soc_initial), 
            # "Open-circuit voltage [V]": True #TODO Check if this parameter make sense. I guess so because it comes from the Chan2020 parameter set
        })

    return model_object, params


def build_ocv_interpolant_from_chen2020(
    thevenin_model: pybamm.BaseModel,
    n_points: int = 501,
    chen_param_set: str = "Chen2020",
    # Límites de estequiometría (Chen et al. 2020)
    x_n_0: float = 0.0279,   # neg @ 0% SOC
    x_n_100: float = 0.9014, # neg @ 100% SOC
    x_p_0: float = 0.9084,   # pos @ 0% SOC
    x_p_100: float = 0.2661, # pos @ 100% SOC
    ):
    """
    Construye OCV(SOC) = Up(sto_p(SOC)) - Un(sto_n(SOC)) a partir de las OCPs de Chen2020
    y devuelve:
      - ocv_interpolant: pybamm.Interpolant(soc_grid, ocv_grid, SoC_symbol)
      - df_curve: DataFrame con 'SOC [pu]' y 'OCV [V]'
    """
    chen = pybamm.ParameterValues(chen_param_set)

    soc_grid = np.linspace(0.0, 1.0, n_points)

    # SOC -> sto (lineal) usando límites reportados
    sto_n = x_n_0 + soc_grid * (x_n_100 - x_n_0)      # aumenta con SOC
    sto_p = x_p_0 + soc_grid * (x_p_100 - x_p_0)      # disminuye con SOC

    # OCPs (funciones en Chen2020) evaluadas en 'sto'
    sto_in = pybamm.InputParameter("sto")
    Un_expr = pybamm.FunctionParameter("Negative electrode OCP [V]", {"sto": sto_in})
    Up_expr = pybamm.FunctionParameter("Positive electrode OCP [V]", {"sto": sto_in})

    # Eval punto a punto (evita líos de tamaños al evaluar arrays)
    Un_vals = np.array([float(chen.evaluate(Un_expr, inputs={"sto": float(s)})) for s in sto_n])
    Up_vals = np.array([float(chen.evaluate(Up_expr, inputs={"sto": float(s)})) for s in sto_p])

    ocv_grid = Up_vals - Un_vals
    ocv_grid[-1] = chen["Upper voltage cut-off [V]"]
    ocv_grid[0] = chen["Lower voltage cut-off [V]"]
    # Interpolant como función del SoC del modelo Thevenin
    soc_symbol = thevenin_model.variables["SoC"]
    ocv_interpolant = pybamm.Interpolant(soc_grid, ocv_grid, soc_symbol)

    df_curve = pd.DataFrame({"SOC [pu]": soc_grid, "OCV [V]": ocv_grid})
    return ocv_interpolant, df_curve
