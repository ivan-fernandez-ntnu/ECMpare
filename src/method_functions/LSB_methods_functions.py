import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
import scipy as sp
import pyslsqp

from src.utils.registry import Registry
from typing import Callable, Dict, Generic, TypeVar, List, Any, Union, Optional, Sequence

from tqdm import tqdm

# region - LS-based methods functions
LSB_method = Registry[Callable[..., Any]]("LSB_method")

@LSB_method.register("LSB_LS_method_backup")
def LSB_LS_method_backup(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """
    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    T_s = method_input["T_s"]
    

    if solve_method == "pinv":
        # Solve θ = (Φ.T · Φ)^(-1) · Φ.T · Y
        theta_array_i = np.linalg.pinv(phi_array.T @ phi_array) @ phi_array.T @ y_array
    
    elif solve_method == "lstsq":
        # Solve min ||Y - Φ · θ||
        theta_array_i, residuals, rank, s = np.linalg.lstsq(phi_array, y_array, rcond = None) #TODO Check this to remember how this np function works

    elif solve_method == "solve":
        # Solve A·θ = b
        A = phi_array.T @ phi_array
        b = phi_array.T @ y_array
        theta_array_i = np.linalg.solve(A, b)   
    
    elif solve_method == "cholesky":
        # Solve A·θ = b with A = L · L.T
        A = phi_array.T @ phi_array
        b = phi_array.T @ y_array
        c, lower = sp.linalg.cho_factor(A)                 # Cholesky
        theta_array_i = sp.linalg.cho_solve((c, lower), b)             # solve Ax=b without the inverse

    #! This values won't affect to the parameter function since the idea is that, for all of them, the value is the same. It is just to keep the structure
    temperature_mean_j = 25
    soc_mean_i = 0.5
    current_mean_i = 1

    key = (
        round(temperature_mean_j, 3),
        round(soc_mean_i, 3),
        round(current_mean_i, 3),
    )
    theta_array = {}
    theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_LS_method")
def LSB_LS_method(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """

    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]

    if model_type == "rint" or model_type == "thevenin_0rc":
        # y_array calc    
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc        
        phi_array = np.array([- i_bulk_k_array]).T      #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT

    elif model_type == "thevenin_1rc":
        # y_array calc  
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc
        phi_array = np.array([u_term_km1_array - u_ocv_km1_array, - i_bulk_k_array, - i_bulk_km1_array]).T  #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT


    elif model_type == "thevenin_2rc":
        # y_array calc      
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc
        phi_array = np.array([u_term_km1_array - u_ocv_km1_array, u_term_km2_array - u_ocv_km2_array, - i_bulk_k_array, - i_bulk_km1_array, - i_bulk_km2_array]).T    #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT
            

    if solve_method == "pinv":
        # Solve θ = (Φ.T · Φ)^(-1) · Φ.T · Y
        theta_array_i = np.linalg.pinv(phi_array.T @ phi_array) @ phi_array.T @ y_array
    
    elif solve_method == "lstsq":
        # Solve min ||Y - Φ · θ||
        theta_array_i, residuals, rank, s = np.linalg.lstsq(phi_array, y_array, rcond = None) #TODO Check this to remember how this np function works

    elif solve_method == "solve":
        # Solve A·θ = b
        A = phi_array.T @ phi_array
        b = phi_array.T @ y_array
        theta_array_i = np.linalg.solve(A, b)   
    
    elif solve_method == "cholesky":
        # Solve A·θ = b with A = L · L.T
        A = phi_array.T @ phi_array
        b = phi_array.T @ y_array
        c, lower = sp.linalg.cho_factor(A)                 # Cholesky
        theta_array_i = sp.linalg.cho_solve((c, lower), b)             # solve Ax=b without the inverse

    #! This values won't affect to the parameter function since the idea is that, for all of them, the value is the same. It is just to keep the structure
    temperature_mean_j = 25
    soc_mean_i = 0.5
    current_mean_i = 1

    key = (
        round(temperature_mean_j, 3),
        round(soc_mean_i, 3),
        round(current_mean_i, 3),
    )
    theta_array = {}
    theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_NLLS_method")
def LSB_NLLS_method(method_input, solve_method):
    """
    Solve the 2RC ECM parameter estimation problem with nonlinear least squares.

    Input expected in method_input:
        - method_input["phi_array"] : np.ndarray with shape (N, 5)
            Columns must be:
            [y[k-1], y[k-2], i[k], i[k-1], i[k-2]]

        - method_input["y_array"] : np.ndarray with shape (N,)
            Target:
            y[k] = U_term[k] - U_oc[k]

        - method_input["T_s"] : float
            Fixed sampling time

    Optional keys in method_input:
        - "initial_guess" : [R0, R1, tau1, R2, tau2]
        - "bounds" : (lower_bounds, upper_bounds)
        - "max_n_samples" : int
        - "loss" : str for scipy.optimize.least_squares
        - "f_scale" : float for scipy.optimize.least_squares

    Parameters returned:
        theta_array = [R0, R1, tau1, R2, tau2]
    """

    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]

    # # Column used to detect changes
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     ref_col = phi_array[:, 1]
    # elif model_type == "thevenin_2rc":
    #     ref_col = phi_array[:, 2]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)
    # # Append new time column
    # phi_array = np.hstack((phi_array, time_col))
    
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     mask = phi_array[:, 1] != 0
    # elif model_type == "thevenin_2rc":
    #     mask = phi_array[:, 2] != 0
    # phi_array = phi_array[mask]
    # y_array = y_array[mask]

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    # u_term_k_array = u_term_k_array[mask]
    # u_term_km1_array = u_term_km1_array[mask]
    # u_term_km2_array = u_term_km2_array[mask]
    # u_ocv_k_array = u_ocv_k_array[mask]
    # u_ocv_km1_array = u_ocv_km1_array[mask]
    # u_ocv_km2_array = u_ocv_km2_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    # i_bulk_km1_array = i_bulk_km1_array[mask]
    # i_bulk_km2_array = i_bulk_km2_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            # "u_term_k_array": u_term_k_array,
            # "u_term_km1_array": u_term_km1_array,
            # "u_term_km2_array": u_term_km2_array,
            # "u_ocv_k_array": u_ocv_k_array,
            # "u_ocv_km1_array": u_ocv_km1_array,
            # "u_ocv_km2_array": u_ocv_km2_array,
            "i_bulk_k_array": i_bulk_k_array,
            # "i_bulk_km1_array": i_bulk_km1_array,
            # "i_bulk_km2_array": i_bulk_km2_array,
        })
    #! This method is not good for long amount of data, let's figure out first how to make the slices of data to local estimation.
    
    theta_array_i = np.empty(0)
    #* Declare model functions
    skip_0rc = True
    
    if skip_0rc:
        if model_type in ["rint", "thevenin_0rc"]:
            # Solve 0RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_0RC, phi_array, y_array, initial_guess[0], lb[0], ub[0], [0.001])
            phi_array.attrs["r_0"] = theta_array_raw[0][0]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_0"]])
        if model_type in ["thevenin_1rc", "thevenin_2rc"]:
            # Solve 1RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_1RC_skip_0rc, phi_array, y_array, initial_guess[:3], lb[:3], ub[:3], [0.001, 0.001, 1])
            phi_array.attrs["r_0"] = theta_array_raw[0][0]
            phi_array.attrs["r_1"] = theta_array_raw[0][1]
            phi_array.attrs["tau_1"] = theta_array_raw[0][2]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_0"], phi_array.attrs["r_1"], phi_array.attrs["tau_1"]])
        if model_type in ["thevenin_2rc"]:
            # Solve 2RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_2RC, phi_array, y_array, initial_guess[3:5], lb[3:5], ub[3:5], [0.001, 1])
            phi_array.attrs["r_2"] = theta_array_raw[0][0]
            phi_array.attrs["tau_2"] = theta_array_raw[0][1]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_2"], phi_array.attrs["tau_2"]])

    else:
        if model_type in ["rint", "thevenin_0rc", "thevenin_1rc", "thevenin_2rc"]:
            # Solve 0RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_0RC, phi_array, y_array, initial_guess[0], lb[0], ub[0], [0.001])
            phi_array.attrs["r_0"] = theta_array_raw[0][0]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_0"]])
        if model_type in ["thevenin_1rc", "thevenin_2rc"]:
            # Solve 1RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_1RC, phi_array, y_array, initial_guess[1:3], lb[1:3], ub[1:3], [0.001, 1])
            phi_array.attrs["r_1"] = theta_array_raw[0][0]
            phi_array.attrs["tau_1"] = theta_array_raw[0][1]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_1"], phi_array.attrs["tau_1"]])
        if model_type in ["thevenin_2rc"]:
            # Solve 2RC
            theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_2RC, phi_array, y_array, initial_guess[3:5], lb[3:5], ub[3:5], [0.001, 1])
            phi_array.attrs["r_2"] = theta_array_raw[0][0]
            phi_array.attrs["tau_2"] = theta_array_raw[0][1]
            theta_array_i = np.append(theta_array_i, [phi_array.attrs["r_2"], phi_array.attrs["tau_2"]])

    #! This values won't affect to the parameter function since the idea is that, for all of them, the value is the same. It is just to keep the structure
    temperature_mean_j = 25
    soc_mean_i = 0.5
    current_mean_i = 1

    key = (
        round(temperature_mean_j, 3),
        round(soc_mean_i, 3),
        round(current_mean_i, 3),
    )
    theta_array = {}
    theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_TROLS_method")
def LSB_TROLS_method(method_input, solve_method):
    """
    Solve the 2RC ECM parameter estimation problem with nonlinear least squares.

    Input expected in method_input:
        - method_input["phi_array"] : np.ndarray with shape (N, 5)
            Columns must be:
            [y[k-1], y[k-2], i[k], i[k-1], i[k-2]]

        - method_input["y_array"] : np.ndarray with shape (N,)
            Target:
            y[k] = U_term[k] - U_oc[k]

        - method_input["T_s"] : float
            Fixed sampling time

    Optional keys in method_input:
        - "initial_guess" : [R0, R1, tau1, R2, tau2]
        - "bounds" : (lower_bounds, upper_bounds)
        - "max_n_samples" : int
        - "loss" : str for scipy.optimize.least_squares
        - "f_scale" : float for scipy.optimize.least_squares

    Parameters returned:
        theta_array = [R0, R1, tau1, R2, tau2]
    """

    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]

    # # Column used to detect changes
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     ref_col = phi_array[:, 1]
    # elif model_type == "thevenin_2rc":
    #     ref_col = phi_array[:, 2]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)
    # # Append new time column
    # phi_array = np.hstack((phi_array, time_col))
    
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     mask = phi_array[:, 1] != 0
    # elif model_type == "thevenin_2rc":
    #     mask = phi_array[:, 2] != 0
    # phi_array = phi_array[mask]
    # y_array = y_array[mask]

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    # u_term_k_array = u_term_k_array[mask]
    # u_term_km1_array = u_term_km1_array[mask]
    # u_term_km2_array = u_term_km2_array[mask]
    # u_ocv_k_array = u_ocv_k_array[mask]
    # u_ocv_km1_array = u_ocv_km1_array[mask]
    # u_ocv_km2_array = u_ocv_km2_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    # i_bulk_km1_array = i_bulk_km1_array[mask]
    # i_bulk_km2_array = i_bulk_km2_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            # "u_term_k_array": u_term_k_array,
            # "u_term_km1_array": u_term_km1_array,
            # "u_term_km2_array": u_term_km2_array,
            # "u_ocv_k_array": u_ocv_k_array,
            # "u_ocv_km1_array": u_ocv_km1_array,
            # "u_ocv_km2_array": u_ocv_km2_array,
            "i_bulk_k_array": i_bulk_k_array,
            # "i_bulk_km1_array": i_bulk_km1_array,
            # "i_bulk_km2_array": i_bulk_km2_array,
        })
    #! This method is not good for long amount of data, let's figure out first how to make the slices of data to local estimation.
    
    theta_array_i = np.empty(0)
    #* Declare model functions

    

    if model_type in ["rint", "thevenin_0rc"]:
        # Solve 0RC
        theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_0RC, phi_array, y_array, initial_guess[0], lb[0], ub[0], [0.001])[0]
    if model_type in ["thevenin_1rc"]:
        # Solve 1RC
        theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_1RC, phi_array, y_array, initial_guess[:3], lb[:3], ub[:3], [0.001, 0.001, 1])[0]
    if model_type in ["thevenin_2rc"]:
        # Solve 2RC
        theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_2RC, phi_array, y_array, initial_guess, lb, ub, [0.001, 0.001, 1, 0.001, 1])[0] 

    #! This values won't affect to the parameter function since the idea is that, for all of them, the value is the same. It is just to keep the structure
    temperature_mean_j = 25
    soc_mean_i = 0.5
    current_mean_i = 1

    key = (
        round(temperature_mean_j, 3),
        round(soc_mean_i, 3),
        round(current_mean_i, 3),
    )
    theta_array = {}
    theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_SLSQP_method")
def LSB_SLSQP_method(method_input, solve_method):
    """
    Solve the 2RC ECM parameter estimation problem with nonlinear least squares.

    Input expected in method_input:
        - method_input["phi_array"] : np.ndarray with shape (N, 5)
            Columns must be:
            [y[k-1], y[k-2], i[k], i[k-1], i[k-2]]

        - method_input["y_array"] : np.ndarray with shape (N,)
            Target:
            y[k] = U_term[k] - U_oc[k]

        - method_input["T_s"] : float
            Fixed sampling time

    Optional keys in method_input:
        - "initial_guess" : [R0, R1, tau1, R2, tau2]
        - "bounds" : (lower_bounds, upper_bounds)
        - "max_n_samples" : int
        - "loss" : str for scipy.optimize.least_squares
        - "f_scale" : float for scipy.optimize.least_squares

    Parameters returned:
        theta_array = [R0, R1, tau1, R2, tau2]
    """

    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]

    # # Column used to detect changes
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     ref_col = phi_array[:, 1]
    # elif model_type == "thevenin_2rc":
    #     ref_col = phi_array[:, 2]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)
    # # Append new time column
    # phi_array = np.hstack((phi_array, time_col))
    
    # if model_type == "thevenin_1rc":
    # #* Delete row with current equal to zero
    #     mask = phi_array[:, 1] != 0
    # elif model_type == "thevenin_2rc":
    #     mask = phi_array[:, 2] != 0
    # phi_array = phi_array[mask]
    # y_array = y_array[mask]

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    # u_term_k_array = u_term_k_array[mask]
    # u_term_km1_array = u_term_km1_array[mask]
    # u_term_km2_array = u_term_km2_array[mask]
    # u_ocv_k_array = u_ocv_k_array[mask]
    # u_ocv_km1_array = u_ocv_km1_array[mask]
    # u_ocv_km2_array = u_ocv_km2_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    # i_bulk_km1_array = i_bulk_km1_array[mask]
    # i_bulk_km2_array = i_bulk_km2_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            # "u_term_k_array": u_term_k_array,
            # "u_term_km1_array": u_term_km1_array,
            # "u_term_km2_array": u_term_km2_array,
            # "u_ocv_k_array": u_ocv_k_array,
            # "u_ocv_km1_array": u_ocv_km1_array,
            # "u_ocv_km2_array": u_ocv_km2_array,
            "i_bulk_k_array": i_bulk_k_array,
            # "i_bulk_km1_array": i_bulk_km1_array,
            # "i_bulk_km2_array": i_bulk_km2_array,
        })
    #! This method is not good for long amount of data, let's figure out first how to make the slices of data to local estimation.
    
    theta_array_i = np.empty(0)
    #* Declare model functions
    
    if model_type in ["rint", "thevenin_0rc"]:
        # Solve 0RC
        theta_array_i = SLSQP_solve(solve_method, SLSQP_model_0RC, phi_array, y_array, initial_guess, lb, ub)
    if model_type in ["thevenin_1rc"]:
        # Solve 1RC
        theta_array_i = SLSQP_solve(solve_method, SLSQP_model_1RC, phi_array, y_array, initial_guess, lb, ub)
    if model_type in ["thevenin_2rc"]:
        # Solve 2RC
        theta_array_i = SLSQP_solve(solve_method, SLSQP_model_2RC, phi_array, y_array, initial_guess, lb, ub)

    #! This values won't affect to the parameter function since the idea is that, for all of them, the value is the same. It is just to keep the structure
    temperature_mean_j = 25
    soc_mean_i = 0.5
    current_mean_i = 1

    key = (
        round(temperature_mean_j, 3),
        round(soc_mean_i, 3),
        round(current_mean_i, 3),
    )
    theta_array = {}
    theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_MWLS_method")
def LSB_MWLS_method(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """
    model_type = method_input["model_type"]
    # Here a function to make the slices should be done
    temperature_array = method_input["temperature_array"]
    cycle_array = method_input["cycle_array"]
    soc_array = method_input["soc_array"]

    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    T_s = method_input["T_s"]

    theta_array = {}
    # First mask: per temperature value
    #TODO complete this mask generation bc I will skip it at the moment. It should be grouped by 5degree or something
    temperature_mean_j = 25

    # Second mask: per cycle
    for cycle_value in np.unique(cycle_array):
        mask = cycle_array == cycle_value

        # Jump to the next value if the variation of the soc is too big (>5%SOC)
        soc_array_i = soc_array[mask]
        if (max(soc_array_i)-min(soc_array_i))> 0.05:
            continue

        soc_mean_i = np.mean(soc_array_i)

        if soc_mean_i < 0.005:
            continue


        phi_array_i = phi_array[mask]
        y_array_i = y_array[mask]

        if model_type == "thevenin_0rc" or model_type == "rint":
            current_mean_i = np.mean(phi_array_i[:, 0])
        elif model_type == "thevenin_1rc":
            current_mean_i = np.mean(phi_array_i[:, 1])
        elif model_type == "thevenin_2rc":
            current_mean_i = np.mean(phi_array_i[:, 2])
        
        try:
            if solve_method == "pinv":
                # Solve θ = (Φ.T · Φ)^(-1) · Φ.T · Y
                theta_array_i = np.linalg.pinv(phi_array_i.T @ phi_array_i) @ phi_array_i.T @ y_array_i
            
            elif solve_method == "lstsq":
                # Solve min ||Y - Φ · θ||
                theta_array_i, residuals, rank, s = np.linalg.lstsq(phi_array_i, y_array_i, rcond = None) #TODO Check this to remember how this np function works

            elif solve_method == "solve":
                # Solve A·θ = b
                A = phi_array_i.T @ phi_array_i
                b = phi_array_i.T @ y_array_i
                theta_array_i = np.linalg.solve(A, b)   
            
            elif solve_method == "cholesky":
                # Solve A·θ = b with A = L · L.T
                A = phi_array_i.T @ phi_array_i
                b = phi_array_i.T @ y_array_i
                c, lower = sp.linalg.cho_factor(A)                 # Cholesky
                theta_array_i = sp.linalg.cho_solve((c, lower), b)             # solve Ax=b without the inverse
        except:
            continue


        key = (
            round(temperature_mean_j, 3),
            round(soc_mean_i, 3),
            round(current_mean_i, 3),
        )

        theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_MWNLLS_method")
def LSB_MWNLLS_method(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """
    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]
    soc_array = method_input["soc_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    cycle_array = cycle_array[mask]
    soc_array = soc_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            "i_bulk_k_array": i_bulk_k_array,
        })

    theta_array = {}
    # First mask: per temperature value
    #TODO complete this mask generation bc I will skip it at the moment. It should be grouped by 5degree or something
    temperature_mean_j = 25

    # Second mask: per cycle
    for cycle_value in tqdm(np.unique(cycle_array)):
        mask = cycle_array == cycle_value

        # Jump to the next value if the variation of the soc is too big (>5%SOC)
        soc_array_i = soc_array[mask]
        if (max(soc_array_i)-min(soc_array_i))> 0.05:
            continue

        soc_mean_i = np.mean(soc_array_i)

        if soc_mean_i < 0.005:
            continue


        phi_array_i = phi_array[mask]
        y_array_i = y_array[mask]

        current_mean_i = np.mean(phi_array_i["i_bulk_k_array"])

        try:
            theta_array_i = np.empty(0)
            skip_0rc = True
            
            if skip_0rc:
                if model_type in ["rint", "thevenin_0rc"]:
                    # Solve 0RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_0RC, phi_array_i, y_array_i, initial_guess[0], lb[0], ub[0], [0.001])
                    phi_array_i.attrs["r_0"] = theta_array_raw[0][0]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_0"]])
                if model_type in ["thevenin_1rc", "thevenin_2rc"]:
                    # Solve 1RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_1RC_skip_0rc, phi_array_i, y_array_i, initial_guess[:3], lb[:3], ub[:3], [0.001, 0.001, 1])
                    phi_array_i.attrs["r_0"] = theta_array_raw[0][0]
                    phi_array_i.attrs["r_1"] = theta_array_raw[0][1]
                    phi_array_i.attrs["tau_1"] = theta_array_raw[0][2]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_0"], phi_array_i.attrs["r_1"], phi_array_i.attrs["tau_1"]])
                if model_type in ["thevenin_2rc"]:
                    # Solve 2RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_2RC, phi_array_i, y_array_i, initial_guess[3:5], lb[3:5], ub[3:5], [0.001, 1])
                    phi_array_i.attrs["r_2"] = theta_array_raw[0][0]
                    phi_array_i.attrs["tau_2"] = theta_array_raw[0][1]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_2"], phi_array_i.attrs["tau_2"]])

            else:
                if model_type in ["rint", "thevenin_0rc", "thevenin_1rc", "thevenin_2rc"]:
                    # Solve 0RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_0RC, phi_array_i, y_array_i, initial_guess[0], lb[0], ub[0], [0.001])
                    phi_array_i.attrs["r_0"] = theta_array_raw[0][0]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_0"]])
                if model_type in ["thevenin_1rc", "thevenin_2rc"]:
                    # Solve 1RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_1RC, phi_array_i, y_array_i, initial_guess[1:3], lb[1:3], ub[1:3], [0.001, 1])
                    phi_array_i.attrs["r_1"] = theta_array_raw[0][0]
                    phi_array_i.attrs["tau_1"] = theta_array_raw[0][1]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_1"], phi_array_i.attrs["tau_1"]])
                if model_type in ["thevenin_2rc"]:
                    # Solve 2RC
                    theta_array_raw = NLLS_solve_curve_fit(solve_method, NLLS_model_2RC, phi_array_i, y_array_i, initial_guess[3:5], lb[3:5], ub[3:5], [0.001, 1])
                    phi_array_i.attrs["r_2"] = theta_array_raw[0][0]
                    phi_array_i.attrs["tau_2"] = theta_array_raw[0][1]
                    theta_array_i = np.append(theta_array_i, [phi_array_i.attrs["r_2"], phi_array_i.attrs["tau_2"]])
        except:
            continue


        key = (
            round(temperature_mean_j, 3),
            round(soc_mean_i, 3),
            round(current_mean_i, 3),
        )

        theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_MWSLSQP_method")
def LSB_MWSLSQP_method(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """
    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]
    soc_array = method_input["soc_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    cycle_array = cycle_array[mask]
    soc_array = soc_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            "i_bulk_k_array": i_bulk_k_array,
        })

    theta_array = {}
    # First mask: per temperature value
    #TODO complete this mask generation bc I will skip it at the moment. It should be grouped by 5degree or something
    temperature_mean_j = 25

    # Second mask: per cycle
    for cycle_value in tqdm(np.unique(cycle_array)):
        mask = cycle_array == cycle_value

        # Jump to the next value if the variation of the soc is too big (>5%SOC)
        soc_array_i = soc_array[mask]
        if (max(soc_array_i)-min(soc_array_i))> 0.05:
            continue

        soc_mean_i = np.mean(soc_array_i)

        if soc_mean_i < 0.005:
            continue


        phi_array_i = phi_array[mask]
        y_array_i = y_array[mask]

        current_mean_i = np.mean(phi_array_i["i_bulk_k_array"])

        try:
            theta_array_i = np.empty(0)
            #* Declare model functions
            
            if model_type in ["rint", "thevenin_0rc"]:
                # Solve 0RC
                theta_array_i = SLSQP_solve(solve_method, SLSQP_model_0RC, phi_array_i, y_array_i, initial_guess, lb, ub)
            if model_type in ["thevenin_1rc"]:
                # Solve 1RC
                theta_array_i = SLSQP_solve(solve_method, SLSQP_model_1RC, phi_array_i, y_array_i, initial_guess, lb, ub)
            if model_type in ["thevenin_2rc"]:
                # Solve 2RC
                theta_array_i = SLSQP_solve(solve_method, SLSQP_model_2RC, phi_array_i, y_array_i, initial_guess, lb, ub)
        except:
            continue


        key = (
            round(temperature_mean_j, 3),
            round(soc_mean_i, 3),
            round(current_mean_i, 3),
        )

        theta_array[key] = theta_array_i

    return theta_array

@LSB_method.register("LSB_MWTROLS_method")
def LSB_MWTROLS_method(method_input, solve_method):
    """
    This method solves the equations of the Least Square method for parameter estimation.
    The type of model should not affect to the method as long as the provided arrays are sensible.
    For this method is required a regular grid of data with a constant T_s.
    
    :param phi_array: Description
    :param y_array: Description
    :param solve_method: Description
    :param T_s: Description
    """
    phi_array = method_input["phi_array"]
    y_array = method_input["y_array"]
    u_term_k_array = method_input["u_term_k_array"] 
    u_term_km1_array = method_input["u_term_km1_array"] 
    u_term_km2_array = method_input["u_term_km2_array"] 
    u_ocv_k_array = method_input["u_ocv_k_array"] 
    u_ocv_km1_array = method_input["u_ocv_km1_array"] 
    u_ocv_km2_array = method_input["u_ocv_km2_array"] 
    i_bulk_k_array = method_input["i_bulk_k_array"] 
    i_bulk_km1_array = method_input["i_bulk_km1_array"] 
    i_bulk_km2_array = method_input["i_bulk_km2_array"] 
    T_s = method_input["T_s"]
    model_type = method_input["model_type"]
    lb = method_input["lower_bound"]
    ub = method_input["upper_bound"]
    initial_guess = method_input["initial_guess"]
    cycle_array = method_input["cycle_array"]
    soc_array = method_input["soc_array"]

    #* Add temporal column
    n_rows = phi_array.shape[0]
    

    # Detect the start of each new segment
    change_mask = np.empty(n_rows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cycle_array[1:] != cycle_array[:-1]

    # Segment id for each row: 0, 0, 0, 1, 1, 2, ...
    segment_id = cycle_array
    segment_id = segment_id.astype(int)

    # Row index where each segment starts
    segment_start_idx = np.flatnonzero(change_mask)

    # Time counter reset at the start of each segment
    time_col = ((np.arange(n_rows, dtype=np.float64) - segment_start_idx[segment_id]) * T_s).reshape(-1, 1)
    time_col = np.ravel(time_col)

    mask = i_bulk_k_array != 0

    time_col = time_col[mask]

    y_array = y_array[mask]
    i_bulk_k_array = i_bulk_k_array[mask]
    cycle_array = cycle_array[mask]
    soc_array = soc_array[mask]

    phi_array = pd.DataFrame({
            "time_col": time_col,
            "i_bulk_k_array": i_bulk_k_array,
        })

    theta_array = {}
    # First mask: per temperature value
    #TODO complete this mask generation bc I will skip it at the moment. It should be grouped by 5degree or something
    temperature_mean_j = 25

    # Second mask: per cycle
    for cycle_value in tqdm(np.unique(cycle_array)):
        mask = cycle_array == cycle_value

        # Jump to the next value if the variation of the soc is too big (>5%SOC)
        soc_array_i = soc_array[mask]
        if (max(soc_array_i)-min(soc_array_i))> 0.05:
            continue

        soc_mean_i = np.mean(soc_array_i)

        if soc_mean_i < 0.005:
            continue


        phi_array_i = phi_array[mask]
        y_array_i = y_array[mask]

        current_mean_i = np.mean(phi_array_i["i_bulk_k_array"])

        try:            
            if model_type in ["rint", "thevenin_0rc"]:
                # Solve 0RC
                theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_0RC, phi_array_i, y_array_i, initial_guess[0], lb[0], ub[0], [0.001])[0]
            if model_type in ["thevenin_1rc"]:
                # Solve 1RC
                theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_1RC, phi_array_i, y_array_i, initial_guess[:3], lb[:3], ub[:3], [0.001, 0.001, 1])[0]
            if model_type in ["thevenin_2rc"]:
                # Solve 2RC
                theta_array_i = TROLS_solve_curve_fit(solve_method, TROLS_model_2RC, phi_array_i, y_array_i, initial_guess, lb, ub, [0.001, 0.001, 1, 0.001, 1])[0] 
        except:
            continue


        key = (
            round(temperature_mean_j, 3),
            round(soc_mean_i, 3),
            round(current_mean_i, 3),
        )

        theta_array[key] = theta_array_i

    return theta_array


# endregion

def NLLS_solve_curve_fit(solve_method, model_function, phi_array, y_array, initial_guess, lb, ub, x_scale):
    if solve_method == "sp.optimize.curve_fit.lm":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
        )
    elif solve_method == "sp.optimize.curve_fit.trf":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
            bounds=(lb, ub),
            method="trf",
            x_scale = x_scale,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=2
        )
    elif solve_method == "sp.optimize.curve_fit.dogbox":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
            bounds=(lb, ub),
            method="dogbox",
            x_scale = x_scale,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=2
        )
    
    return theta_array

def TROLS_solve_curve_fit(solve_method, model_function, phi_array, y_array, initial_guess, lb, ub, x_scale):
    if solve_method == "sp.optimize.curve_fit.lm":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
        )
    elif solve_method == "sp.optimize.curve_fit.trf":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
            bounds=(lb, ub),
            method="trf",
            x_scale = x_scale,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=2
        )
    elif solve_method == "sp.optimize.curve_fit.dogbox":
        theta_array = sp.optimize.curve_fit(
            model_function, 
            phi_array, 
            y_array,
            p0=initial_guess,
            bounds=(lb, ub),
            method="dogbox",
            x_scale = x_scale,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=2
        )
    
    return theta_array

def NLLS_model_0RC(phi, r_0):
    i_k = np.array(phi["i_bulk_k_array"])

    y_hat = - i_k * r_0

    return y_hat

def NLLS_model_1RC(phi, r_1, tau_1):
    i_k = np.array(phi["i_bulk_k_array"])
    temp = np.array(phi["time_col"])

    r_0 = phi.attrs["r_0"]

    y_hat = - i_k * (r_0 + r_1 *(1 - np.exp(-temp / (tau_1))))

    return y_hat

def NLLS_model_1RC_skip_0rc(phi, r_0, r_1, tau_1):
    i_k = np.array(phi["i_bulk_k_array"])
    temp = np.array(phi["time_col"])

    y_hat = - i_k * (r_0 + r_1 *(1 - np.exp(-temp / (tau_1))))

    return y_hat

def NLLS_model_2RC(phi, r_2, tau_2):
    i_k = phi["i_bulk_k_array"]
    temp = np.array(phi["time_col"])

    r_0 = phi.attrs["r_0"]
    r_1 = phi.attrs["r_1"]
    tau_1 = phi.attrs["tau_1"]

    y_hat = - i_k *(r_0 + r_1 * (1 - np.exp(-temp / (tau_1))) + r_2 * (1 - np.exp(-temp / (tau_2))))

    return y_hat

def TROLS_model_0RC(phi, r_0):
    i_k = np.array(phi["i_bulk_k_array"])

    y_hat = - i_k * r_0

    return y_hat

def TROLS_model_1RC(phi, r_0, r_1, tau_1):
    i_k = np.array(phi["i_bulk_k_array"])
    temp = np.array(phi["time_col"])

    y_hat = - i_k * (r_0 + r_1 *(1 - np.exp(-temp / (tau_1))))

    return y_hat

def TROLS_model_2RC(phi, r_0, r_1, tau_1, r_2, tau_2):
    i_k = phi["i_bulk_k_array"]
    temp = np.array(phi["time_col"])

    y_hat = - i_k *(r_0 + r_1 * (1 - np.exp(-temp / (tau_1))) + r_2 * (1 - np.exp(-temp / (tau_2))))

    return y_hat

def SLSQP_solve(solve_method, model_function, phi_array, y_array, initial_guess, lb, ub):   

    results = pyslsqp.optimize(
            x0=initial_guess,
            obj=lambda x: model_function(x, phi_array, y_array),
            xl=lb,
            xu=ub
        )

    return results["x"]

def SLSQP_model_0RC(x, phi, y):
    i_k = np.array(phi["i_bulk_k_array"])

    r_0 = x[0]

    y_hat = - i_k * r_0

    return np.sum((y - y_hat)**2)

def SLSQP_model_1RC(x, phi, y):
    i_k = np.array(phi["i_bulk_k_array"])
    temp = np.array(phi["time_col"])

    r_0 = x[0]
    r_1 = x[1]
    tau_1 = x[2]

    y_hat = - i_k * (r_0 + r_1 *(1 - np.exp(-temp / (tau_1))))

    return np.sum((y - y_hat)**2)

def SLSQP_model_2RC(x, phi, y):
    i_k = phi["i_bulk_k_array"]
    temp = np.array(phi["time_col"])

    r_0 = x[0]
    r_1 = x[1]
    tau_1 = x[2]
    r_2 = x[3]
    tau_2 = x[4]

    y_hat = - i_k *(r_0 + r_1 * (1 - np.exp(-temp / (tau_1))) + r_2 * (1 - np.exp(-temp / (tau_2))))

    return np.sum((y - y_hat)**2)