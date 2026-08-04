from __future__ import annotations
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
import scipy as sp


def data_to_method(data, output_vars, data_ts, method_group, method_type, model_type, initial_guess, lower_bound, upper_bound, study_case):
    
    method_input = method_input_dict_generator()

    #* List of methods that requires a regular grid.
    # Be careful in this step with the current so it is not interpolated.
    # Regularize the period time in the data set.
    #! This part does not make sense any more since the study is limited to regular period data.
    # if method_type in ["LS","MWLS"]:
    #     data = make_constant_period(data.copy(), method_ts, ["Current [A]", "Cycle", "Step"], "time [s]")

    # Generate the method_input parameters
    #! I think that all the methods will receive the same structure, only dependant on the EECM to parametrise
    # if method_type in ["LS","MWLS", "NLLS", "MWNLLS"]:
        
    method_input[method_group][method_type] = close_function_adapter(data, output_vars, model_type, data_ts, initial_guess, lower_bound, upper_bound, study_case)


    return method_input

def close_function_adapter(data, output_vars, model_type, data_ts, initial_guess, lower_bound, upper_bound, study_case):

    input_dict = {}

    # Terminal voltage and previous values
    u_term_k_array = data["Voltage [V]"]
    u_term_km1_array = data["Voltage [V]"].shift(1)
    u_term_km2_array = data["Voltage [V]"].shift(2)

    # Open-circuit voltage and previous values
    u_ocv_k_array = data["Open-circuit voltage [V]"]
    u_ocv_km1_array = data["Open-circuit voltage [V]"].shift(1)
    u_ocv_km2_array = data["Open-circuit voltage [V]"].shift(2)

    # Bulk current and previous values
    i_bulk_k_array = data["Current [A]"]
    i_bulk_km1_array = data["Current [A]"].shift(1)
    i_bulk_km2_array = data["Current [A]"].shift(2)

    # Additional variables
    soc = data["SOC [-]"]
    temperature = data["X-averaged cell temperature [K]"]
    cycle = data["Cycle"]

    if model_type == "rint" or model_type == "thevenin_0rc":
        # y_array calc    
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc        
        phi_array = np.array([- i_bulk_k_array]).T      #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT

        # Boundaries and initial guess
        input_dict["initial_guess"] = np.array(initial_guess[0])  # [R0]
        input_dict["lower_bound"] = np.array(lower_bound[0])  # [R0]
        input_dict["upper_bound"] = np.array(upper_bound[0])  # [R0]

    elif model_type == "thevenin_1rc":
        # y_array calc  
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc
        phi_array = np.array([u_term_km1_array - u_ocv_km1_array, - i_bulk_k_array, - i_bulk_km1_array]).T  #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT

        # Boundaries and initial guess
        input_dict["initial_guess"] = np.array(initial_guess[:3])  # [R0, R1, tau1]
        input_dict["lower_bound"] = np.array(lower_bound[:3])  # [R0, R1, tau1]
        input_dict["upper_bound"] = np.array(upper_bound[:3])  # [R0, R1, tau1]

    elif model_type == "thevenin_2rc":
        # y_array calc      
        y_array = np.array(u_term_k_array - u_ocv_k_array)

        # phi_array calc
        phi_array = np.array([u_term_km1_array - u_ocv_km1_array, u_term_km2_array - u_ocv_km2_array, - i_bulk_k_array, - i_bulk_km1_array, - i_bulk_km2_array]).T    #! ATTENTION: THIS CHANGE IS BECAUSE THE LS METHOD (WHICH USE PHI_ARRAY) EXPECT DISCHARGE WITH NEGATIVE CURRENT
        
        # Boundaries and initial guess
        input_dict["initial_guess"] = np.array(initial_guess)  # [R0, R1, tau1, R2, tau2]
        input_dict["lower_bound"] = np.array(lower_bound)  # [R0, R1, tau1, R2, tau2]
        input_dict["upper_bound"] = np.array(upper_bound)  # [R0, R1, tau1, R2, tau2]
 

    # generate the dictionary and detele the first two rows of each to avoid nan
    delete_array = [0,1]    #* If at some point the number of RC branches is bigger, this list should be bigger too.
    input_dict["y_array"] = np.delete(y_array, delete_array, 0)
    input_dict["phi_array"] = np.delete(phi_array, delete_array, 0)
    input_dict["u_term_k_array"] = np.delete(np.array(u_term_k_array), delete_array, 0)
    input_dict["u_term_km1_array"] = np.delete(np.array(u_term_km1_array), delete_array, 0)
    input_dict["u_term_km2_array"] = np.delete(np.array(u_term_km2_array), delete_array, 0)
    input_dict["u_ocv_k_array"] = np.delete(np.array(u_ocv_k_array), delete_array, 0)
    input_dict["u_ocv_km1_array"] = np.delete(np.array(u_ocv_km1_array), delete_array, 0)
    input_dict["u_ocv_km2_array"] = np.delete(np.array(u_ocv_km2_array), delete_array, 0)
    input_dict["i_bulk_k_array"] = np.delete(np.array(i_bulk_k_array), delete_array, 0)
    input_dict["i_bulk_km1_array"] = np.delete(np.array(i_bulk_km1_array), delete_array, 0)
    input_dict["i_bulk_km2_array"] = np.delete(np.array(i_bulk_km2_array), delete_array, 0)
    input_dict["T_s"] = data_ts
    input_dict["model_type"] = model_type
    input_dict["soc_array"] = np.delete(soc, delete_array, 0)
    input_dict["temperature_array"] = np.delete(temperature, delete_array, 0) - 273.15 # From kelvin to celsius
    input_dict["cycle_array"] = np.delete(cycle, delete_array, 0)


    return input_dict

def method_input_dict_generator():
    """
    This function generate the standardised dictionary with the output required for each method.
    """
    method_input = {
        "LSB":{
            "LS":{
                "y_array": "np.array()",
                "phi_array": "np.array()",
                "T_s": "float()"
            }
        }
    }

    return method_input

def make_constant_period(
    data: pd.DataFrame,
    t_s: float,
    discret_columns: List[str],
    time_column: str,
) -> pd.DataFrame:
    """
    Resample a DataFrame onto a constant time grid of step `t_s` seconds.

    - Columns in `discret_columns` are treated as piecewise-constant signals and
      are resampled with Zero-Order Hold (ZOH): forward-fill (ffill) to avoid
      ramping artifacts.
    - All other columns are linearly interpolated in time.

    The output keeps the same column names and returns `time_column` as a column
    (not as index), preserving the original column order.
    """
    if t_s <= 0:
        raise ValueError("t_s must be > 0 (seconds).")
    if time_column not in data.columns:
        raise KeyError(f"time_column '{time_column}' not found in data.")
    missing = [c for c in discret_columns if c not in data.columns]
    if missing:
        raise KeyError(f"discret_columns not found in data: {missing}")

    df = data.copy()

    # Detect time type and build a new constant grid
    t = df[time_column]
    step = pd.to_timedelta(t_s, unit="s")

    if pd.api.types.is_datetime64_any_dtype(t):
        idx = pd.DatetimeIndex(t)
        new_index = pd.date_range(start=idx.min(), end=idx.max(), freq=step)
        interp_method = "time"
        out_time_values = new_index
    elif pd.api.types.is_timedelta64_dtype(t):
        idx = pd.TimedeltaIndex(t)
        new_index = pd.timedelta_range(start=idx.min(), end=idx.max(), freq=step)
        interp_method = "time"
        out_time_values = new_index
    else:
        # Numeric time assumed in seconds
        t_num = pd.to_numeric(t, errors="raise").astype(float).to_numpy()
        t0 = float(np.nanmin(t_num))
        t1 = float(np.nanmax(t_num))
        n = int(np.floor((t1 - t0) / t_s + 1e-12)) + 1
        new_vals = t0 + np.arange(n) * t_s
        idx = pd.Index(t_num, name=time_column)
        new_index = pd.Index(new_vals, name=time_column)
        interp_method = "values"
        out_time_values = new_vals

    # Sort, de-duplicate timestamps, and set index
    df = df.assign(**{time_column: idx}).sort_values(time_column)
    df = df.drop_duplicates(subset=[time_column], keep="last").set_index(time_column)

    discrete = list(discret_columns)
    continuous = [c for c in df.columns if c not in discrete]

    # Work on the union index so interpolation/ffill "sees" original points
    work_index = df.index.union(new_index)
    work = df.reindex(work_index)

    out = pd.DataFrame(index=new_index)

    # Discrete columns: ZOH (ffill) to avoid ramping artifacts
    if discrete:
        disc = work[discrete].ffill().bfill()
        out[discrete] = disc.reindex(new_index)

    # Continuous columns: time interpolation
    if continuous:
        cont = work[continuous].interpolate(method=interp_method, limit_direction="both")
        cont = cont.ffill().bfill()
        out[continuous] = cont.reindex(new_index)

    # Put time back as a column, keep original column order, and drop the index
    out.insert(0, time_column, out_time_values)
    out = out[data.columns].reset_index(drop=True)

    return out

def select_arbin_current_range(max_abs_current):
    """
    Select the smallest Arbin current range that contains the full current signal.
    """
    available_ranges = np.array([0.02, 0.5, 5.0, 30.0])  # A

    valid_ranges = available_ranges[max_abs_current <= available_ranges]

    if len(valid_ranges) == 0:
        raise ValueError(
            f"Current exceeds the maximum Arbin range: "
            f"{max_abs_current:.6f} A > 30 A"
        )

    return float(valid_ranges[0])

def apply_arbin_lbt_measurement_error(
    data,
    voltage_col="Voltage [V]",
    current_col="Current [A]",
    current_range=None,
    seed=123,
    coverage_factor=3.0,
):
    """
    Apply synthetic Arbin LBT21084HC measurement error to voltage and current.

    The dataframe structure is preserved:
    - same number of rows
    - same number of columns
    - same column names

    Only the values in voltage_col and current_col are overwritten locally.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.

    voltage_col : str
        Voltage column name in V.

    current_col : str
        Current column name in A.

    current_range : float, optional
        Active current full-scale range in A. Valid values are 30.0, 5.0, 0.5, 0.02.
        If None, the smallest valid range containing the full current signal is selected.

    seed : int
        Random seed for reproducibility.

    coverage_factor : float
        Factor used to convert specification bounds to standard deviations.
        For example, coverage_factor=3.0 interprets the +/- specification as
        approximately a 3-sigma bound.

    Returns
    -------
    pd.DataFrame
        Dataframe with the same structure and noisy voltage/current values.

    dict
        Metadata describing the applied error.
    """
    if voltage_col not in data.columns:
        raise KeyError(f"Voltage column not found: {voltage_col}")

    if current_col not in data.columns:
        raise KeyError(f"Current column not found: {current_col}")

    data_error = data.copy()

    rng = np.random.default_rng(seed)

    voltage_true = data_error[voltage_col].to_numpy(dtype=float)
    current_true = data_error[current_col].to_numpy(dtype=float)

    # -------------------------------------------------
    # Arbin LBT21084HC measurement specifications
    # -------------------------------------------------
    voltage_fsr = 5.0  # V, 0 to 5 V range

    current_ranges = [30.0, 5.0, 0.5, 0.02]  # A

    precision_fraction = 0.0001  # +/- 0.01% FSR
    accuracy_fraction = 0.0002   # +/- 0.02% FSR

    if current_range is None:
        max_abs_current = np.nanmax(np.abs(current_true))
        current_range = select_arbin_current_range(max_abs_current)

    if current_range not in current_ranges:
        raise ValueError(
            f"Invalid current range: {current_range}. "
            f"Use one of {current_ranges} A."
        )

    # -------------------------------------------------
    # Absolute precision and accuracy bounds
    # -------------------------------------------------
    voltage_precision_abs = precision_fraction * voltage_fsr
    voltage_accuracy_abs = accuracy_fraction * voltage_fsr

    current_precision_abs = precision_fraction * current_range
    current_accuracy_abs = accuracy_fraction * current_range

    # -------------------------------------------------
    # Split total accuracy into precision noise and systematic bias
    # -------------------------------------------------
    voltage_bias_bound = np.sqrt(
        max(voltage_accuracy_abs**2 - voltage_precision_abs**2, 0.0)
    )

    current_bias_bound = np.sqrt(
        max(current_accuracy_abs**2 - current_precision_abs**2, 0.0)
    )

    voltage_noise_std = voltage_precision_abs / coverage_factor
    current_noise_std = current_precision_abs / coverage_factor

    voltage_bias_std = voltage_bias_bound / coverage_factor
    current_bias_std = current_bias_bound / coverage_factor

    # Constant bias for the whole experiment
    voltage_bias = rng.normal(loc=0.0, scale=voltage_bias_std)
    current_bias = rng.normal(loc=0.0, scale=current_bias_std)

    # Time-dependent measurement noise
    voltage_noise = rng.normal(
        loc=0.0,
        scale=voltage_noise_std,
        size=voltage_true.shape,
    )

    current_noise = rng.normal(
        loc=0.0,
        scale=current_noise_std,
        size=current_true.shape,
    )

    # -------------------------------------------------
    # Overwrite the original columns
    # -------------------------------------------------
    data_error[voltage_col] = voltage_true + voltage_bias + voltage_noise
    data_error[current_col] = current_true + current_bias + current_noise

    metadata = {
        "measurement_error_applied": True,
        "voltage_fsr_V": voltage_fsr,
        "current_range_A": current_range,
        "precision_fraction": precision_fraction,
        "accuracy_fraction": accuracy_fraction,
        "coverage_factor": coverage_factor,
        "voltage_precision_abs_V": voltage_precision_abs,
        "voltage_accuracy_abs_V": voltage_accuracy_abs,
        "voltage_bias_bound_V": voltage_bias_bound,
        "voltage_noise_std_V": voltage_noise_std,
        "voltage_bias_std_V": voltage_bias_std,
        "voltage_bias_used_V": voltage_bias,
        "current_precision_abs_A": current_precision_abs,
        "current_accuracy_abs_A": current_accuracy_abs,
        "current_bias_bound_A": current_bias_bound,
        "current_noise_std_A": current_noise_std,
        "current_bias_std_A": current_bias_std,
        "current_bias_used_A": current_bias,
    }

    return data_error, metadata

def apply_soc_range_error(
    soc,
    range_reduction=0.05,
    clip=True,
):
    """
    Apply a synthetic SOC error caused by overestimating the available capacity.

    The first SOC value is assumed to be correct. Then, the SOC range is reduced
    around this initial value. This emulates the case where the battery reaches
    its voltage limits before the estimated SOC reaches 0 or 1.

    For example:
    - initial SOC = 1.0 -> range becomes 0.05 to 1.00
    - initial SOC = 0.0 -> range becomes 0.00 to 0.95
    - initial SOC = 0.8 -> range becomes 0.04 to 0.99

    Parameters
    ----------
    soc : pd.Series or array-like
        Original SOC values in per unit.

    range_reduction : float
        Fraction of the SOC range to reduce. For example, 0.05 means 5%.

    clip : bool
        If True, the resulting SOC values are clipped to [0, 1].

    Returns
    -------
    pd.Series or np.ndarray
        SOC values with range error applied. The output type is preserved when
        the input is a pandas Series.

    dict
        Metadata describing the applied SOC error.
    """
    if not 0.0 <= range_reduction < 1.0:
        raise ValueError(
            f"range_reduction must be in [0, 1). Received: {range_reduction}"
        )

    is_series = isinstance(soc, pd.Series)

    if is_series:
        soc_values = soc.to_numpy(dtype=float)
        soc_index = soc.index
        soc_name = soc.name
    else:
        soc_values = np.asarray(soc, dtype=float)
        soc_index = None
        soc_name = None

    if len(soc_values) == 0:
        raise ValueError("SOC array is empty.")

    soc_initial = float(soc_values[0])

    if not 0.0 <= soc_initial <= 1.0:
        raise ValueError(
            f"The initial SOC must be between 0 and 1. Received: {soc_initial}"
        )

    scale_factor = 1.0 - range_reduction

    soc_error_values = soc_initial + scale_factor * (soc_values - soc_initial)

    if clip:
        soc_error_values = np.clip(soc_error_values, 0.0, 1.0)

    soc_lower_limit = soc_initial * range_reduction
    soc_upper_limit = 1.0 - (1.0 - soc_initial) * range_reduction

    if is_series:
        soc_error = pd.Series(
            soc_error_values,
            index=soc_index,
            name=soc_name,
        )
    else:
        soc_error = soc_error_values

    metadata = {
        "soc_error_applied": True,
        "range_reduction": range_reduction,
        "scale_factor": scale_factor,
        "soc_initial": soc_initial,
        "soc_lower_limit_theoretical": soc_lower_limit,
        "soc_upper_limit_theoretical": soc_upper_limit,
        "clip": clip,
    }

    return soc_error, metadata

def apply_pseudo_discharge_ocv_error(
    u_ocv,
    error_fraction=0.02,
    ocv_full_range_v=None,
    ocv_min_v=None,
    ocv_max_v=None,
    clip=False,
):
    """
    Apply a synthetic OCV error caused by using a pseudo-OCV discharge curve.

    The pseudo-OCV discharge curve is assumed to include residual overpotential,
    which makes the estimated OCV lower than the true equilibrium OCV.

    Parameters
    ----------
    u_ocv : pd.Series or array-like
        Original OCV values in V.

    error_fraction : float
        Fraction of the full OCV operating range used as negative offset.
        For example, 0.02 means 2% of the full operating voltage range.

    ocv_full_range_v : float, optional
        Full OCV operating range in V. For example, 4.2 - 2.5.
        If None, the range is computed from ocv_min_v and ocv_max_v.
        If those are also None, the range is estimated from the input data.

    ocv_min_v : float, optional
        Minimum OCV operating value in V.

    ocv_max_v : float, optional
        Maximum OCV operating value in V.

    clip : bool
        If True, clip the corrected OCV to [ocv_min_v, ocv_max_v].
        Clipping is only applied if ocv_min_v and ocv_max_v are provided.

    Returns
    -------
    pd.Series or np.ndarray
        OCV values with pseudo-discharge error applied.

    dict
        Metadata describing the applied OCV error.
    """
    if not 0.0 <= error_fraction < 1.0:
        raise ValueError(
            f"error_fraction must be in [0, 1). Received: {error_fraction}"
        )

    is_series = isinstance(u_ocv, pd.Series)

    if is_series:
        u_ocv_values = u_ocv.to_numpy(dtype=float)
        u_ocv_index = u_ocv.index
        u_ocv_name = u_ocv.name
    else:
        u_ocv_values = np.asarray(u_ocv, dtype=float)
        u_ocv_index = None
        u_ocv_name = None

    if len(u_ocv_values) == 0:
        raise ValueError("OCV array is empty.")

    if ocv_full_range_v is None:
        if ocv_min_v is not None and ocv_max_v is not None:
            ocv_full_range_v = ocv_max_v - ocv_min_v
        else:
            ocv_full_range_v = np.nanmax(u_ocv_values) - np.nanmin(u_ocv_values)

    if ocv_full_range_v <= 0.0:
        raise ValueError(
            f"ocv_full_range_v must be positive. Received: {ocv_full_range_v}"
        )

    ocv_offset_v = error_fraction * ocv_full_range_v

    u_ocv_error_values = u_ocv_values - ocv_offset_v

    if clip and ocv_min_v is not None and ocv_max_v is not None:
        u_ocv_error_values = np.clip(
            u_ocv_error_values,
            ocv_min_v,
            ocv_max_v,
        )

    if is_series:
        u_ocv_error = pd.Series(
            u_ocv_error_values,
            index=u_ocv_index,
            name=u_ocv_name,
        )
    else:
        u_ocv_error = u_ocv_error_values

    metadata = {
        "ocv_error_applied": True,
        "ocv_error_type": "pseudo_discharge_ocv_negative_bias",
        "error_fraction": error_fraction,
        "ocv_full_range_v": ocv_full_range_v,
        "ocv_offset_v": ocv_offset_v,
        "ocv_min_v": ocv_min_v,
        "ocv_max_v": ocv_max_v,
        "clip": clip,
    }

    return u_ocv_error, metadata

def create_study_cases(
    data: pd.DataFrame,
    study_case: str,
    *,
    voltage_col: str = "Voltage [V]",
    current_col: str = "Current [A]",
    positive_ocv_col: str = (
        "Positive electrode bulk open-circuit potential [V]"
    ),
    negative_ocv_col: str = (
        "Negative electrode bulk open-circuit potential [V]"
    ),
    negative_stoichiometry_col: str = (
        "Average negative particle stoichiometry"
    ),
    output_ocv_col: str = "Open-circuit voltage [V]",
    output_soc_col: str = "SOC [-]",
    measurement_error_seed: Optional[int] = 123,
    current_range: Optional[float] = None,
    coverage_factor: float = 3.0,
    soc_range_reduction: float = 0.05,
    ocv_error_fraction: float = 0.02,
    ocv_min_v: float = 2.5,
    ocv_max_v: float = 4.2,
) -> pd.DataFrame:
    """
    Create a new dataframe containing the modifications associated with a
    selected study case.

    The input dataframe is copied before applying any transformation.
    Therefore, the original dataframe is not modified.

    Study cases
    -----------
    sc-0
        Original data without additional errors.
    sc-1
        Voltage and current measurement errors.
    sc-2
        SOC-range error.
    sc-3
        Pseudo-discharge OCV error.
    sc-4
        Combined measurement, SOC, and OCV errors.

    Parameters
    ----------
    data : pandas.DataFrame
        Original simulation or experimental data.
    study_case : str
        Study-case identifier: "sc-0", "sc-1", "sc-2", "sc-3", or "sc-4".
    voltage_col : str
        Name of the terminal-voltage column.
    current_col : str
        Name of the current column.
    positive_ocv_col : str
        Name of the positive-electrode bulk OCP column.
    negative_ocv_col : str
        Name of the negative-electrode bulk OCP column.
    negative_stoichiometry_col : str
        Name of the average negative-particle stoichiometry column.
    output_ocv_col : str
        Name assigned to the resulting OCV column.
    output_soc_col : str
        Name assigned to the resulting SOC column.
    measurement_error_seed : int or None
        Random seed used for the measurement-error generation.
    current_range : float or None
        Current range passed to ``apply_arbin_lbt_measurement_error``.
    coverage_factor : float
        Coverage factor used by the measurement-error model.
    soc_range_reduction : float
        Fractional reduction of the SOC operating range.
    ocv_error_fraction : float
        OCV error expressed as a fraction of the full OCV range.
    ocv_min_v : float
        Minimum voltage used to define the full OCV range.
    ocv_max_v : float
        Maximum voltage used to define the full OCV range.

    Returns
    -------
    pandas.DataFrame
        Independent dataframe containing the modified voltage, current,
        OCV, and SOC data.

        Transformation metadata are stored in:

        ``result.attrs["study_case_metadata"]``
    """

    valid_study_cases = {"sc-0", "sc-1", "sc-2", "sc-3", "sc-4"}

    if study_case not in valid_study_cases:
        raise ValueError(
            f"Invalid study case {study_case!r}. "
            f"Expected one of {sorted(valid_study_cases)}."
        )

    required_columns = {
        voltage_col,
        current_col,
        positive_ocv_col,
        negative_ocv_col,
        negative_stoichiometry_col,
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise KeyError(
            "The input dataframe is missing the following required "
            f"columns: {sorted(missing_columns)}"
        )

    # Always work with an independent dataframe.
    study_case_data = data.copy(deep=True)

    metadata = {
        "study_case": study_case,
        "measurement_error": {
            "measurement_error_applied": False,
        },
        "soc_error": {
            "soc_error_applied": False,
        },
        "ocv_error": {
            "ocv_error_applied": False,
        },
    }

    # -------------------------------------------------
    # Measurement error
    # -------------------------------------------------
    if study_case in {"sc-1", "sc-4"}:
        study_case_data, measurement_error_metadata = (
            apply_arbin_lbt_measurement_error(
                data=study_case_data,
                voltage_col=voltage_col,
                current_col=current_col,
                current_range=current_range,
                seed=measurement_error_seed,
                coverage_factor=coverage_factor,
            )
        )

        # Ensure that the returned dataframe remains independent.
        study_case_data = study_case_data.copy(deep=True)

        metadata["measurement_error"] = measurement_error_metadata

    # -------------------------------------------------
    # Open-circuit voltage
    # -------------------------------------------------
    ocv = (
        study_case_data[positive_ocv_col]
        - study_case_data[negative_ocv_col]
    )

    if study_case in {"sc-3", "sc-4"}:
        ocv, ocv_error_metadata = apply_pseudo_discharge_ocv_error(
            u_ocv=ocv,
            error_fraction=ocv_error_fraction,
            ocv_full_range_v=ocv_max_v - ocv_min_v,
            ocv_min_v=ocv_min_v,
            ocv_max_v=ocv_max_v,
            clip=False,
        )

        metadata["ocv_error"] = ocv_error_metadata

    study_case_data[output_ocv_col] = _convert_to_aligned_series(
        values=ocv,
        index=study_case_data.index,
        name=output_ocv_col,
    )

    # -------------------------------------------------
    # State of charge
    # -------------------------------------------------
    parameter_values = pybamm.ParameterValues("Chen2020")

    model = pybamm.lithium_ion.DFN(
        options={
            "particle size": "distribution",
            "thermal": "lumped",
            "surface temperature": "ambient",
            "current collector": "uniform",
            "dimensionality": 0,
        }
    )

    x_0, x_100, _, _ = pybamm.lithium_ion.get_min_max_stoichiometries(
        parameter_values,
        options=model.options,
    )

    soc = (
        study_case_data[negative_stoichiometry_col] - x_0
    ) / (x_100 - x_0)

    if study_case in {"sc-2", "sc-4"}:
        soc, soc_error_metadata = apply_soc_range_error(
            soc=soc,
            range_reduction=soc_range_reduction,
            clip=True,
        )

        metadata["soc_error"] = soc_error_metadata

    study_case_data[output_soc_col] = _convert_to_aligned_series(
        values=soc,
        index=study_case_data.index,
        name=output_soc_col,
    )

    # Store the applied settings without changing the return type.
    study_case_data.attrs["study_case"] = study_case
    study_case_data.attrs["study_case_metadata"] = metadata

    return study_case_data


def _convert_to_aligned_series(
    values,
    index: pd.Index,
    name: str,
) -> pd.Series:
    """
    Convert an array-like object into a Series aligned with a given index.

    This allows the study-case helper functions to return either a NumPy
    array or a pandas Series.
    """

    values_array = np.asarray(values).reshape(-1)

    if len(values_array) != len(index):
        raise ValueError(
            f"The generated {name!r} array contains {len(values_array)} "
            f"values, but the dataframe contains {len(index)} rows."
        )

    return pd.Series(
        data=values_array,
        index=index,
        name=name,
        dtype=float,
    )