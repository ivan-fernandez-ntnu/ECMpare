# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Funciones

def select_arbin_current_range(max_abs_current: float) -> float:
    """
    Select the smallest Arbin current range that can contain the current signal.

    Parameters
    ----------
    max_abs_current : float
        Maximum absolute current in A.

    Returns
    -------
    float
        Selected current full-scale range in A.
    """
    available_ranges = np.array([0.02, 0.5, 5.0, 30.0])  # A

    valid_ranges = available_ranges[max_abs_current <= available_ranges]

    if len(valid_ranges) == 0:
        raise ValueError(
            f"Current exceeds the maximum Arbin range: "
            f"{max_abs_current:.6f} A > 30 A"
        )

    return float(valid_ranges[0])


def add_arbin_measurement_error_to_dataframe(
    data_df: pd.DataFrame,
    voltage_col: str = "Voltage [V]",
    current_col: str = "Current [A]",
    current_range: float | None = None,
    seed: int | None = 123,
    coverage_factor: float = 3.0,
    copy: bool = True,
    replace: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Add synthetic Arbin LBT21084HC measurement error to voltage and current data.

    The Arbin specification gives precision and accuracy as percentages of the
    full-scale range (FSR). Since the specification gives bounds and not standard
    deviations, the bounds are converted to Gaussian standard deviations using
    a coverage factor.

    Parameters
    ----------
    data_df : pd.DataFrame
        Input dataframe containing simulated voltage and current.

    voltage_col : str
        Name of the voltage column in V.

    current_col : str
        Name of the current column in A.

    current_range : float, optional
        Active current range in A. Valid values are 30.0, 5.0, 0.5, and 0.02.
        If None, the smallest range that contains the full current signal is used.

    seed : int, optional
        Random seed for reproducibility.

    coverage_factor : float
        Factor used to convert specification bounds into standard deviations.
        A value of 3.0 means that the quoted +/- specification is treated as
        approximately a 3-sigma interval.

    copy : bool
        If True, a copy of the dataframe is returned.

    replace : bool
        If True, the original voltage and current columns are overwritten.
        If False, new columns are created.

    Returns
    -------
    pd.DataFrame
        Dataframe with measurement error added.

    dict
        Dictionary containing the error model metadata.
    """
    if voltage_col not in data_df.columns:
        raise KeyError(f"Voltage column not found: {voltage_col}")

    if current_col not in data_df.columns:
        raise KeyError(f"Current column not found: {current_col}")

    rng = np.random.default_rng(seed)

    output_df = data_df.copy() if copy else data_df

    voltage_true = output_df[voltage_col].to_numpy(dtype=float)
    current_true = output_df[current_col].to_numpy(dtype=float)

    # -------------------------------------------------
    # Arbin LBT21084HC measurement specifications
    # -------------------------------------------------
    voltage_fsr = 5.0  # V, from 0 to 5 V range

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
    # Separate accuracy into systematic bias and random precision noise
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

    voltage_measured = voltage_true + voltage_bias + voltage_noise
    current_measured = current_true + current_bias + current_noise

    # -------------------------------------------------
    # Store results
    # -------------------------------------------------
    if replace:
        output_df[voltage_col] = voltage_measured
        output_df[current_col] = current_measured
    else:
        output_df["Voltage_measured_error_case_1 [V]"] = voltage_measured
        output_df["Current_measured_error_case_1 [A]"] = current_measured

        output_df["Voltage_measurement_error_case_1 [V]"] = (
            voltage_measured - voltage_true
        )
        output_df["Current_measurement_error_case_1 [A]"] = (
            current_measured - current_true
        )

    metadata = {
        "study_case": "case_1_measurement_error",
        "voltage_column": voltage_col,
        "current_column": current_col,
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

    return output_df, metadata



# Definir ejecucion

data_model_type  = "MP-DFN"
param_set = "Chen2020"
solver_name = "IDAKLUSolver"
test_protocol = "APP_short_rest_100soc"
initial_cell_temp_k = 298.15 + 0
max_time_step = 0.01

initial_cell_temp_k_text = str(initial_cell_temp_k).replace(".","p")
max_time_step_text = str(max_time_step).replace(".","p") 

# Cargar archivos 

# selected_data_file = f"hack\\model\\output\\csv\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.csv"
selected_data_file = f"hack\\model\\output\\parquet\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}.parquet"
selected_data_path= os.path.join(os.getcwd(),selected_data_file)
# data_t = pd.read_csv(selected_data_path)
data_t = pd.read_parquet(selected_data_path)

# Study case 0 : Original 

data_t.to_parquet(f"hack\\model\\output\\parquet\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_{"sc-0"}.parquet", compression="gzip")


# Study case 1 : Error in measurement value

data_t_case_1, error_metadata_case_1 = add_arbin_measurement_error_to_dataframe(
    data_df=data_t,
    voltage_col="Voltage [V]",
    current_col="Current [A]",
    current_range=None,      # Automatically selects the smallest valid Arbin range
    seed=123,                # Reproducible noise
    coverage_factor=3.0,     # Treat +/- specification as approximately 3 sigma
    copy=True,
    replace=True,            # Overwrite Voltage [V] and Current [A]
)

# Check that the dataframe structure has not changed
assert data_t_case_1.shape[1] == data_t.shape[1]
assert list(data_t_case_1.columns) == list(data_t.columns)

data_t_case_1.to_parquet(f"hack\\model\\output\\parquet\\simulation_{data_model_type}_{param_set}_{solver_name}_{test_protocol}_T{initial_cell_temp_k_text}_{max_time_step_text}_{"sc-1"}.parquet", compression="gzip")

# Study case 2 : Error in SOC value


# Study case 3 : Error in OCV value


# Study case 4 : Combined error





