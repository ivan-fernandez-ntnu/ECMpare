import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
import scipy as sp

from src.utils.registry import Registry
from typing import Callable, Dict, Generic, TypeVar, List, Any, Union, Optional, Sequence


def method_to_model(estimated_parameter, method_input, method_group, method_type, model_type, ocv_curve_df):
    
    model_input = model_input_dict_generator()

    if method_group == "LSB":

        parameter_adapter_fn = from_LSB_method_to_model_parameter.get(method_type+"_method_to_model")

        model_input[model_type] = parameter_adapter_fn(estimated_parameter, model_type, method_input[method_group][method_type], ocv_curve_df)

    
    return model_input


def model_input_dict_generator():
    """
    This function generate the standardised dictionary with the output required for each method. This dictionary will often be overwritten, but it is useful to have the structure declared here.
    """
    model_input = {
        "thevenin_1rc":{
            "r_0": "float()",
            "r_1": "float()",
            "c_1": "float()",
            "u_ocv": "float()"
        }
    }

    return model_input


# region - callables and interpolator generator functions
def generate_constant_callable(constant, n_inputs):
    """
    This function generate a callable with a constant return with a certain number of inputs (that do not affect, only for consistence)
    
    :param constant: Constant parameter value that wants to be returned
    :param n_inputs: Number of inputs. An int from 0 to 5
    """

    if n_inputs == 0:
        def callable_parameter():
            return constant
    elif n_inputs == 1:
        def callable_parameter(input_1):
            return constant
    elif n_inputs == 2:
        def callable_parameter(input_1, input_2):
            return constant
    elif n_inputs == 3:
        def callable_parameter(input_1, input_2, input_3):
            return constant
    elif n_inputs == 4:
        def callable_parameter(input_1, input_2, input_3, input_4):
            return constant
    elif n_inputs == 5:
        def callable_parameter(input_1, input_2, input_3, input_4, input_5):
            return constant

    return callable_parameter

def generate_dependent_interpolant(
    df: pd.DataFrame,
    target_column: str,
    input_columns: list[str],
    children: list[pybamm.Symbol],
    *,
    interpolator: str = "linear",
    extrapolate: bool = True,
    name: str | None = None,
    agg: str = "mean",
    expected_n_points: int | None = None,
    strict_expected_points: bool = False,
    return_diagnostic: bool = False,
):
    """
    Build a pybamm.Interpolant from a DataFrame lookup table.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain input_columns + [target_column].
    target_column : str
        Dependent variable column, for example "OCV [V]".
    input_columns : list[str]
        Independent variable columns, for example ["SOC [pu]"].
        ORDER MATTERS and must match `children`.
    children : list[pybamm.Symbol]
        The model symbols to feed to the interpolant.
    interpolator : {"linear", "cubic", "linear"}
        Interpolator type. For 1D OCV, "linear" is usually a good choice.
    extrapolate : bool
        Whether to extrapolate outside the provided range.
    name : str | None
        Optional name for the interpolant.
    agg : {"mean", "last", "error"}
        How to handle duplicate input grid points.
        - "mean": average duplicates
        - "last": keep last duplicate
        - "error": raise an error if duplicates exist
    expected_n_points : int | None
        Expected number of support points in the 1D case.
    strict_expected_points : bool
        If True, raise an error when the number of final support points
        does not match expected_n_points.
    return_diagnostic : bool
        If True, return (interpolant, diagnostic_dict).

    Returns
    -------
    pybamm.Interpolant
        Or (pybamm.Interpolant, diagnostic_dict) if return_diagnostic=True.
    """

    # -------------------------------------------------
    # Basic validation
    # -------------------------------------------------
    if target_column not in df.columns:
        raise KeyError(f"target_column '{target_column}' not in df.columns")

    for c in input_columns:
        if c not in df.columns:
            raise KeyError(f"input column '{c}' not in df.columns")

    if len(input_columns) != len(children):
        raise ValueError(
            f"len(input_columns)={len(input_columns)} must match "
            f"len(children)={len(children)}"
        )

    ndim = len(input_columns)
    if ndim < 1 or ndim > 3:
        raise NotImplementedError("Only 1D/2D/3D interpolants are supported.")

    # -------------------------------------------------
    # Diagnostic container
    # -------------------------------------------------
    diagnostic: dict[str, Any] = {
        "name": name,
        "target_column": target_column,
        "input_columns": input_columns,
        "ndim": ndim,
        "interpolator": interpolator,
        "extrapolate": extrapolate,
        "agg": agg,
        "n_rows_original": len(df),
        "n_rows_after_dropna": None,
        "n_rows_after_agg": None,
        "n_duplicates_before_agg": None,
        "n_support_points": None,
        "axis_lengths": None,
        "expected_n_points": expected_n_points,
        "warnings": [],
    }

    # -------------------------------------------------
    # Keep only relevant columns and drop NaNs
    # -------------------------------------------------
    work = df[input_columns + [target_column]].copy()
    work = work.dropna()
    diagnostic["n_rows_after_dropna"] = len(work)

    # -------------------------------------------------
    # Count duplicates before aggregation
    # -------------------------------------------------
    duplicate_mask = work.duplicated(subset=input_columns, keep=False)
    diagnostic["n_duplicates_before_agg"] = int(duplicate_mask.sum())

    if agg == "error":
        if duplicate_mask.any():
            duplicate_examples = (
                work.loc[duplicate_mask, input_columns + [target_column]]
                .sort_values(input_columns)
                .head(10)
            )
            raise ValueError(
                "Duplicate input grid points detected and agg='error'.\n"
                f"Input columns: {input_columns}\n"
                f"Number of duplicate rows: {int(duplicate_mask.sum())}\n"
                f"Examples:\n{duplicate_examples}"
            )
    elif agg == "mean":
        work = work.groupby(input_columns, as_index=False)[target_column].mean()
    elif agg == "last":
        work = work.sort_values(input_columns).drop_duplicates(input_columns, keep="last")
    else:
        raise ValueError("agg must be one of {'mean', 'last', 'error'}")

    diagnostic["n_rows_after_agg"] = len(work)

    # -------------------------------------------------
    # Build axes
    # -------------------------------------------------
    axes = [np.sort(work[c].astype(float).unique()) for c in input_columns]
    diagnostic["axis_lengths"] = [len(a) for a in axes]

    # -------------------------------------------------
    # 1D case
    # -------------------------------------------------
    if ndim == 1:
        axis = axes[0]
        series = (
            work.set_index(input_columns[0])[target_column]
            .astype(float)
            .reindex(axis)
        )

        y = series.to_numpy(dtype=float)
        x = axis
        child = children[0]

        diagnostic["n_support_points"] = len(x)
        diagnostic["x"] = x.copy()
        diagnostic["y"] = y.copy()

        if expected_n_points is not None and len(x) != expected_n_points:
            msg = (
                f"Expected {expected_n_points} support points, but got {len(x)} "
                f"after preprocessing."
            )
            diagnostic["warnings"].append(msg)
            if strict_expected_points:
                raise ValueError(msg)

        interpolant = pybamm.Interpolant(
            x,
            y,
            child,
            name=name,
            interpolator=interpolator,
            extrapolate=extrapolate,
        )

        if return_diagnostic:
            return interpolant, diagnostic
        return interpolant

    # -------------------------------------------------
    # 2D/3D full-grid validation
    # -------------------------------------------------
    expected = int(np.prod([len(a) for a in axes]))
    if len(work) != expected:
        full_index = pd.MultiIndex.from_product(axes, names=input_columns)
        got_index = pd.MultiIndex.from_frame(work[input_columns].astype(float))
        missing = full_index.difference(got_index)

        raise ValueError(
            "The DataFrame does not form a full regular grid required for "
            "2D/3D pybamm.Interpolant.\n"
            f"Expected points: {expected}, got: {len(work)}.\n"
            f"Missing points example (up to 10): {list(missing[:10])}"
        )

    full_index = pd.MultiIndex.from_product(axes, names=input_columns)
    series = (
        work.set_index(input_columns)[target_column]
        .astype(float)
        .reindex(full_index)
    )

    diagnostic["n_support_points"] = len(series)

    if ndim == 2:
        y = series.to_numpy(dtype=float).reshape(len(axes[0]), len(axes[1]))
        x = (axes[0], axes[1])

        interpolant = pybamm.Interpolant(
            x,
            y,
            children,
            name=name,
            interpolator=interpolator,
            extrapolate=extrapolate,
        )

        if return_diagnostic:
            diagnostic["x"] = x
            diagnostic["y_shape"] = y.shape
            return interpolant, diagnostic
        return interpolant

    # ndim == 3
    y = series.to_numpy(dtype=float).reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    x = (axes[0], axes[1], axes[2])

    interpolant = pybamm.Interpolant(
        x,
        y,
        children,
        name=name,
        interpolator=interpolator,
        extrapolate=extrapolate,
    )

    if return_diagnostic:
        diagnostic["x"] = x
        diagnostic["y_shape"] = y.shape
        return interpolant, diagnostic
    return interpolant

def get_function_parameter_children(model, param_name: str):
    fp = next(
        p for p in model.parameters
        if isinstance(p, pybamm.FunctionParameter) and p.name == param_name
    )
    return list(fp.children)

def build_model_param_interpolator(
    model_param_dict,
    param_name,
    *,
    use_nearest_fallback=True,
    constant_tol=1e-12,
):
    """
    Build an interpolator for one model parameter:
        (temperature, soc, current) -> parameter_value

    The returned callable always has the signature:
        f(temperature, soc, current)

    However, the interpolation is internally built only on the dimensions
    that actually vary in the available data. This avoids failures when one
    coordinate is constant in the dataset.
    """
    if not isinstance(model_param_dict, dict) or len(model_param_dict) == 0:
        raise ValueError("'model_param_dict' must be a non-empty dictionary.")

    points_list = []
    values_list = []

    for key, value_dict in model_param_dict.items():
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(
                "Each key in 'model_param_dict' must be a tuple of "
                "(temperature, current, soc)."
            )

        if not isinstance(value_dict, dict):
            raise ValueError(
                "Each value in 'model_param_dict' must be a dictionary "
                "of model parameters."
            )

        if param_name not in value_dict:
            continue

        param_value = float(value_dict[param_name])

        if not np.isfinite(param_value):
            continue

        temperature = float(key[0])
        soc = float(key[1])
        current = float(key[2])

        points_list.append([temperature, current, soc])
        values_list.append(param_value)

    if len(points_list) == 0:
        raise ValueError(
            f"No valid data found for parameter '{param_name}'."
        )

    points = np.asarray(points_list, dtype=float)
    values = np.asarray(values_list, dtype=float)

    coord_names = np.array(["temperature", "soc", "current"], dtype=object)

    # Detect which dimensions actually vary
    span = np.ptp(points, axis=0)
    active_mask = span > constant_tol
    active_idx = np.where(active_mask)[0]
    inactive_idx = np.where(~active_mask)[0]

    inactive_values = {
        coord_names[i]: float(points[0, i]) for i in inactive_idx
    }

    if len(active_idx) == 0:
        # All coordinates are constant -> return a constant function
        constant_value = float(np.mean(values))

        def interpolator(temperature, current, soc):
            return constant_value

    elif len(active_idx) == 1:
        # 1D interpolation
        reduced_points = points[:, active_idx[0]]
        sort_idx = np.argsort(reduced_points)

        x = reduced_points[sort_idx]
        y = values[sort_idx]

        def interpolator(temperature, current, soc):
            query = np.array([temperature, current, soc], dtype=float)
            xq = float(query[active_idx[0]])

            value = np.interp(xq, x, y, left=np.nan, right=np.nan)

            if use_nearest_fallback and not np.isfinite(value):
                nearest_idx = np.argmin(np.abs(x - xq))
                value = y[nearest_idx]

            return float(value)

    else:
        # 2D or 3D scattered interpolation
        reduced_points = points[:, active_idx]

        linear_interp = None
        nearest_interp = None

        try:
            linear_interp = sp.interpolate.LinearNDInterpolator(
                reduced_points,
                values,
                fill_value=np.nan,
            )
        except Exception:
            if not use_nearest_fallback:
                raise RuntimeError(
                    f"Could not build LinearNDInterpolator for '{param_name}'."
                )

        if use_nearest_fallback:
            nearest_interp = sp.interpolate.NearestNDInterpolator(
                reduced_points,
                values,
            )

        def interpolator(temperature, current, soc):
            query_full = np.array([temperature, current, soc], dtype=float)
            query_reduced = query_full[active_idx]

            value = np.nan

            if linear_interp is not None:
                value = linear_interp(query_reduced.reshape(1, -1))[0]

            if use_nearest_fallback and not np.isfinite(value):
                value = nearest_interp(query_reduced.reshape(1, -1))[0]

            return float(value)

    # Attach some useful metadata
    interpolator.param_name = param_name
    interpolator.active_dimensions = coord_names[active_idx].tolist()
    interpolator.inactive_dimensions = inactive_values
    interpolator.n_points = len(values)

    return interpolator

def _extend_axis_to_bounds(axis_values, bounds, tol=1e-12):
    """
    Extend an axis only outside its current range.

    Example
    -------
    axis_values = [0.1, 0.2, 0.4]
    bounds = (0.0, 1.0)

    -> [0.0, 0.1, 0.2, 0.4, 1.0]
    """
    axis_values = np.unique(np.asarray(axis_values, dtype=float))

    if bounds is None:
        return axis_values

    lower, upper = bounds
    extra = []

    if lower is not None and lower < np.min(axis_values) - tol:
        extra.append(float(lower))

    if upper is not None and upper > np.max(axis_values) + tol:
        extra.append(float(upper))

    if extra:
        axis_values = np.unique(np.concatenate([axis_values, np.asarray(extra, dtype=float)]))

    return axis_values

def _clip_symbol_to_bounds(symbol, lower=None, upper=None):
    """
    Clip a PyBaMM symbolic variable to [lower, upper].
    """
    out = symbol

    if lower is not None:
        out = pybamm.maximum(out, pybamm.Scalar(float(lower)))

    if upper is not None:
        out = pybamm.minimum(out, pybamm.Scalar(float(upper)))

    return out

def build_pybamm_param_interpolator_from_scattered(
    model_param_dict,
    param_name,
    *,
    input_order=("temperature", "current", "soc"),
    grid_axes=None,
    extend_bounds=None,
    use_nearest_fallback=True,
    constant_tol=1e-12,
    interpolator="linear",
    extrapolate=False,
    clamp_inputs_to_grid=True,
):
    """
    Build a PyBaMM-compatible interpolator for one model parameter from
    scattered data.

    Parameters
    ----------
    model_param_dict : dict
        Keys are coordinate tuples.
        Values are dicts with model parameters.

    param_name : str
        Parameter name to interpolate, e.g. "r_0", "r_1", "c_1".

    input_order : tuple[str, str, str]
        Order of coordinates in the dictionary keys.

    grid_axes : dict or None
        Optional target grid for the active dimensions.
        Example:
            {
                "soc": np.array([...]),
                "current": np.array([...]),
            }

        If None, the unique coordinate values from the raw data are used.

    extend_bounds : dict or None
        Optional extension bounds for each coordinate.
        Example:
            {
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            }

        The axis is only extended OUTSIDE the available raw range.
        Values in the extended region are obtained by clipping to the raw
        boundary, which creates a constant edge extension.

    use_nearest_fallback : bool
        If True, use nearest-neighbor fallback where linear interpolation
        returns NaN.

    constant_tol : float
        Tolerance to detect constant dimensions.

    interpolator : str
        Interpolation mode for pybamm.Interpolant.

    extrapolate : bool
        Extrapolation flag for pybamm.Interpolant.

    clamp_inputs_to_grid : bool
        If True, the symbolic PyBaMM inputs are clipped to the final grid
        bounds before evaluating the interpolant. This avoids extrapolation
        warnings when the model goes slightly outside the grid.

    Returns
    -------
    param_fun : callable
        Function with signature:
            param_fun(temperature, current, soc)
        suitable for PyBaMM parameter values.
    """
    valid_names = {"temperature", "current", "soc"}
    if set(input_order) != valid_names:
        raise ValueError(
            "'input_order' must be a permutation of "
            "('temperature', 'current', 'soc')."
        )

    if extend_bounds is None:
        extend_bounds = {}

    coord_pos = {name: i for i, name in enumerate(input_order)}

    points = []
    values = []

    for key, value_dict in model_param_dict.items():
        if param_name not in value_dict:
            continue

        param_value = float(value_dict[param_name])
        if not np.isfinite(param_value):
            continue

        temperature = float(key[coord_pos["temperature"]])
        current = float(key[coord_pos["current"]])
        soc = float(key[coord_pos["soc"]])

        # Internal fixed order: [temperature, soc, current]
        points.append([temperature, soc, current])
        values.append(param_value)

    if not points:
        raise ValueError(f"No valid data found for parameter '{param_name}'.")

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)

    coord_names = np.array(["temperature", "soc", "current"], dtype=object)

    span = np.ptp(points, axis=0)
    active_idx = np.where(span > constant_tol)[0]
    active_names = coord_names[active_idx].tolist()

    # All coordinates constant
    if len(active_idx) == 0:
        constant_value = float(np.mean(values))

        def param_fun(temperature, current, soc):
            return pybamm.Scalar(constant_value)

        param_fun.param_name = param_name
        param_fun.active_dimensions = []
        param_fun.grid_axes = {}
        param_fun.raw_bounds = {}
        return param_fun

    reduced_points = points[:, active_idx]

    # Raw bounds in active dimensions
    raw_bounds = {}
    for local_i, dim_name in enumerate(active_names):
        raw_bounds[dim_name] = (
            float(np.min(reduced_points[:, local_i])),
            float(np.max(reduced_points[:, local_i])),
        )

    # Build target axes
    if grid_axes is None:
        axes = []
        for local_i, dim_name in enumerate(active_names):
            axis_values = np.unique(reduced_points[:, local_i]).astype(float)
            axis_values = _extend_axis_to_bounds(
                axis_values,
                extend_bounds.get(dim_name),
                tol=constant_tol,
            )
            axes.append(axis_values)
    else:
        axes = []
        for dim_name in active_names:
            if dim_name not in grid_axes:
                raise ValueError(
                    f"Missing grid axis for active dimension '{dim_name}'."
                )

            axis_values = np.asarray(grid_axes[dim_name], dtype=float)
            axis_values = np.unique(axis_values)

            if axis_values.size == 0:
                raise ValueError(f"Grid axis for '{dim_name}' is empty.")

            axis_values = _extend_axis_to_bounds(
                axis_values,
                extend_bounds.get(dim_name),
                tol=constant_tol,
            )
            axes.append(axis_values)

    # 1D case
    if len(active_idx) == 1:
        x_data = reduced_points[:, 0]
        sort_idx = np.argsort(x_data)

        x_data = x_data[sort_idx]
        y_data = values[sort_idx]

        raw_min = float(np.min(x_data))
        raw_max = float(np.max(x_data))

        x_grid = axes[0]

        # Clip query positions to the raw data range
        x_eval = np.clip(x_grid, raw_min, raw_max)

        # This creates constant extension outside [raw_min, raw_max]
        y_grid = np.interp(x_eval, x_data, y_data)

        if use_nearest_fallback:
            nan_mask = ~np.isfinite(y_grid)
            if np.any(nan_mask):
                for i in np.where(nan_mask)[0]:
                    nearest_idx = np.argmin(np.abs(x_data - x_eval[i]))
                    y_grid[i] = y_data[nearest_idx]

        if np.any(~np.isfinite(y_grid)):
            raise ValueError(
                f"Could not build a valid regular grid for parameter '{param_name}'."
            )

        def param_fun(temperature, current, soc):
            children_map = {
                "temperature": temperature,
                "current": current,
                "soc": soc,
            }

            child = children_map[active_names[0]]

            if clamp_inputs_to_grid:
                child = _clip_symbol_to_bounds(
                    child,
                    lower=float(np.min(x_grid)),
                    upper=float(np.max(x_grid)),
                )

            return pybamm.Interpolant(
                x_grid,
                y_grid,
                child,
                interpolator=interpolator,
                extrapolate=extrapolate,
            )

        param_fun.param_name = param_name
        param_fun.active_dimensions = active_names
        param_fun.grid_axes = {active_names[0]: x_grid}
        param_fun.raw_bounds = raw_bounds

        return param_fun

    # 2D / 3D case
    linear_interp = sp.interpolate.LinearNDInterpolator(
        reduced_points,
        values,
        fill_value=np.nan,
    )

    nearest_interp = None
    if use_nearest_fallback:
        nearest_interp = sp.interpolate.NearestNDInterpolator(
            reduced_points,
            values,
        )

    mesh = np.meshgrid(*axes, indexing="ij")
    query_points = np.column_stack([m.ravel() for m in mesh])

    # Clip each active coordinate independently to the raw data bounds
    # before evaluating the scattered interpolator.
    query_points_eval = query_points.copy()
    for local_i, dim_name in enumerate(active_names):
        raw_min, raw_max = raw_bounds[dim_name]
        query_points_eval[:, local_i] = np.clip(
            query_points_eval[:, local_i],
            raw_min,
            raw_max,
        )

    y_flat = linear_interp(query_points_eval)

    if use_nearest_fallback:
        nan_mask = ~np.isfinite(y_flat)
        if np.any(nan_mask):
            y_flat[nan_mask] = nearest_interp(query_points_eval[nan_mask])

    if np.any(~np.isfinite(y_flat)):
        raise ValueError(
            f"Could not fill the target regular grid for parameter '{param_name}'."
        )

    y_grid = y_flat.reshape(tuple(len(ax) for ax in axes))

    def param_fun(temperature, current, soc):
        children_map = {
            "temperature": temperature,
            "current": current,
            "soc": soc,
        }

        children = []
        for local_i, dim_name in enumerate(active_names):
            child = children_map[dim_name]

            if clamp_inputs_to_grid:
                child = _clip_symbol_to_bounds(
                    child,
                    lower=float(np.min(axes[local_i])),
                    upper=float(np.max(axes[local_i])),
                )

            children.append(child)

        children = tuple(children)

        return pybamm.Interpolant(
            tuple(axes),
            y_grid,
            children,
            interpolator=interpolator,
            extrapolate=extrapolate,
        )

    param_fun.param_name = param_name
    param_fun.active_dimensions = active_names
    param_fun.grid_axes = {
        active_names[i]: axes[i] for i in range(len(active_names))
    }
    param_fun.raw_bounds = raw_bounds

    return param_fun

def build_pybamm_param_interpolator(
    model_param_dict,
    param_name,
    *,
    input_order=("temperature", "soc", "current"),
    interpolator="linear",
    extrapolate=False,
    constant_tol=1e-12,
):
    """
    Build a PyBaMM-compatible parameter function for one model parameter.

    Parameters
    ----------
    model_param_dict : dict
        Keys: (temperature, soc, current)
        Values: dict with entries like {"r_0": ..., "r_1": ..., "c_1": ...}

    param_name : str
        Parameter to interpolate, e.g. "r_0", "r_1", "c_1".

    input_order : tuple[str, str, str]
        Logical order of the coordinates stored in model_param_dict keys.
        For your dictionary this should normally be:
            ("temperature", "soc", "current")

    interpolator : str
        Passed to pybamm.Interpolant.

    extrapolate : bool
        Passed to pybamm.Interpolant.

    constant_tol : float
        Tolerance to detect constant coordinates.

    Returns
    -------
    callable
        A function that can be passed into PyBaMM parameter values.
        It accepts symbolic inputs and returns either:
        - pybamm.Scalar
        - pybamm.Interpolant
    """
    if not model_param_dict:
        raise ValueError("'model_param_dict' is empty.")

    valid_names = {"temperature", "soc", "current"}
    if set(input_order) != valid_names:
        raise ValueError(
            "'input_order' must be a permutation of "
            "('temperature', 'soc', 'current')."
        )

    coord_pos = {name: i for i, name in enumerate(input_order)}

    points = []
    values = []

    for key, value_dict in model_param_dict.items():
        if param_name not in value_dict:
            continue

        temperature = float(key[coord_pos["temperature"]])
        soc = float(key[coord_pos["soc"]])
        current = float(key[coord_pos["current"]])
        param_value = float(value_dict[param_name])

        if not np.isfinite(param_value):
            continue

        points.append([temperature, soc, current])
        values.append(param_value)

    if not points:
        raise ValueError(f"No valid values found for parameter '{param_name}'.")

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)

    coord_names = np.array(["temperature", "soc", "current"], dtype=object)

    # Detect active dimensions
    span = np.ptp(points, axis=0)
    active_mask = span > constant_tol
    active_idx = np.where(active_mask)[0]

    # All constant -> return scalar
    if len(active_idx) == 0:
        constant_value = float(np.mean(values))

        def param_fun(temperature, current, soc):
            return pybamm.Scalar(constant_value)

        return param_fun

    # Build unique coordinate arrays for active dimensions
    unique_coords = [np.unique(points[:, i]) for i in active_idx]

    # Check full regular grid
    expected_size = int(np.prod([len(v) for v in unique_coords]))
    if expected_size != len(values):
        raise ValueError(
            f"Parameter '{param_name}' is not defined on a full regular grid "
            f"for the active dimensions {coord_names[active_idx].tolist()}. "
            f"PyBaMM Interpolant needs regular-grid data."
        )

    # Map point -> value
    value_map = {
        tuple(row): val for row, val in zip(points, values)
    }

    # 1D grid
    if len(active_idx) == 1:
        idx = active_idx[0]
        x = unique_coords[0]
        y = np.empty(len(x), dtype=float)

        for i, xv in enumerate(x):
            point = [points[0, 0], points[0, 1], points[0, 2]]
            point[idx] = xv
            y[i] = value_map[tuple(point)]

        def param_fun(temperature, current, soc):
            children_map = {
                "temperature": temperature,
                "current": current,
                "soc": soc,
            }
            child = children_map[coord_names[idx]]
            return pybamm.Interpolant(
                x,
                y,
                child,
                interpolator=interpolator,
                extrapolate=extrapolate,
            )

        return param_fun

    # 2D grid
    elif len(active_idx) == 2:
        idx0, idx1 = active_idx
        x0 = unique_coords[0]
        x1 = unique_coords[1]
        y = np.empty((len(x0), len(x1)), dtype=float)

        base_point = [points[0, 0], points[0, 1], points[0, 2]]

        for i, v0 in enumerate(x0):
            for j, v1 in enumerate(x1):
                point = base_point.copy()
                point[idx0] = v0
                point[idx1] = v1
                y[i, j] = value_map[tuple(point)]

        def param_fun(temperature, current, soc):
            children_map = {
                "temperature": temperature,
                "current": current,
                "soc": soc,
            }
            child0 = children_map[coord_names[idx0]]
            child1 = children_map[coord_names[idx1]]

            return pybamm.Interpolant(
                (x0, x1),
                y,
                (child0, child1),
                interpolator=interpolator,
                extrapolate=extrapolate,
            )

        return param_fun

    # 3D grid
    elif len(active_idx) == 3:
        xT = np.unique(points[:, 0])
        xSOC = np.unique(points[:, 1])
        xI = np.unique(points[:, 2])

        y = np.empty((len(xT), len(xSOC), len(xI)), dtype=float)

        for i, T in enumerate(xT):
            for j, SOC in enumerate(xSOC):
                for k, I in enumerate(xI):
                    y[i, j, k] = value_map[(T, SOC, I)]

        def param_fun(temperature, current, soc):
            # Important:
            # keys in value_map are (temperature, soc, current)
            # so axes are (T, SOC, I)
            return pybamm.Interpolant(
                (xT, xSOC, xI),
                y,
                (temperature, soc, current),
                interpolator=interpolator,
                extrapolate=extrapolate,
            )

        return param_fun

    else:
        raise RuntimeError("Unexpected number of active dimensions.")
# endregion

from_LSB_method_to_model_parameter = Registry[Callable[..., Any]]("from_LSB_method_to_model_parameter")

@from_LSB_method_to_model_parameter.register("LS_method_to_model_backup")
def LS_method_to_model_backup(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":
        
        T_s = method_input["T_s"]
        r_0 = theta_array[0]
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 

        model_options = {
            "number of rc elements": 0,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":
        
        T_s = method_input["T_s"]
        tau_1 = - (T_s + T_s*theta_array[0]) / (2*(theta_array[0] - 1))
        r_0 = (theta_array[1] - theta_array[2]) / (theta_array[0] + 1)
        r_1 = - (2*theta_array[2] + 2*theta_array[0]*theta_array[1]) / (theta_array[0]*theta_array[0] - 1)
        c_1 = tau_1 / r_1
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 
        model_param_dict["r_1"] = pybamm.Scalar(r_1) 
        model_param_dict["c_1"] = pybamm.Scalar(c_1) 

        model_options = {
            "number of rc elements": 1,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":
        
        T_s = method_input["T_s"]

        a = - (T_s*T_s*(theta_array[0] - theta_array[1] + 1)) / (4* (theta_array[0] + theta_array[1] - 1))
        b = - (T_s*(theta_array[1] + 1)) / (theta_array[0] + theta_array[1] - 1)
        c = - (theta_array[3] + theta_array[2] + theta_array[4]) / (theta_array[0] + theta_array[1] - 1)
        d = - (T_s*(theta_array[2] - theta_array[4])) / (theta_array[0] + theta_array[1] - 1) 
        r_0 = (theta_array[2] - theta_array[3] + theta_array[4]) / (theta_array[0] - theta_array[1] + 1)

        tau_1 = np.min([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
        tau_2 = np.max([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
        r_2 = (r_0 * (tau_1 + tau_2) - d + tau_2*(c - r_0)) / (tau_2 - tau_1)
        r_1 = c - r_2 - r_0

        c_1 = tau_1 / r_1
        c_2 = tau_2 / r_2
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 
        model_param_dict["r_1"] = pybamm.Scalar(r_1) 
        model_param_dict["c_1"] = pybamm.Scalar(c_1) 
        model_param_dict["r_2"] = pybamm.Scalar(r_2) 
        model_param_dict["c_2"] = pybamm.Scalar(c_2) 

        model_options = {
            "number of rc elements": 2,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("LS_method_to_model")
def LS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]
            r_0 = theta_array_i[0]
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]
            tau_1 = - (T_s + T_s*theta_array_i[0]) / (2*(theta_array_i[0] - 1))
            r_0 = (theta_array_i[1] - theta_array_i[2]) / (theta_array_i[0] + 1)
            r_1 = - (2*theta_array_i[2] + 2*theta_array_i[0]*theta_array_i[1]) / (theta_array_i[0]*theta_array_i[0] - 1)
            c_1 = tau_1 / r_1
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0]: #or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]: #or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            a = - (T_s*T_s*(theta_array_i[0] - theta_array_i[1] + 1)) / (4* (theta_array_i[0] + theta_array_i[1] - 1))
            b = - (T_s*(theta_array_i[1] + 1)) / (theta_array_i[0] + theta_array_i[1] - 1)
            c = - (theta_array_i[3] + theta_array_i[2] + theta_array_i[4]) / (theta_array_i[0] + theta_array_i[1] - 1)
            d = - (T_s*(theta_array_i[2] - theta_array_i[4])) / (theta_array_i[0] + theta_array_i[1] - 1) 
            r_0 = (theta_array_i[2] - theta_array_i[3] + theta_array_i[4]) / (theta_array_i[0] - theta_array_i[1] + 1)

            tau_1 = np.min([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
            tau_2 = np.max([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
            r_2 = (r_0 * (tau_1 + tau_2) - d + tau_2*(c - r_0)) / (tau_2 - tau_1)
            r_1 = c - r_2 - r_0

            c_1 = tau_1 / r_1
            c_2 = tau_2 / r_2
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2

            if r_0 < method_input["lower_bound"][0]: #or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]: #or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3]: #or r_2 > method_input["upper_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("NLLS_method_to_model_backup")
def NLLS_method_to_model_backup(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":
        
        T_s = method_input["T_s"]

        r_0 = theta_array[0] 
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 

        model_options = {
            "number of rc elements": 0,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":
        
        T_s = method_input["T_s"]

        r_0 = theta_array[0]
        r_1 = theta_array[1]
        tau_1 = theta_array[2]
        c_1 = tau_1 / r_1      
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 
        model_param_dict["r_1"] = pybamm.Scalar(r_1) 
        model_param_dict["c_1"] = pybamm.Scalar(c_1) 

        model_options = {
            "number of rc elements": 1,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":
        
        T_s = method_input["T_s"]

        r_0 = theta_array[0]
        r_1 = theta_array[1]
        tau_1 = theta_array[2]
        r_2 = theta_array[3]
        tau_2 = theta_array[4]
        c_1 = tau_1 / r_1      
        c_2 = tau_2 / r_2      
        
        model_param_dict["r_0"] = pybamm.Scalar(r_0) 
        model_param_dict["r_1"] = pybamm.Scalar(r_1) 
        model_param_dict["c_1"] = pybamm.Scalar(c_1) 
        model_param_dict["r_2"] = pybamm.Scalar(r_2) 
        model_param_dict["c_2"] = pybamm.Scalar(c_2) 

        model_options = {
            "number of rc elements": 2,
            # "calculate discharge energy": "false",
            # "diffusion element": "false",
            # "operating mode": "current"
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("NLLS_method_to_model")
def NLLS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("TROLS_method_to_model")
def TROLS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("MWTROLS_method_to_model")
def MWTROLS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("SLSQP_method_to_model")
def SLSQP_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3] or r_2 > method_input["upper_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("MWSLSQP_method_to_model")
def MWSLSQP_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3] or r_2 > method_input["upper_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("MWLS_method_to_model")
def MWLS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]
            r_0 = theta_array_i[0]
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]
            tau_1 = - (T_s + T_s*theta_array_i[0]) / (2*(theta_array_i[0] - 1))
            r_0 = (theta_array_i[1] - theta_array_i[2]) / (theta_array_i[0] + 1)
            r_1 = - (2*theta_array_i[2] + 2*theta_array_i[0]*theta_array_i[1]) / (theta_array_i[0]*theta_array_i[0] - 1)
            c_1 = tau_1 / r_1
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            a = - (T_s*T_s*(theta_array_i[0] - theta_array_i[1] + 1)) / (4* (theta_array_i[0] + theta_array_i[1] - 1))
            b = - (T_s*(theta_array_i[1] + 1)) / (theta_array_i[0] + theta_array_i[1] - 1)
            c = - (theta_array_i[3] + theta_array_i[2] + theta_array_i[4]) / (theta_array_i[0] + theta_array_i[1] - 1)
            d = - (T_s*(theta_array_i[2] - theta_array_i[4])) / (theta_array_i[0] + theta_array_i[1] - 1) 
            r_0 = (theta_array_i[2] - theta_array_i[3] + theta_array_i[4]) / (theta_array_i[0] - theta_array_i[1] + 1)

            tau_1 = np.min([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
            tau_2 = np.max([(b + np.sqrt(b*b - 4*a))/2 , (b - np.sqrt(b*b - 4*a))/2])
            r_2 = (r_0 * (tau_1 + tau_2) - d + tau_2*(c - r_0)) / (tau_2 - tau_1)
            r_1 = c - r_2 - r_0

            c_1 = tau_1 / r_1
            c_2 = tau_2 / r_2
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2

            if r_0 < method_input["lower_bound"][0] or r_0 > method_input["upper_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1] or r_1 > method_input["upper_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3] or r_2 > method_input["upper_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict

@from_LSB_method_to_model_parameter.register("MWNLLS_method_to_model")
def MWNLLS_method_to_model(theta_array, model_type, method_input, ocv_curve_df):

    model_param_dict_raw = {}

    for meas_state in theta_array:

        model_param_dict_i = {}
        theta_array_i = theta_array[meas_state]

        if model_type == "thevenin_0rc" or model_type == "rint":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0] 
            
            model_param_dict_i["r_0"] = r_0 
            
            if r_0 < method_input["lower_bound"][0]:
                continue

        elif model_type == "thevenin_1rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            c_1 = tau_1 / r_1      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            
            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue

        elif model_type == "thevenin_2rc":
            
            T_s = method_input["T_s"]

            r_0 = theta_array_i[0]
            r_1 = theta_array_i[1]
            tau_1 = theta_array_i[2]
            r_2 = theta_array_i[3]
            tau_2 = theta_array_i[4]
            c_1 = tau_1 / r_1      
            c_2 = tau_2 / r_2      
            
            model_param_dict_i["r_0"] = r_0 
            model_param_dict_i["r_1"] = r_1 
            model_param_dict_i["c_1"] = c_1 
            model_param_dict_i["r_2"] = r_2 
            model_param_dict_i["c_2"] = c_2 

            #TODO The limit is stabllished for tau, not for C, so it will be only restricted as positive at the moment

            if r_0 < method_input["lower_bound"][0]:
                continue
            if r_1 <= method_input["lower_bound"][1]:
                continue
            if c_1 < 0:
                continue
            if r_2 <= method_input["lower_bound"][3]:
                continue
            if c_2 < 0:
                continue

        model_param_dict_raw[meas_state] = model_param_dict_i

    model_param_dict = {}

    if model_type == "thevenin_0rc" or model_type == "rint":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 

        model_options = {
                "number of rc elements": 0,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_1rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )
        
        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 

        model_options = {
                "number of rc elements": 1,
            }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    elif model_type == "thevenin_2rc":

        r0_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_0",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        r2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="r_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c1_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_1",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        c2_interp = build_pybamm_param_interpolator_from_scattered(
            model_param_dict=model_param_dict_raw,
            param_name="c_2",
            input_order=("temperature", "soc", "current"),
            extend_bounds={
                "soc": (0.0, 1.0),
                "temperature": (0.0, 80.0),
                "current": (-50.0, 50.0),
            },
            interpolator="linear",
            extrapolate=False,
            clamp_inputs_to_grid=True,
        )

        model_param_dict["r_0"] = r0_interp 
        model_param_dict["r_1"] = r1_interp 
        model_param_dict["c_1"] = c1_interp 
        model_param_dict["r_2"] = r2_interp 
        model_param_dict["c_2"] = c2_interp 

        model_options = {
            "number of rc elements": 2,
        }

        model = pybamm.equivalent_circuit.Thevenin(options=model_options, build=True)
        children_uocv = get_function_parameter_children(model, "Open-circuit voltage [V]")

        uocv_interp, uocv_diag = generate_dependent_interpolant(
            df=ocv_curve_df,
            target_column="OCV [V]",
            input_columns=["SOC [pu]"],
            children=children_uocv,
            interpolator="linear",
            extrapolate=False,
            name="OCV [V]",
            agg="error",
            expected_n_points=501,
            strict_expected_points=True,
            return_diagnostic=True,
        )

        model_param_dict["u_ocv"] = uocv_interp 

    return model_param_dict