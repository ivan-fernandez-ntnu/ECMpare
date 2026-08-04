import pybamm


def generate_battery_model(model_type: str, param_set: str, options: dict):
    """
    Create a PyBaMM battery model + ParameterValues, supporting:
      - MP-DFN: DFN + particle-size distribution (PSD) + lumped thermal
      - MSMR: MSMR + lumped thermal

    Parameters
    ----------
    model_type
        "MP-DFN" or "MSMR"
    param_set
        ParameterValues name. Recommended:
          - MP-DFN: "ORegan2022" (LG M50) or "Chen2020"
          - MSMR:   "MSMR_Example" (since DFN-oriented sets like ORegan2022 are not MSMR-ready)
    options
        Dict keyed by model_type, each value is a dict of PyBaMM options for that model.

    Returns
    -------
    model, params, options_dict_used
    """
    if options is None:
        options = get_default_options_for_fidelity()

    if model_type not in options:
        raise ValueError(f"Missing options for model_type='{model_type}'. Provided keys: {list(options.keys())}")

    options_dict = dict(options[model_type])  # copy

    if model_type == "MP-DFN":
        # Enforce PSD + thermal fidelity bundle
        # options_dict.setdefault("particle size", "distribution")
        # options_dict.setdefault("thermal", "lumped")
        # options_dict.setdefault("surface temperature", "lumped")
        # options_dict.setdefault("current collector", "uniform")
        # options_dict.setdefault("dimensionality", 0)

        model = pybamm.lithium_ion.DFN(
            options=options_dict,
            name=f"MP-DFN ({param_set} base, PSD + lumped thermal)",
        )

        params = pybamm.ParameterValues(param_set)

        # Add PSD parameters (default lognormal assumption)
        params = pybamm.get_size_distribution_parameters(params)

        # Sanity check for PSD keys
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
        # MSMR cannot use DFN parameter sets directly; prefer MSMR_Example.
        # We keep your API (param_set) but default to MSMR_Example if a DFN set is passed.
        msmr_reactions = options_dict.pop("number of MSMR reactions", ("6", "4"))

        # MSMR supports the same "thermal"/"surface temperature"/CC options via BaseBatteryModel
        model = pybamm.lithium_ion.MSMR(
            {"number of MSMR reactions": msmr_reactions},
            options=options_dict,
            name=f"MSMR ({msmr_reactions}, {param_set})",
        )

        params = None
        # try:
        #     params = pybamm.ParameterValues(param_set)
        # except Exception:
        #     # Fallback to the built-in MSMR-compatible set if the requested one isn't compatible
        #     params = pybamm.ParameterValues("MSMR_Example")

    elif model_type == "DFN":

        model = pybamm.lithium_ion.DFN(
            options=options_dict,
            name=f"Coupled-degradation DFN ({param_set} + lumped thermal)",
        )

        params = pybamm.ParameterValues(param_set)
        
    else:
        raise ValueError(f"Unsupported model_type='{model_type}'. Expected 'MP-DFN' or 'MSMR'.")

    return model, params


def get_default_options_for_fidelity():
    """
    Default, high-fidelity option bundles discussed in this conversation.

    - MP-DFN: DFN + particle-size distribution + thermal (lumped) + surface temperature (lumped),
             uniform current collector, 0D CC (robust + compatible with PSD).
    - MSMR:   MSMR with thermal (lumped) + surface temperature (lumped) + uniform CC.
             (MSMR does not use PSD.)

    Notes:
    - "x-full" thermal is not compatible with particle-size distribution models in PyBaMM.
    - Cylindrical geometry is primarily handled through geometry/parameter values
      (cell radius/height/area/volume) rather than a single "cylindrical" option switch.
    """
    return {
        "MP-DFN": {
            "particle size": "distribution",
            "thermal": "lumped",
            "surface temperature": "lumped",
            "current collector": "uniform",
            "dimensionality": 0,
        },
        "MSMR": {
            "number of MSMR reactions": ("6", "4"),
            "thermal": "lumped",
            "surface temperature": "lumped",
            "current collector": "uniform",
            "dimensionality": 0,
        },
    }