import pybamm

key = "Negative area-weighted particle-size distribution [m-1]"

# bundled parameter set names vary by version; this tries the registry if available
try:
    names = list(pybamm.parameter_sets.keys())
except Exception:
    names = ["Chen2020", "Marquis2019", "Xu2019", "ORegan2022"]  # add your candidates

hits = []
augment_ok = []
for name in names:
    try:
        p = pybamm.ParameterValues(name)
        if key in p:
            hits.append(name)
        p2 = pybamm.get_size_distribution_parameters(p)
        if key in p2:
            augment_ok.append(name)
    except Exception:
        pass

print("Already contains PSD key:", hits)
print("Can be augmented via get_size_distribution_parameters:", augment_ok)
