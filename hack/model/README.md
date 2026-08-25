# Synthetic Data Generation

This directory contains the synthetic-data generation tools used by ECMpare.

The simulations are built with [PyBaMM](https://www.pybamm.org/) and are used to generate reference battery data for the parameter-estimation comparison framework implemented under `src/`.

Detailed documentation of the models, protocols, and solvers is available in the project Wiki.

---

## Directory structure

```text
hack/model/
├── drive_cycles/       # Input files defining dynamic drive cycles
├── functions/          # Model, protocol, solver, and simulation utilities
├── output/             # Generated simulation data and parameter files
└── *.py                # Scripts used to generate synthetic datasets
```

---

## Workflow

Synthetic datasets are generated following the general workflow:

```text
Battery model
     |
     v
Experimental protocol
     |
     v
Numerical solver
     |
     v
PyBaMM simulation
     |
     v
Selected output variables
     |
     v
Synthetic dataset
```

The generated datasets are subsequently used by the comparison framework under:

```text
src/
```

where study cases are created and model parameters are estimated and evaluated.

---

## Battery models

Battery-model construction utilities are located in:

```text
hack/model/functions/model_functions.py
```

The current synthetic-data workflow supports electrochemical reference models such as:

```text
MP-DFN
MSMR
```

Model configuration and parameter sets are handled through PyBaMM.

See the Wiki page **Battery Models for synthetic data generation** for further details.

---

## Experimental protocols

Experimental and evaluation protocols are defined in:

```text
hack/model/functions/protocol_functions.py
```

The implemented protocols include examples based on:

```text
HPPC
GITT
ICI
APP
FUDS
DST
ICA
```

Drive-cycle definitions required by protocols such as FUDS and DST are stored under:

```text
hack/model/drive_cycles/
```

See the Wiki page **Experimetal protocols** for the current protocol definitions.

---

## Numerical solvers

Solver construction is handled in:

```text
hack/model/functions/solver_functions.py
```

The current implementation includes configurations for PyBaMM solvers such as:

```text
IDAKLUSolver
CasadiSolver
```

See the Wiki page **Numerical Solvers** for the solver configuration and failure-handling behavior.

---

## Generated data

Simulation outputs are written under:

```text
hack/model/output/
```

Depending on the generation script, this directory may contain:

```text
parquet/        # Synthetic time-series datasets
parameters/     # Derived parameter data such as OCV-SOC curves
figures/        # Simulation or diagnostic figures
```

Generated simulation outputs can be large and are not intended to be tracked by Git.

The required output directories should therefore exist locally even when their generated contents are excluded from the repository.

---

## Running the synthetic-data scripts

Run scripts from the repository root so that project-relative paths are resolved consistently.

For example:

```bash
python hack/model/<simulation_script>.py
```

The exact script depends on the reference model and dataset to be generated.

Simulation configuration is currently defined within the corresponding generation scripts and may include:

```text
Model type
PyBaMM parameter set
Experimental protocol
Numerical solver
Sampling period
Maximum solver time step
Initial temperature
Requested output variables
```

---

## Installation

The synthetic-data generation tools use the same Python environment as the rest of ECMpare.

From the repository root:

```bash
pip install -r requirements.txt
```

A Conda environment is recommended for reproducibility.

---

## Relationship with the comparison framework

The `hack/model/` directory is responsible for producing reference data.

The main comparison workflow is located under:

```text
src/
```

The relationship between both parts of the repository is:

```text
hack/model/
Synthetic reference-data generation
        |
        v
hack/model/output/
        |
        v
src/
Study cases
        |
        v
Parameter estimation
        |
        v
Equivalent circuit model
        |
        v
Evaluation against reference data
```

Synthetic-data generation and parameter estimation are intentionally kept separate so that the same comparison framework can later be used with other compatible data sources.

---

## Documentation

For detailed information, see the ECMpare Wiki:

- **Battery Models for synthetic data generation**
- **Experimetal protocols**
- **Numerical Solvers**
- **Framework Architecture**
- **Study Cases**
- **How to Test a Data-Method-Model Combination in ECMpare**

The root [`README.md`](../../README.md) provides the general project overview and installation instructions.