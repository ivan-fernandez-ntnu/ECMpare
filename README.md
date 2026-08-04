# ECMpare - Model Parameter Estimation Comparison Framework

Research framework for generating synthetic battery data and comparing parameter-estimation methods across different models, input protocols, and study cases.

The project currently focuses on equivalent circuit models (ECMs), with synthetic reference data generated using electrochemical battery models in PyBaMM. It is intended to support reproducible comparisons of parameter-estimation approaches.

> **Status:** Research software under active development. The repository structure and interfaces may change.

## Repository structure

```text
.
├── docs/
│   ├── descriptions/       # Model and parameter descriptions
│   └── schemas/            # Diagrams describing the framework
│
├── hack/model/
│   ├── drive_cycles/       # Drive-cycle definitions
│   ├── functions/          # Simulation, protocol, and solver utilities
│   ├── output/             # Generated simulation outputs
│   └── *.py                # Synthetic-data generation scripts
│
├── src/
│   ├── adapter_functions/  # Interfaces between data, methods, and models
│   ├── data/               # Generated study-case data
│   ├── method_functions/   # Parameter-estimation methods
│   ├── model_functions/    # Model construction and evaluation
│   ├── output/             # Estimation results and figures
│   └── main.py             # Main comparison workflow
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `hack/model/` directory contains the tools used to generate synthetic measurements. The main comparison framework is located in `src/`.

Generated datasets, figures, parameter files, and other results are written to the corresponding `output/` directories and are not tracked by Git.

## Installation

Clone the repository and create a Python environment:

```bash
git clone <repository-url>
cd <repository-name>

conda create -n parameter-comparison python=<python-version>
conda activate parameter-comparison

python -m pip install -r requirements.txt
```

## Running the framework

Run the main comparison workflow from the repository root:

```bash
python src/main.py
```

Synthetic battery measurements can be generated using the scripts in:

```text
hack/model/
```

The execution settings, input paths, models, protocols, methods, and study cases are currently configured within the corresponding Python scripts.

## Implemented components

The repository currently contains:

* Synthetic battery-data generation using PyBaMM models
* Battery test and drive-cycle definitions
* Equivalent circuit model generation
* Least-squares and recursive parameter-estimation methods
* Adapter functions connecting datasets, methods, and models
* Study-case generation
* Result export and visualisation utilities

## Data

Large generated datasets and intermediate simulation files are not included in the Git repository. Information about downloading the research dataset will be added when the corresponding data release becomes available.

## Documentation

Framework and model-structure diagrams are available under `docs/schemas/`.

## Citation

Citation information for the associated paper and archived software release will be added after publication.

## Licence and funding

Except where otherwise stated, this project is licensed under the [Creative Commons Attribution 4.0 International Licence](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). Reuse and adaptation are permitted provided that appropriate credit is given and any changes are indicated. See the [`LICENSE`](LICENSE) file for details.

This work was carried out as part of **FME Battery** and was funded by the **Research Council of Norway** under project number **350373**.

Third-party software and materials remain subject to their respective licences.

