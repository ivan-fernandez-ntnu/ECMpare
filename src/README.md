# Comparison Framework

This directory contains the main ECMpare comparison workflow.

It takes prepared battery data, converts it into the format required by a parameter-estimation method, estimates model parameters, converts those parameters into model-compatible functions, simulates the resulting model, and evaluates its performance against the reference data.

Detailed component-level documentation is available in the project Wiki.

---

## Directory structure

```text
src/
├── adapter_functions/     # Interfaces between data, methods, and models
├── data/                  # Input and generated study-case data
├── method_functions/      # Parameter-estimation methods
├── model_functions/       # Equivalent circuit model generation
├── result_functions/      # Evaluation, metrics, plots, and result export
├── utils/                 # Shared utilities such as registries
├── output/                # Generated comparison results
└── main.py                # Main comparison workflow
```

---

## Workflow

The core ECMpare structure is:

```text
DATA
 |
 v
Data-to-Method Adapter
 |
 v
METHOD
 |
 v
Method-to-Model Adapter
 |
 v
MODEL
 |
 v
Evaluation
 |
 v
RESULTS
```

More explicitly:

```text
Reference data
    |
    v
Study case
    |
    v
data_to_method()
    |
    v
Method-specific input
    |
    v
estimate_eecm_parameters()
    |
    v
Estimated parameter values
    |
    v
method_to_model()
    |
    v
Model-compatible parameter functions
    |
    v
model_generator()
    |
    v
Parameterized PyBaMM model
    |
    v
Evaluation simulation
    |
    v
calculate_reference_vs_solution_errors()
    |
    v
append_execution_results()
```

This separation allows data sources, estimation methods, and models to be modified independently as long as they satisfy the interfaces between each layer.

---

## Main script

The comparison workflow is configured and executed from:

```text
src/main.py
```

Run it from the repository root:

```bash
python src/main.py
```

The main configuration defines the components to be tested, including:

```text
Reference parameter set
Study cases
Training protocol
Evaluation protocol
Estimated model type
Parameter-estimation method
Method solver
Initial parameter guess
Parameter bounds
Evaluation solver
Initial SOC
Initial temperature
```

The script then iterates through the requested combinations and records the results.

---

## Data and study cases

Input data used by the comparison framework are stored under:

```text
src/data/
```

Study-case generation is implemented in:

```text
src/adapter_functions/data_to_method_functions.py
```

The current study cases include:

```text
sc-0    Baseline
sc-1    Measurement error
sc-2    SOC error
sc-3    OCV error
sc-4    Combined errors
```

The study-case functions prepare the variables required by the parameter-estimation workflow, including:

```text
Voltage [V]
Current [A]
Open-circuit voltage [V]
SOC [-]
Temperature
Cycle
```

See the Wiki page **Study Cases** for details.

---

## Data-to-Method Adapter

The interface between input data and parameter-estimation methods is implemented in:

```text
src/adapter_functions/data_to_method_functions.py
```

The main function is:

```python
data_to_method()
```

Its role is to convert the standardized input DataFrame into the arrays and metadata required by the selected estimation method.

Depending on the model and method, the adapter prepares quantities such as:

```text
Terminal voltage
OCV
Current
Lagged voltage values
Lagged current values
SOC
Temperature
Cycle
Sampling period
Initial guesses
Parameter bounds
```

The estimation methods therefore do not need to know the original structure of the source dataset.

See the Wiki page **Data-to-Method Adapter**.

---

## Parameter-estimation methods

Parameter-estimation methods are implemented under:

```text
src/method_functions/
```

The main dispatcher is:

```python
estimate_eecm_parameters()
```

The currently implemented methods include:

```text
LS
MWLS
NLLS
MWNLLS
TROLS
MWTROLS
SLSQP
MWSLSQP
```

The methods return a common structure:

```python
{
    (temperature, soc, current): theta_array,
    ...
}
```

This allows both global and operating-condition-dependent parameter estimates to use the same downstream interface.

See the Wiki page **Paramater Estimation Methods**.

---

## Method-to-Model Adapter

The conversion from method output to model parameters is implemented in:

```text
src/adapter_functions/method_to_model_functions.py
```

The main interface is:

```python
method_to_model()
```

This layer is responsible for:

```text
Converting estimator coefficients to physical parameters
Filtering invalid parameter values
Converting time constants into capacitances where required
Creating parameter functions
Creating the OCV-SOC interpolant
```

For example:

```text
Estimator output:
[R0, R1, tau1]

        |
        v

Model input:
R0
R1
C1 = tau1 / R1
OCV(SOC)
```

The generated resistance and capacitance parameters can depend on:

```text
Temperature
Current
SOC
```

See the Wiki page **Method to Model**.

---

## Equivalent Circuit Models

Model construction is implemented in:

```text
src/model_functions/model_generator_functions.py
```

through:

```python
model_generator()
```

The currently supported model structures are:

```text
rint
thevenin_0rc
thevenin_1rc
thevenin_2rc
```

These models are implemented using:

```python
pybamm.equivalent_circuit.Thevenin
```

The model generator combines:

```text
Estimated parameter functions
OCV-SOC relationship
Reference cell capacity
Voltage limits
```

with PyBaMM's ECM parameter infrastructure.

See the Wiki page **Equivalent Circuit Models**.

---

## Model evaluation

Model evaluation and result processing are implemented in:

```text
src/result_functions/result_functions.py
```

The main comparison function is:

```python
calculate_reference_vs_solution_errors()
```

The current implementation compares:

```text
Terminal voltage
Reaction + concentration overpotential
Ohmic losses
```

between the reference model and the estimated ECM.

The calculated metrics include:

```text
MAE
RMSE
MBE
R²
```

Metrics are calculated both:

```text
Across the complete evaluation profile
```

and:

```text
Within SOC intervals
```

The default SOC bin width is:

```text
0.01
```

corresponding to 1% SOC.

---

## Result export

Results can be aggregated using:

```python
append_execution_results()
```

The generated files can include:

```text
execution_summary.csv
execution_summary_by_soc.csv
residuals/
```

The global summary contains one row per tested configuration, including metadata such as:

```text
Model
Parameter set
Method
Evaluation solver
Training protocol
Evaluation protocol
Temperature
Time step
Study case
```

along with the calculated error metrics.

Generated files are stored under:

```text
src/output/
```

and are generally not intended to be tracked by Git.

---

## Parameter surfaces

Estimated model parameters can also be visualized using:

```python
plot_parameter_surfaces_vs_c_rate_soc_temperature()
```

This evaluates a parameter function over:

```text
Temperature
C-rate
SOC
```

and generates 3D surfaces for parameters such as:

```text
R0
R1
C1
R2
C2
```

This is useful for inspecting how locally estimated parameters are interpolated across the operating domain.

---

## Adding a new method

A new parameter-estimation method normally requires:

```text
1. Implement and register the estimator
   src/method_functions/

2. Ensure the data-to-method adapter provides its required inputs
   src/adapter_functions/data_to_method_functions.py

3. Implement the method-to-model conversion
   src/adapter_functions/method_to_model_functions.py

4. Add the method configuration to src/main.py
```

If the method targets an already supported ECM and uses the existing standardized data representation, no changes to the model generator may be required.

---

## Adding a new model

A new model may require changes in several components:

```text
Data-to-method adapter
Parameter-estimation methods
Method-to-model adapter
Model generator
Evaluation functions
```

The complete combination should preserve the main ECMpare interface:

```text
Data
 -> Method
 -> Model
 -> Evaluation
```

See the Wiki page **How to Test a Data-Method-Model Combination in ECMpare** for the current integration requirements.

---

## Outputs

Generated files under:

```text
src/output/
```

can include:

```text
Aggregated metric tables
SOC-resolved metric tables
Residual time series
Evaluation figures
Parameter-surface figures
Simulation outputs
```

These files can become large and should generally remain excluded from version control.

---

## Installation

The comparison framework uses the environment defined for the complete ECMpare repository.

From the repository root:

```bash
pip install -r requirements.txt
```

A Conda environment is recommended for reproducibility.

---

## Documentation

Detailed documentation is available in the ECMpare Wiki:

- **Framework Architecture**
- **Study Cases**
- **Data-to-Method Adapter**
- **Paramater Estimation Methods**
- **Method to Model**
- **Equivalent Circuit Models**
- **Numerical Solvers**
- **Model Evaluation and Results**
- **How to Test a Data-Method-Model Combination in ECMpare**

Synthetic-data generation is documented separately under:

```text
hack/model/
```

and in the corresponding Wiki pages.

The root [`README.md`](../README.md) provides the general project overview, installation instructions, licensing information, and project context.