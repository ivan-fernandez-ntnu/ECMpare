import pybamm

def generate_solver(solver_name, output_vars, max_time_step = 0.01):
    if solver_name == "IDAKLUSolver":             
        solver = pybamm.IDAKLUSolver(
                output_variables = output_vars,
                rtol=1e-6,
                atol=1e-6,
                on_failure="error",   # "warn", "error", or "ignore" (default is raise)
                options={
                    "dt_max": max_time_step,        # cap internal step to 1 s (try 1.0, 0.5, 0.1 ...)
                    "dt_init": 0.0,       # 0.0 = let solver choose
                    "dt_min": 0.0,        # 0.0 = let solver choose
                    "max_num_steps": 200000,  # bump if you see "max steps" type failures
                    "num_threads": 1,
                },
            )
    elif solver_name == "CasadiSolver":
        solver = pybamm.CasadiSolver(
                # output_variables = output_vars,
                mode="safe",
                dt_max=max_time_step,                      # start smaller than the default 600 s 
                max_step_decrease_count=20,      # allow more halvings before error 
                return_solution_if_failed_early=True, 
                )
    else:
        solver = pybamm.solver()

    return solver

import pybamm


class RaiseOnExperimentFailure(pybamm.callbacks.Callback):
    """Raise an exception when a PyBaMM experiment becomes infeasible."""

    def on_experiment_error(self, logs):
        """Raise solver errors caught during experiment execution."""
        error = logs.get("error")

        if isinstance(error, Exception):
            raise error

        raise pybamm.SolverError(f"PyBaMM experiment solver error: {error}")

    def on_experiment_infeasible_event(self, logs):
        """Raise when the experiment stops due to an infeasible event."""
        termination = logs.get("termination", "unknown termination")
        step_number = logs.get("step number", "unknown step")
        cycle_number = logs.get("cycle number", "unknown cycle")
        operating_conditions = logs.get("step operating conditions", "unknown step")

        raise pybamm.SolverError(
            "PyBaMM experiment became infeasible.\n"
            f"Termination: {termination}\n"
            f"Cycle: {cycle_number}\n"
            f"Step: {step_number}\n"
            f"Operating conditions: {operating_conditions}"
        )

    def on_experiment_infeasible_time(self, logs):
        """Raise when the experiment reaches an unexpected default duration."""
        duration = logs.get("step duration", "unknown duration")
        operating_conditions = logs.get("step operating conditions", "unknown step")

        raise pybamm.SolverError(
            "PyBaMM experiment became infeasible due to time termination.\n"
            f"Duration: {duration}\n"
            f"Operating conditions: {operating_conditions}"
        )