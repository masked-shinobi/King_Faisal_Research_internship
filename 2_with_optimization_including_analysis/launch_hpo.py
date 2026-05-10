from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange
from clearml.automation.optuna.optuna import OptimizerOptuna

def run_optimizer():
    # Initialize the ClearML optimizer task
    task = Task.init(
        project_name="TrashNext Optimization",
        task_name="XGBoost HPO Launcher",
        task_type=Task.TaskTypes.optimizer
    )

    # --- Define hyperparameter search space ---
    param_space = [
        UniformParameterRange(name="General/lr", min_value=0.001, max_value=0.1),
        DiscreteParameterRange(name="General/max_depth", values=[3,4,5,6,7,8,9,10]),
        UniformParameterRange(name="General/subsample", min_value=0.5, max_value=1.0),
        UniformParameterRange(name="General/colsample_bytree", min_value=0.5, max_value=1.0),
        UniformParameterRange(name="General/gamma", min_value=0.0, max_value=0.5),
        DiscreteParameterRange(name="General/num_rounds", values=[100,150,200,250,300]),
        DiscreteParameterRange(name="General/batch_size", values=[16,32,64,128])
    ]

    # --- Initialize the HPO optimizer ---
    optimizer = HyperParameterOptimizer(
        base_task_id="376e1b326ede43308abbcdd416a127b1",
    hyper_parameters=param_space,
    objective_metric_title="Validation Metrics",
    objective_metric_series="Loss (mlogloss)",
    objective_metric_sign="min",
    optimizer_class=OptimizerOptuna,
    max_iteration_per_job=1,
    execution_queue="default",
    total_max_jobs=20,
    max_concurrent_iterations=4,
    report_period=0.1
    )

    # --- Start optimization ---
    optimizer.start()
    optimizer.wait()
    optimizer.stop()
    print("HPO completed.")

if __name__ == "__main__":
    run_optimizer()
