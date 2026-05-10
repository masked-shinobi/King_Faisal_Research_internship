import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from clearml import Task, Dataset

def train_xgboost():
    # Initialize a ClearML Task
    task = Task.init(
        project_name="TrashNext Optimization",
        task_name="XGBoost Hyperparameter Search",
        task_type=Task.TaskTypes.training
    )

    # --- Load pre-extracted features from ClearML Dataset ---
    dataset_path = Dataset.get(
        dataset_name="TrashNext Features",        # Name of your dataset in ClearML
        dataset_project="TrashNext Optimization" # Project where the dataset lives
    ).get_local_copy()

    print("Loading pre-extracted features from ClearML Dataset...")
    X_train = np.load(f"{dataset_path}/X_train.npy")
    y_train = np.load(f"{dataset_path}/y_train.npy")
    X_valid = np.load(f"{dataset_path}/X_valid.npy")
    y_valid = np.load(f"{dataset_path}/y_valid.npy")
    print("Features loaded successfully.")

    # --- Create DMatrix ---
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # --- Get Hyperparameters from ClearML ---
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'learning_rate': float(task.get_parameter('General/lr', default=0.01)),
        'max_depth': int(task.get_parameter('General/max_depth', default=5)),
        'subsample': float(task.get_parameter('General/subsample', default=0.8)),
        'colsample_bytree': float(task.get_parameter('General/colsample_bytree', default=0.8)),
        'gamma': float(task.get_parameter('General/gamma', default=0)),
        'seed': 42
    }

    num_rounds = int(task.get_parameter('General/num_rounds', default=200))
    batch_size = int(task.get_parameter('General/batch_size', default=32))

    # --- Training with Evaluation ---
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    evals_result = {}

    print("Starting XGBoost training...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=50
    )
    print("Training finished.")

    # --- Report Metrics to ClearML ---
    final_val_loss = evals_result['valid']['mlogloss'][-1]
    y_valid_pred = np.argmax(model.predict(dvalid), axis=1)
    final_val_accuracy = accuracy_score(y_valid, y_valid_pred)

    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

    logger = task.get_logger()
    # Make sure the series match your optimizer exactly
    logger.report_scalar(title="Validation Metrics", series="Loss (mlogloss)", value=final_val_loss, iteration=0)
    logger.report_scalar(title="Validation Metrics", series="Accuracy", value=final_val_accuracy, iteration=0)

    print("Task completed. Metrics have been reported to ClearML.")

if __name__ == '__main__':
    train_xgboost()
