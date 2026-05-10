# Title

King Faisal Research Paper Code Submission

# Description

This repository contains the hybrid-model code used for the research submission centered on:

- EfficientNetV2-S + LeViT-256 + XGBoost
- EfficientNetV2-S + Swin-Base + XGBoost

The available files include:

- `efficientnetv2s_levit256_xgboost_explainable_ai.ipynb`
  LeViT-based hybrid pipeline with model analysis, evaluation metrics, and SHAP-based explainable AI.
- `efficientnetv2s_swinbase_xgboost_optuna.ipynb`
  Swin-Base hybrid pipeline with feature extraction, evaluation, and Optuna-based XGBoost hyperparameter optimization.
- `xgboost_optimizer.py`
  Standalone XGBoost optimization/training script.
- `launch_hpo.py`
  Hyperparameter optimization launcher using ClearML Optuna automation.

# Dataset Information

The repository does not include the raw dataset files themselves. The notebooks expect an image-classification dataset arranged in folder-based train/validation/test splits.

Observed dataset assumptions from the code:

- LeViT notebook paths:
  - `/content/dataset/Train`
  - `/content/dataset/Valid`
  - `/content/dataset/Test`
- Swin notebook paths:
  - image folders loaded through `torchvision.datasets.ImageFolder`
- `xgboost_optimizer.py` expects precomputed NumPy feature arrays from a ClearML dataset:
  - `X_train.npy`
  - `y_train.npy`
  - `X_valid.npy`
  - `y_valid.npy`

The code also includes dataset-cleaning steps for hidden files and corrupted/truncated images in the LeViT notebook.

# Code Information

Main code components:

1. `efficientnetv2s_levit256_xgboost_explainable_ai.ipynb`
   - Uses EfficientNetV2-S and LeViT-256 for hybrid feature extraction
   - Trains an XGBoost classifier on extracted features
   - Reports metrics such as classification report, confusion matrix, log loss, and weighted AUROC
   - Includes SHAP-based explainable AI workflow

2. `efficientnetv2s_swinbase_xgboost_optuna.ipynb`
   - Uses EfficientNetV2-S and Swin-Base for hybrid feature extraction
   - Trains and tunes XGBoost using Optuna
   - Produces optimization results and final performance analysis

3. `xgboost_optimizer.py`
   - Loads feature arrays from ClearML dataset storage
   - Trains XGBoost and evaluates validation accuracy

4. `launch_hpo.py`
   - Launches Optuna-based HPO through ClearML automation

# Usage Instructions

## To use the notebooks

1. Prepare the dataset in train/validation/test folder structure.
2. Open the required notebook in Jupyter or Google Colab.
3. Update dataset paths if needed.
4. Install the required dependencies listed below.
5. Run cells in order:
   - dataset preparation / cleaning
   - feature extraction
   - XGBoost training or optimization
   - evaluation and analysis
   - SHAP/XAI cells if using the LeViT notebook

## To use the Python scripts

1. Configure ClearML access if using `launch_hpo.py` or `xgboost_optimizer.py`.
2. Ensure the referenced ClearML dataset exists and contains the expected `.npy` files.
3. Run:

```bash
python xgboost_optimizer.py
python launch_hpo.py
```

# Requirements

The codebase references the following main dependencies:

- Python 3.x
- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `xgboost`
- `scikit-learn`
- `matplotlib`
- `shap`
- `optuna`
- `clearml`
- `transformers`
- `datasets`
- `accelerate`
- `Pillow`

A typical install command would be:

```bash
pip install torch torchvision timm numpy pandas xgboost scikit-learn matplotlib shap optuna clearml transformers datasets accelerate pillow
```

# Methodology

The general methodology used in the notebooks is:

1. Load and clean the image dataset.
2. Apply preprocessing and transforms.
3. Use pretrained vision backbones to extract hybrid features:
   - EfficientNetV2-S + LeViT-256
   - EfficientNetV2-S + Swin-Base
4. Concatenate extracted features.
5. Train an XGBoost classifier on the combined features.
6. Evaluate model performance on validation and/or test data.
7. For the Swin-based workflow, perform Optuna-based hyperparameter optimization.
8. For the LeViT-based workflow, generate SHAP-based explainability outputs.

# Citations

If this code or dataset is used in research, cite the relevant research paper, notebook results, and the external model/framework sources where appropriate.

Suggested items to cite where relevant:

- The associated King Faisal research paper
- XGBoost
- Optuna
- SHAP
- PyTorch
- TIMM pretrained model library

# License & Contribution Guidelines

License:

- No explicit license file is currently included in this repository.
- Treat the contents as research/project material unless a formal license is added.

Contribution Guidelines:

- Keep the code focused on approved research deliverables.
- Preserve notebook execution order and document any dataset path changes.
- If adding new experiments, separate them clearly from final submission files.
