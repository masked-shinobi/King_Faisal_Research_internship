# TrashNeXt: Hybrid Deep Learning for Waste Classification

## Description

TrashNeXt is a research project focused on classifying waste materials into 9 categories using hybrid deep learning architectures. The project combines Vision Transformers (LeViT, Swin) with EfficientNetV2-S as feature extractors, followed by XGBoost as the classifier. The research explores various model combinations, hyperparameter optimization with Optuna, and explainability via SHAP analysis.

This project achieves up to **94.37% test accuracy** using the Swin-Base + EfficientNetV2-S hybrid model with XGBoost classifier and Optuna HPO.

---

## Dataset Information

The **TrashNeXt Dataset** contains 23,625 waste images across 9 categories:

| Class | Train | Valid | Test | Total |
|-------|-------|-------|------|-------|
| cardboard | 1,886 | 236 | 235 | 2,357 |
| e-waste | 2,404 | 301 | 301 | 3,006 |
| foam_rubber | 2,289 | 287 | 287 | 2,863 |
| glass | 2,009 | 251 | 252 | 2,512 |
| medical | 1,565 | 196 | 196 | 1,957 |
| metal | 2,065 | 258 | 258 | 2,581 |
| organic | 2,391 | 299 | 299 | 2,989 |
| paper | 2,155 | 269 | 270 | 2,694 |
| plastic | 2,135 | 267 | 267 | 2,669 |

**Dataset Split:**
- Training: 18,898 images (80%)
- Validation: 2,362 images (10%)
- Testing: 2,364 images (10%)

**Preprocessing:** Corrupted images and hidden macOS files (.DS_Store) are automatically removed.

---

## Code Information

### Project Structure

```
├── 01_Paper_Drafts/                    # Research paper drafts
├── 02_Documentation/                   # Project documentation and reports
├── 03_Notebooks_Main_Models/           # Main model implementations
│   ├── LeViT (1).ipynb                 # LeViT-128S standalone classifier
│   └── Vit256+Efficient_V2s with Explainable AI (1).ipynb  # Hybrid model with SHAP
├── 04_Notebooks_Experiments/           # Experiment notebooks
│   ├── levit128-effnetv2s-xg.ipynb     # LeViT-128 + EfficientNetV2-S → XGBoost
│   ├── vit_large-effnetv2s-xg.ipynb     # ViT-Large + EfficientNetV2-S → XGBoost
│   ├── swin_small-optunaHPO.ipynb      # Swin-Small with Optuna HPO
│   ├── deitbase-effnetv2s-xg.ipynb     # DeiT-Base hybrid
│   ├── t2tvit-effnetv2s-xg.ipynb       # T2T-ViT hybrid
│   └── standardvit_base-effnetv2s-xg.ipynb  # Standard ViT-Base hybrid
├── 05_Optimization/Code/                # Hyperparameter optimization scripts
│   ├── launch_hpo.py                   # ClearML HPO launcher
│   └── xgboost_optimizer.py            # XGBoost training with HPO
├── 06_Reference_Materials/             # Reference documents and Linux commands
├── 07_Images/                          # Image assets
├── FULL METRICS/                       # Final metrics and analysis
└── Fully_Done_Draft.docx              # Complete research paper
```

### Hybrid Model Architecture

The project implements a hybrid feature extraction approach:

```
Input Image
    │
    ├──► LeViT-256 / Swin-Base (Transformer branch)
    │         └─► Feature Vector (512 / 1024 dims)
    │
    └──► EfficientNetV2-S (CNN branch)
              └─► Feature Vector (1280 dims)
    │
    ▼
Feature Concatenation (1792 / 2304 dims)
    │
    ▼
PCA Dimensionality Reduction (For Swin-Base: to 256 dims)
    │
    ▼
XGBoost Classifier (with Optuna HPO)
    │
    ▼
9-Class Output (Waste Categories)
```

### Models Tested

| Model Combination | Test Accuracy | Notes |
|-------------------|---------------|-------|
| LeViT-256 + EfficientNetV2-S + XGBoost | 89.81% | Architecture 1 unoptimised |
| LeViT-256 + EfficientNetV2-S + XGBoost | 90.52% | Architecture 1 optimised with Optuna HPO |
| Swin-Base + EfficientNetV2-S + XGBoost | 94.37% | Architecture 2 Best performing with Optuna HPO |

---

## Usage Instructions

### 1. Dataset Setup

Mount Google Drive and extract the dataset:
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
zip_path = "/content/drive/MyDrive/Research_project/TrashNeXt Dataset.zip"
extract_path = "/content"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### 2. Run Hybrid Feature Extraction

```python
# Example: LeViT-256 + EfficientNetV2-S
from HybridFeatureExtractor import HybridFeatureExtractor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridFeatureExtractor().to(device)
model.eval()

# Extract features
features, labels = extract_features(data_loader, model, device)

# Save features
np.save('X_train.npy', features)
```

### 3. Train XGBoost Classifier

```python
import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softprob',
    'num_class': 9,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6
}

model = xgb.train(params, dtrain, num_boost_round=100)
predictions = model.predict(dtest)
```

### 4. Hyperparameter Optimization with Optuna

```python
import optuna

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
```

### 5. SHAP Explainability

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample)
```

---

## Requirements

### Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
timm>=1.0.0
transformers>=4.30.0
xgboost>=2.0.0
optuna>=3.0.0
shap>=0.40.0
scikit-learn>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
clearml>=1.0.0  # For HPO tracking
```

### Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (recommended for feature extraction)
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 5GB for dataset and models

---

## Methodology

### Phase 1: Data Preprocessing
1. Extract dataset from ZIP file
2. Remove corrupted/truncated images
3. Remove macOS hidden files (.DS_Store)
4. Verify image integrity using PIL

### Phase 2: Feature Extraction
1. Load pre-trained transformer models (LeViT/ViT/Swin)
2. Load pre-trained EfficientNetV2-S
3. Extract features from both branches
4. Concatenate features (1792-2304 dimensional vectors)
5. Save features to .npy files

### Phase 3: Model Training
1. Load pre-extracted features
2. Apply StandardScaler normalization
3. Train XGBoost classifier with cross-validation
4. Evaluate on validation set
5. Fine-tune hyperparameters if needed

### Phase 4: Hyperparameter Optimization
1. Define search space (learning rate, max_depth, subsample, etc.)
2. Run Optuna TPE optimizer
3. Use early stopping to prevent overfitting
4. Select best hyperparameters based on validation F1

### Phase 5: Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- True Positive Rate (TPR) and False Positive Rate (FPR)
- Log Loss and AUROC (Macro Average)
- Confusion Matrix analysis

### Phase 6: Explainability (SHAP)
1. Generate SHAP values for test samples
2. Visualize feature importance
3. Analyze prediction explanations
4. Identify model biases and error patterns

---

## Performance Summary

### Best Model: Swin-Base + EfficientNetV2-S + XGBoost (Architecture 2)

| Metric | Training | Validation | Testing |
|--------|----------|------------|---------|
| Accuracy | 99.99% | 100.0% | 94.37% |
| Precision | 99.99% | 100.0% | 94.36% |
| Recall (TPR) | 99.99% | 100.0% | 94.24% |
| F1 Score | 99.99% | 100.0% | 94.29% |
| Loss (mlogloss)| 0.0051 | 0.0043 | N/A |
| AUROC | N/A | N/A | 99.76% |
| FPR | 0.00% | 0.00% | 0.70% |

---

## License & Contribution Guidelines

This research project is open-source for academic purposes.

### Citation

If you use this code or dataset in your research, please cite:

```
TrashNeXt: Hybrid Deep Learning for Waste Classification
King Faisel Research Project
```

### Contribution

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Contact

For questions or collaborations, please contact the research team.