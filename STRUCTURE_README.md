Project structure prepared on 2026-05-10.

Contents are organized into three categories:

1. Hybrid model code without optimization including analysis
2. Hybrid model with optimization including analysis
3. Explainable AI only

Target hybrid models:
- EfficientNetV2-S + LeViT-256 + XGBoost
- EfficientNetV2-S + Swin-Base + XGBoost

Included files:

1. Without optimization including analysis
- `1_without_optimization_including_analysis/efficientnetv2s_or_v2m_levit_pre_optimization_analysis_source.ipynb`
- Source recovered from git commit `af7edb5`
- Note: this recovered notebook indicates `EfficientNetV2-M + LeViT + XGBoost/CatBoost/LightGBM`, so it is not an exact filename/content match for `EfficientNetV2-S + LeViT-256 + XGBoost`.

2. With optimization including analysis
- `2_with_optimization_including_analysis/efficientnetv2s_swinbase_xgboost_optuna_with_analysis.ipynb`
- Supporting scripts:
  - `2_with_optimization_including_analysis/launch_hpo.py`
  - `2_with_optimization_including_analysis/xgboost_optimizer.py`
- This is the optimized Swin-Base + EfficientNetV2-S + XGBoost Optuna notebook available in the repository.

3. Explainable AI only
- `3_explainable_ai_only/efficientnetv2s_levit256_xgboost_explainable_ai.ipynb`
- This notebook includes the hybrid pipeline, evaluation, metrics, and SHAP/XAI sections for the LeViT-based model.

Coverage notes:
- No separate standalone XAI notebook was found for `EfficientNetV2-S + Swin-Base + XGBoost`.
- No exact recovered non-optimized notebook was found for `EfficientNetV2-S + Swin-Base + XGBoost`.
- No exact recovered optimized notebook was found for `EfficientNetV2-S + LeViT-256 + XGBoost`.
