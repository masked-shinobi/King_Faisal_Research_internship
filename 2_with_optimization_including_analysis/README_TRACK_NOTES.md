Title: Optimized Hybrid Model Track Notes

Description: This folder contains the two requested hybrid-model tracks for the optimization section using the strongest matching source files recovered from `Trashnext`.

Code Information:
- `efficientnetv2s_swinbase_xgboost_optuna_with_analysis.ipynb` is the confirmed Swin-Base + EfficientNetV2-S + XGBoost notebook with Optuna hyperparameter optimization.
- `efficientnetv2s_levit256_xgboost_best_available_with_analysis.ipynb` is the strongest LeViT-side source available in `Trashnext`. It is stored under `optuna optimised results`, but the notebook content is a LeViT-256 + EfficientNetV2-S + XGBoost analysis/XAI workflow rather than a direct standalone Optuna notebook.
- `launch_hpo.py` and `xgboost_optimizer.py` are the recovered optimization scripts from `Trashnext/Optimizer`.

Methodology:
1. Verified the real source notebooks in `Trashnext`.
2. Placed the confirmed Optuna Swin notebook into this folder.
3. Placed the closest available LeViT-side source from the optimized-results material into this folder because no separate LeViT Optuna notebook was present.
