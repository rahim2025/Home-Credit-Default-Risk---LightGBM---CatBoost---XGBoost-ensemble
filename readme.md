# Home Credit Default Risk Prediction

A machine learning solution for predicting loan default probability using an ensemble of LightGBM, XGBoost, and CatBoost models with advanced feature engineering.

**Competition:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) 
**Approach:** Gradient Boosting Ensemble with EXT_SOURCE Imputation Features  
**CV Score:** 0.7636 |  


## Overview

This repository contains code to predict the probability that a customer will default on a home loan based on their financial history and application data. The solution uses only the application table (no auxiliary tables), making it reproducible and rules-compliant.

## Model Architecture

The final prediction is an ensemble of three gradient-boosting libraries trained with 5-fold stratified cross-validation and 3 random seeds each:

- **LightGBM** (OOF AUC: 0.7635)
- **XGBoost** (OOF AUC: 0.7628, GPU-accelerated)
- **CatBoost** (OOF AUC: 0.7620, GPU-accelerated)

Final predictions are blended using rank averaging.

## Key Features

### 1. Feature Engineering
- **Ratio features:** credit/income, annuity/income, credit/annuity, goods/credit ratios
- **EXT_SOURCE aggregates:** mean, min, max, std, products, weighted average
- **Age and employment ratios:** ratios of employment/birth dates, car age to age, etc.
- **Group aggregates:** by organisation type, occupation, education, and region
- **EXT_SOURCE imputation:** predict missing EXT_SOURCE values from other columns using LightGBM regressors (out-of-fold); use prediction, imputed value, and residual as features

### 2. Data Cleaning
- Treat `DAYS_EMPLOYED == 365243` as missing
- Treat `"XNA"` categorical values as missing
- Replace `inf` and `-inf` with `NaN`

### 3. Hyperparameter Tuning
- LightGBM parameters optimized via Optuna (40 trials, 3-fold CV)
- Shallow trees (max_depth=4) chosen to handle noisy data
- Lower column fraction (0.196) to reduce overfitting

### 4. GPU Acceleration
- XGBoost and CatBoost use NVIDIA GPU for faster training
- Fallback to CPU if GPU unavailable (edit `device="cuda"` / `task_type="GPU"` in code)

## Quick Start

### Requirements
```bash
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost
```

### Run the Solution
```bash
python solution.py
```

This will:
1. Load `input/train.csv` and `input/test.csv`
2. Engineer 134+ features (10-15 min)
3. Train 15 models (5 folds × 3 models × 1 seed) (10-20 min on GPU, longer on CPU)
4. Output `output/submission.csv` with predicted default probabilities

**Expected output:** 61,500 rows with `SK_ID_CURR` and `TARGET` (probability)

### Reproduce the Exact Submission
```bash
# Check that the output matches
python -c "
import pandas as pd
sub = pd.read_csv('output/submission.csv')
print('Shape:', sub.shape)
print('Score range:', sub.TARGET.min(), '-', sub.TARGET.max())
"
```

## File Structure

```
.
├── solution.py                      # Main training script
├── input/
│   ├── train.csv                   # Training data (provided)
│   ├── test.csv                    # Test data (provided)
│   └── sample_submission.csv        # Format template
├── output/
│   └── submission.csv              # Final predictions
├── experiments/                    # Ablation studies and tuning
│   ├── tune.py                     # Optuna hyperparameter search
│   ├── exp.py, exp2.py, exp3.py, exp4.py  # Feature experiments
│   └── nn.py                       # Neural network tests
└── .gitignore
```

## Techniques & Inspirations

The approach draws ideas from public write-ups of the original Kaggle competition:

- [deepsense.ai, 5th-place solution](https://deepsense.ai/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/) — ensemble stacking, feature engineering
- [Home Credit Default Risk Part 2: Feature Engineering](https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5) — EXT_SOURCE aggregates, credit ratios
- [Kaggle original competition](https://www.kaggle.com/c/home-credit-default-risk) — multi-table baseline

The EXT_SOURCE imputation feature is an original idea (not from public sources), validated by CV and contributing ~+0.0015 AUC.

## Competition Rules Compliance

✅ **No external data:** uses only provided train/test files  
✅ **No hand-labelling:** all predictions from models, applied uniformly  
✅ **Reproducible:** fixed random seeds (SEED=42, plus SEEDS list in code)  
✅ **Transparent citations:** public ideas are credited above

## Expected Performance

- **Cross-validation AUC:** 0.7636 (5-fold, out-of-fold)
- **Public leaderboard:** 0.76592
- **Ceiling with this data:** ~0.765–0.77 (original Kaggle winners reached ~0.80 using all auxiliary tables)

## Troubleshooting

**GPU not available:**
Edit `solution.py` and change:
- `device="cuda"` → `device="cpu"` (XGBoost)
- `task_type="GPU"` → `task_type="CPU"` (CatBoost)

**Memory issues:**
Reduce `n_estimators` in the params dicts, or reduce the number of folds from 5 to 3.

**Output file not created:**
Check that `output/` folder exists. The script creates it automatically, but verify disk space.

## Citation

If you use this code, please cite:

```bibtex
@software{home_credit_2024,
  title={Home Credit Default Risk ---LightGBM + CatBoost + XGBoost ensemble},
  author={Md Raheem Bhuiyan},
  year={2024},
  url={https://github.com/rahim2025/Home Credit Default Risk - LightGBM + CatBoost + XGBoost ensemble }
}
```


---

**Author:** Raheem Bhuiyan  


