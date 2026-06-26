# Chrun
<img width="1917" height="804" alt="Screenshot 2026-06-26 114611" src="https://github.com/user-attachments/assets/3705ce81-70b1-4d21-a823-da910d8a8d3b" />
<img width="1917" height="1001" alt="Screenshot 2026-06-26 114743" src="https://github.com/user-attachments/assets/d50c6096-2f91-4330-976e-17950748cc87" />
<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/f062aa41-2f4a-4c2f-8b7f-d9fc97c0917b" />

XGBoost Experiment Results (2026-06-23)
Experiment Name: xgb_experiment_20260623_195503

Training Configuration:

Total Fits: 30
Training Samples: 4,225
Features: 36
Class Distribution: {0: 3,104, 1: 1,121}
Total Training Time: 511.29 seconds
Model Parameters:

JSON
{
  "objective": "binary:logistic",
  "random_state": 42,
  "max_depth": 6,
  "min_child_weight": 5,
  "subsample": 0.85,
  "colsample_bytree": 0.7,
  "learning_rate": 0.01,
  "n_estimators": 500,
  "scale_pos_weight": 2.769,
  "gamma": 0.05,
  "reg_alpha": 1.0,
  "reg_lambda": 1.0,
  "max_bin": 256,
  "grow_policy": "lossguide",
  "max_leaves": 32,
  "device": "cuda",
  "eval_metric": ["auc", "error", "logloss"],
  "early_stopping_rounds": 50
}
Performance Metrics:

Metric	Train	Validation	Test
Accuracy	0.8182	0.7566	0.7679
AUC	0.9171	0.8458	0.8592
F1-Score	0.7221	0.6308	0.6465
PR-AUC	-	-	0.7024
CV AUC Mean ± Std	-	-	0.8391 ± 0.0184
Classification Report (Test Set):

Class	Precision	Recall	F1-Score	Support
0 (No Churn)	0.9126	0.7565	0.8273	1,035
1 (Churn)	0.5426	0.7995	0.6465	374
Macro Avg	0.7276	0.7780	0.7369	1,409
Weighted Avg	0.8144	0.7679	0.7793	1,409
Confusion Matrix (Test Set):

Code
                Predicted No Churn  Predicted Churn
Actual No Churn          783               252
Actual Churn              75               299
Feature Importance (Top Features):

is_month_to_month: 0.2095
Contract: 0.1812
churn_risk_score: 0.1427
tenure_monthly_ratio: (top ranked)
MonthlyCharges: (included)
Feature Set (36 Features):

customerID, gender, SeniorCitizen, Partner, Dependents
tenure, PhoneService, MultipleLines, InternetService
OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
Contract, PaperlessBilling, PaymentMethod
MonthlyCharges, TotalCharges
tenure_group, tenure_to_max, tenure_monthly_ratio, charge_group
total_services, has_internet, has_phone, is_month_to_month, is_automatic_payment
is_senior, has_dependents, churn_risk_score
SeniorCitizen_zscore, tenure_zscore, MonthlyCharges_zscore, TotalCharges_zscore
Ray Cluster Resources Used:

CPU: 2.0
GPU: 1.0 (T4 accelerator)
Memory: 12.88 GB
Object Store Memory: 6.44 GB
Logs:

Main Log: logs/xgb_experiment_20260623_195503.log
Fits Log: logs/xgb_experiment_20260623_195503_fits.log
Metrics Log: logs/xgb_experiment_20260623_195503_metrics.json


## XGBoost
The XGBoost model was trained for binary churn prediction using 36 features. It was configured with `objective='binary:logistic'`, `max_depth=6`, `min_child_weight=5`, `subsample=0.85`, `colsample_bytree=0.7`, `learning_rate=0.01`, `n_estimators=500`, `gamma=0.05`, `reg_alpha=1.0`, `reg_lambda=1.0`, and `early_stopping_rounds=50` on `cuda`.

Key results:
- Test accuracy: **0.7679**
- Test AUC: **0.8592**
- Test F1-score: **0.6465**
- PR-AUC: **0.7024**
- CV AUC mean: **0.8391 ± 0.0184**

Test classification performance:
- No churn: precision **0.9126**, recall **0.7565**, F1-score **0.8273**
- Churn: precision **0.5426**, recall **0.7995**, F1-score **0.6465**

Top feature importances:
- `is_month_to_month`: **0.2095**
- `Contract`: **0.1812**
- `churn_risk_score`: **0.1427**
- `tenure_monthly_ratio`: **0.0322**
- `OnlineSecurity`: **0.0270**
- `TechSupport`: **0.0221**

Feature importance and metrics visualizations:

![Feature importance](feature_importance_20260623_200350.png)

![Metrics](metrics_20260623_200350.png)

(with EDA handling NaN values by deriving tenure-based and monthly charge-based imputations)


## Xgboost
(old version)
![image](https://github.com/user-attachments/assets/becf4ad5-d714-42ff-bf9d-6a8e9d8bab84)

## Lightbgm
()
``Fitting 5 folds for each of 720 candidates, totalling 3600 fits``
![image](https://github.com/user-attachments/assets/073d1a09-7153-4e93-9747-13a755164591)


