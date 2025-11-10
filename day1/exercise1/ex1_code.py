import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
import shap


## 2. Data Loading & Preprocessing
df = pd.read_excel('/gpfsnyu/scratch/ys6132/2025_summer_school_ML4MS/day1/exercise1/Exercise_1_Regression_Algorithms.xlsx', engine='openpyxl')
feature_names = ['feat1', 'feat2', 'feat3', …]   # descriptor columns
   target_name   = 'y'                              # the regression label
   X = df[feature_names]
   y = df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


## 3. Model Construction & Hyperparameter Optimization
### 3.1 Define XGBoost Objective
def objective(params):
    model = XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              early_stopping_rounds=20, verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {'loss': rmse, 'status': STATUS_OK}
### 3.2 Hyperparameter Search Space
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'max_depth':    hp.quniform('max_depth', 3, 12, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample':    hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
}
### 3.3 Run Optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=np.random.RandomState(42)
)
print("Best hyperparameters:", best)

## 4. Model Evaluation
# Train final model with best params
best_params = {
       'n_estimators': int(best['n_estimators']),
       'max_depth':    int(best['max_depth']),
       'learning_rate': best['learning_rate'],
       'subsample':    best['subsample'],
       'colsample_bytree': best['colsample_bytree'],
       'random_state': 42,
       'tree_method':  'hist'
   }
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)
# Predict & compute metric
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
ev   = explained_variance_score(y_test, y_pred)
   
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
print(f"Explained Variance: {ev:.4f}")

# Scatter plot of predictions vs. true values
   plt.figure(figsize=(6,6))
   sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
   plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2)
   plt.xlabel('True Values')
   plt.ylabel('Predictions')
   plt.title('Prediction vs. True Scatter')
   plt.show()

## 5. SHAP Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')
shap.summary_plot(shap_values, X_test)  # beeswarm