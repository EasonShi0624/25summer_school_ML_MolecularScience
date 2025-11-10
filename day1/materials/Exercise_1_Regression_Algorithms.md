# Exercise 1: Regression Algorithms


## 1. Environment Setup & Library Imports

1. **Create a Python 3.9 virtual environment**

   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

2. **Install required packages**

   ```bash
   pip install rdkit-pypi tqdm joblib openpyxl seaborn scikit-learn hyperopt shap xgboost==2.1.4 tensorflow==2.17
   ```

3. **Import libraries**

   ```python
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
   ```

------

## 2. Data Loading & Preprocessing

1. **Load your dataset** (replace with your actual path):

   ```python
   df = pd.read_excel('path/to/your/data.xlsx', engine='openpyxl')
   ```

2. **Define feature and target columns**:

   ```python
   feature_names = ['feat1', 'feat2', 'feat3', …]   # descriptor columns
   target_name   = 'y'                              # the regression label
   X = df[feature_names]
   y = df[target_name]
   ```

3. **Train–test split** (default 80:20; adjust `test_size` as needed):

   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

4. **Optional scaling** (e.g. for neural nets):

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled  = scaler.transform(X_test)
   ```

------

## 3. Model Construction & Hyperparameter Optimization

### 3.1 Define XGBoost Objective

```python
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
```

### 3.2 Hyperparameter Search Space

```python
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'max_depth':    hp.quniform('max_depth', 3, 12, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample':    hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
}
```

### 3.3 Run Optimization

```python
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
```

------

## 4. Model Evaluation

1. **Train final model with best params**

   ```python
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
   ```

2. **Predict & compute metrics**

   ```python
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
   ```

3. **Scatter plot of predictions vs. true values**

   ```python
   plt.figure(figsize=(6,6))
   sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
   plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2)
   plt.xlabel('True Values')
   plt.ylabel('Predictions')
   plt.title('Prediction vs. True Scatter')
   plt.show()
   ```

------

## 5. SHAP Analysis

1. **Compute SHAP values**

   ```python
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_test)
   ```

2. **Feature importance summary**

   ```python
   shap.summary_plot(shap_values, X_test, plot_type='bar')
   shap.summary_plot(shap_values, X_test)  # beeswarm
   ```

------

## 6. Alternative Regression Models

- **Gradient Boosting Regressor (GBR)**

  ```python
  from sklearn.ensemble import GradientBoostingRegressor
  gbr = GradientBoostingRegressor(random_state=42)
  gbr.fit(X_train, y_train)
  ```

- **Neural Network Regressor**

  ```python
  from sklearn.neural_network import MLPRegressor
  nn = MLPRegressor(hidden_layer_sizes=(128,64),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=42)
  nn.fit(X_train_scaled, y_train)
  ```

> *Adjust the evaluation code above for each new model to compare performance.*

------

## 7. Next Steps & Best Practices

- **Cross-validation**: use `cross_val_score` or `KFold` for robust performance estimates.
- **Pipelines**: chain preprocessing and modeling via `Pipeline`.
- **Feature engineering**: explore polynomial terms, interactions, or embedding methods (e.g. with RDKit for chemical descriptors).
- **Regularization**: include L1/L2 penalties or early stopping in tree-based models.
- **Ensembling**: blend multiple regressors (e.g. stacking, voting) to improve generalization.
- **Logging & reproducibility**: record parameter choices, seeds, and dataset versions.

