# ex1_code.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
import shap

# 1. Load data
df = pd.read_excel(
    'Exercise_1_Regression_Algorithms.xlsx',
    engine='openpyxl'
)

# 2. Define features & target
#    drop ID/SMILES‐like columns; target is the activation energy 'Ea'
drop_cols = ['ID', 'Reactant', 'Radical', 'Ea']
feature_names = [c for c in df.columns if c not in drop_cols]
target_name   = 'Ea'

X = df[feature_names].values
y = df[target_name].values

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5. Hyperopt objective (using scaled data)
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
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],

        verbose=False
    )
    preds = model.predict(X_test_scaled)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    return {'loss': rmse, 'status': STATUS_OK}

space = {
    'n_estimators':    hp.quniform('n_estimators', 50, 500, 10),
    'max_depth':       hp.quniform('max_depth', 3, 12, 1),
    'learning_rate':   hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample':       hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree':hp.uniform('colsample_bytree', 0.6, 1.0),
}

trials = Trials()
rng = np.random.default_rng(42)
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=rng
)
print("Best hyperparameters:", best)

# 6. Train final model
best_params = {
    'n_estimators':    int(best['n_estimators']),
    'max_depth':       int(best['max_depth']),
    'learning_rate':   best['learning_rate'],
    'subsample':       best['subsample'],
    'colsample_bytree':best['colsample_bytree'],
    'random_state':    42,
    'tree_method':     'hist'
}
model = XGBRegressor(**best_params)
model.fit(X_train_scaled, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_scaled)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
ev   = explained_variance_score(y_test, y_pred)

print(f"MSE:               {mse:.4f}")
print(f"RMSE:              {rmse:.4f}")
print(f"MAE:               {mae:.4f}")
print(f"R²:                {r2:.4f}")
print(f"Explained Variance:{ev:.4f}")

# 8. Scatter plot (matplotlib only)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.xlabel('True Ea')
plt.ylabel('Predicted Ea')
plt.title('True vs Predicted Activation Energy')
plt.tight_layout()
plt.show()

# 9. SHAP analysis
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Summary bar plot
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test_scaled, columns=feature_names),
    plot_type='bar'
)

# Beeswarm plot
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test_scaled, columns=feature_names)
)
