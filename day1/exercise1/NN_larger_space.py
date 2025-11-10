#!/usr/bin/env python3
"""
Neural Network regression with hyperparameter optimization using Hyperopt and scikit-learn.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

#=============================
# Configuration
#=============================
random_seed = 42
excel_path  = 'Exercise_1_Regression_Algorithms.xlsx'
target_col  = 'Ea'
drop_cols   = ['ID', 'Reactant', 'Radical', target_col]

#=============================
# 1) Load & preprocess data
#=============================
df = pd.read_excel(excel_path, engine='openpyxl')
features = [c for c in df.columns if c not in drop_cols]
X = df[features].values
y = df[target_col].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

#=============================
# 2) Define Hyperopt search space
#=============================
parameter_space_nn = {
    'mlp__hidden_layer_sizes': hp.choice(
        'hidden_layer_sizes',
        [(50,), (100,), (50, 50), (100, 50), (128, 64), (128, 128, 64), (256, 128, 64)]
    ),
    'mlp__activation': hp.choice(
        'activation',
        ['tanh', 'relu', 'logistic']
    ),
    'mlp__solver': hp.choice(
        'solver',
        ['sgd', 'adam']
    ),
    'mlp__alpha': hp.loguniform(
        'alpha',
        np.log(1e-8),
        np.log(1e-1)
    ),
    'mlp__learning_rate': hp.choice(
        'learning_rate',
        ['constant', 'invscaling', 'adaptive']
    ),
    'mlp__learning_rate_init': hp.loguniform(
        'lr_init',
        np.log(1e-5),
        np.log(1e-1)
    ),
    'mlp__momentum': hp.uniform(
        'momentum',
        0.1,
        0.99
    ),
    'mlp__batch_size': hp.choice(
        'batch_size',
        [16, 32, 64, 128, 256]
    ),
    'mlp__tol': hp.loguniform(
        'tol',
        np.log(1e-6),
        np.log(1e-2)
    ),
    'mlp__max_iter': hp.choice(
        'max_iter',
        [500, 1000, 2000]
    ),
}

#=============================
# 3) Build pipeline
#=============================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
         max_iter=1000,
         early_stopping=True,
         validation_fraction=0.1,
         n_iter_no_change=20,
         random_state=random_seed
    ))
])

#=============================
# 4) Objective function
#=============================
def nn_objective(params):
    """
    Hyperopt objective: returns RMSE on 10-fold CV.
    """
    pipe.set_params(**params)
    neg_mse = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=10,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    rmse = np.sqrt(-neg_mse.mean())
    print(f"Params: {params} -> RMSE: {rmse:.4f}")
    return {'loss': rmse, 'status': STATUS_OK}

#=============================
# 5) Run Hyperopt search
#=============================
trials_nn = Trials()
best_params_nn = fmin(
    fn=nn_objective,
    space=parameter_space_nn,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials_nn,
    rstate=np.random.default_rng(random_seed)
)
print("Best hyperparameters (raw indices/values):", best_params_nn)

#=============================
# 6) Decode hp.choice indices
#=============================
choice_mappings = {
    'hidden_layer_sizes': [
        (50,), (100,), (50, 50), (100, 50),
        (128, 64), (128, 128, 64), (256, 128, 64)
    ],
    'activation':    ['tanh', 'relu', 'logistic'],
    'solver':        ['sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'batch_size':    [16, 32, 64, 128, 256],
    'max_iter':      [500, 1000, 2000],
}

final_params = {
    'hidden_layer_sizes': choice_mappings['hidden_layer_sizes'][best_params_nn['hidden_layer_sizes']],
    'activation':         choice_mappings['activation'][best_params_nn['activation']],
    'solver':             choice_mappings['solver'][best_params_nn['solver']],
    'alpha':              best_params_nn['alpha'],
    'learning_rate':      choice_mappings['learning_rate'][best_params_nn['learning_rate']],
    'learning_rate_init': best_params_nn['lr_init'],
    'momentum':           best_params_nn['momentum'],
    'batch_size':         choice_mappings['batch_size'][best_params_nn['batch_size']],
    'tol':                best_params_nn['tol'],
    'max_iter':           choice_mappings['max_iter'][best_params_nn['max_iter']],
}
print("Decoded best hyperparameters:", final_params)

#=============================
# 7) Train final model
#=============================
best_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
         **final_params,
         early_stopping=True,
         validation_fraction=0.1,
         n_iter_no_change=20,
         random_state=random_seed
    ))
])
best_nn.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_nn, 'nn_improved.pkl')
print("Saved improved NN model to 'nn_improved.pkl'")

#=============================
# 8) Evaluate on test set
#=============================
y_pred = best_nn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)

print("\n=== Neural Network Test Performance ===")
print(f"MSE:               {mse:.4f}")
print(f"RMSE:              {rmse:.4f}")
print(f"MAE:               {mae:.4f}")
print(f"R2:                {r2:.4f}")
print(f"Explained Variance:{ev:.4f}")

#=============================
# 9) Scatter plot True vs Pred
#=============================
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2)
plt.xlabel('True Ea')
plt.ylabel('NN Predicted Ea')
plt.title('NN: True vs Predicted Activation Energy')
plt.tight_layout()
plt.savefig('nn_true_vs_pred.png', dpi=300)
plt.show()

#=============================
# 10) SHAP Analysis for NN
#=============================
background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
explainer = shap.KernelExplainer(best_nn.predict, background)
shap_values = explainer.shap_values(X_test)

# 10a) Summary bar plot
plt.figure()
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test, columns=features),
    plot_type='bar',
    show=False
)
plt.tight_layout()
plt.savefig('nn_shap_summary_bar.png', dpi=300)
plt.close()

# 10b) Beeswarm plot
plt.figure()
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test, columns=features),
    show=False
)
plt.tight_layout()
plt.savefig('nn_shap_beeswarm.png', dpi=300)
plt.show()
