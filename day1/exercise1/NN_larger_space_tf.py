#!/usr/bin/env python3
"""
Neural Network regression with hyperparameter optimization using Hyperopt and TensorFlow/Keras.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import shap

from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

#=============================
# Configuration & GPU setup
#=============================
random_seed = 42
excel_path  = 'Exercise_1_Regression_Algorithms.xlsx'
target_col  = 'Ea'
drop_cols   = ['ID', 'Reactant', 'Radical', target_col]

# Allow GPU memory growth if GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#=============================
# 1) Load & preprocess data
#=============================
df = pd.read_excel(excel_path, engine='openpyxl')
features = [c for c in df.columns if c not in drop_cols]
X = df[features].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#=============================
# 2) Define Hyperopt search space
#=============================
parameter_space_tf = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes',
        [(50,), (100,), (50, 50), (100, 50), (128, 64), (128, 128, 64), (256, 128, 64)]
    ),
    'activation':          hp.choice('activation', ['relu', 'tanh', 'elu']),
    'learning_rate_init':  hp.loguniform('lr_init', np.log(1e-5), np.log(1e-2)),
    'batch_size':          hp.choice('batch_size', [16, 32, 64, 128, 256]),
    'dropout':             hp.uniform('dropout', 0.0, 0.5),
    'l2':                  hp.loguniform('l2', np.log(1e-6), np.log(1e-2)),
    'epochs':              hp.choice('epochs', [50, 100, 200])
}

#=============================
# Model builder
#=============================
def build_model(params):
    model = models.Sequential()
    # add hidden layers
    for units in params['hidden_layer_sizes']:
        model.add(
            layers.Dense(
                units,
                activation=params['activation'],
                kernel_regularizer=regularizers.l2(params['l2'])
            )
        )
        model.add(layers.Dropout(params['dropout']))
    # output
    model.add(layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate_init']),
        loss='mse'
    )
    return model

#=============================
# 3) Hyperopt objective
#=============================
def nn_objective(params):
    # split for early validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2, random_state=random_seed
    )
    model = build_model(params)
    es = callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        callbacks=[es],
        verbose=0
    )
    preds = model.predict(X_val).ravel()
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    tf.keras.backend.clear_session()
    return {'loss': rmse, 'status': STATUS_OK}

#=============================
# 4) Run Hyperopt search
#=============================
trials = Trials()
best = fmin(
    fn=nn_objective,
    space=parameter_space_tf,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=np.random.default_rng(random_seed)
)
print("Best hyperparameters (raw):", best)

#=============================
# 5) Decode choices
#=============================
choice_mappings = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50), (128,64), (128,128,64), (256,128,64)],
    'activation':          ['relu','tanh','elu'],
    'batch_size':          [16,32,64,128,256],
    'epochs':              [50,100,200]
}
final_params = {
    'hidden_layer_sizes': choice_mappings['hidden_layer_sizes'][best['hidden_layer_sizes']],
    'activation':         choice_mappings['activation'][best['activation']],
    'learning_rate_init': best['lr_init'],
    'batch_size':         choice_mappings['batch_size'][best['batch_size']],
    'dropout':            best['dropout'],
    'l2':                 best['l2'],
    'epochs':             choice_mappings['epochs'][best['epochs']]
}
print("Decoded best hyperparameters:", final_params)

#=============================
# 6) Train final model
#=============================
final_model = build_model(final_params)
es = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
final_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=final_params['epochs'],
    batch_size=final_params['batch_size'],
    callbacks=[es],
    verbose=1
)
# Save model
final_model.save('nn_tf_model.h5')

#=============================
# 7) Evaluate on test set
#=============================
y_pred = final_model.predict(X_test_scaled).ravel()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)

print("\n=== TF Neural Network Test Performance ===")
print(f"MSE:               {mse:.4f}")
print(f"RMSE:              {rmse:.4f}")
print(f"MAE:               {mae:.4f}")
print(f"R2:                {r2:.4f}")
print(f"Explained Variance:{ev:.4f}")

#=============================
# 8) Scatter plot True vs Pred
#=============================
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Ea')
plt.ylabel('TF NN Predicted Ea')
plt.title('TF NN: True vs Predicted Ea')
plt.tight_layout()
plt.savefig('nn_tf_true_vs_pred.png', dpi=300)
plt.show()

#=============================
# 9) SHAP Analysis
#=============================
explainer = shap.DeepExplainer(final_model, X_train_scaled[:100])
shap_values = explainer.shap_values(X_test_scaled)

plt.figure()
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test_scaled, columns=features),
    plot_type='bar',
    show=False
)
plt.tight_layout()
plt.savefig('nn_tf_shap_summary_bar.png', dpi=300)
plt.close()

plt.figure()
shap.summary_plot(
    shap_values,
    pd.DataFrame(X_test_scaled, columns=features),
    show=False
)
plt.tight_layout()
plt.savefig('nn_tf_shap_beeswarm.png', dpi=300)
plt.show()
