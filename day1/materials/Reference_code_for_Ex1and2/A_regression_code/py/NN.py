# NN Hyperopt
# 定义超参数空间

parameter_space_nn = {
    'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50)]),  
    'activation': hp.choice('activation', ['relu', 'tanh', 'logistic']),
    'solver': hp.choice('solver', ['adam', 'sgd']),
    'alpha': hp.loguniform('alpha', -7, 0),  # 10^-7 到 10^-2
    'learning_rate': hp.choice('learning_rate', ['constant', 'adaptive']),
    'batch_size': hp.choice('batch_size', [27, 54, 108]),    
}

# 定义目标函数
def nn_objective(params):
    # 处理 activation 参数
    if isinstance(params['activation'], int):  # 如果是整数，映射到字符串
        activation_map = ['relu', 'tanh', 'logistic']
        params['activation'] = activation_map[params['activation']]
    # 打印当前参数
    print("Current parameters: ", params)
    
    # 初始化模型
    model = MLPRegressor(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation=params['activation'],
        solver=params['solver'],
        alpha=params['alpha'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'],
        max_iter=200,
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_nn = Trials()
best_params_nn = fmin(
    fn=nn_objective,
    space=parameter_space_nn,
    algo=tpe.suggest,
    max_evals=evals,
    trials=trials_nn,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'hidden_layer_sizes': [(70,), (100,), (70, 70), (100, 70)][best_params_nn['hidden_layer_sizes']],
    'activation': ['relu', 'tanh', 'logistic'][best_params_nn['activation']],
    'solver': ['adam', 'sgd'][best_params_nn['solver']],
    'alpha': best_params_nn['alpha'],
    'learning_rate': ['constant', 'adaptive'][best_params_nn['learning_rate']],
    'batch_size': [27, 54, 108][best_params_nn['batch_size']],
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
best_model_nn = MLPRegressor(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    alpha=best_params['alpha'],
    learning_rate=best_params['learning_rate'],
    batch_size=best_params['batch_size'],
    max_iter=200,
    random_state=random_seed
)
best_model_nn.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_nn = './out/nn.pkl'
joblib.dump(best_model_nn, model_nn)
print(f"Model saved to {model_nn}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_nn)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)