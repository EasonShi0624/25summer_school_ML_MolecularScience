# SVR Hyperopt
# 定义超参数空间

parameter_space_svr = {
    'C': hp.loguniform('C', -5, 2),  # C 在 e^-5 到 e^2 之间
    'epsilon': hp.uniform('epsilon', 0, 1),  # epsilon 在 0 到 1 之间
    'gamma': hp.loguniform('gamma', -5, 2),  # gamma 在 e^-5 到 e^2 之间
}

# 定义目标函数
def svr_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 SVR 模型
    model = SVR(
        C=params['C'], 
        epsilon=params['epsilon'], 
        gamma=params['gamma'],
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_svr = Trials()
best_params_svr = fmin(
    fn=svr_objective,
    space=parameter_space_svr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_svr,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'C' : best_params_svr['C'],
    'epsilon' : best_params_svr['epsilon'],
    'gamma' : best_params_svr['gamma']
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 SVR 模型
best_model_svr = SVR(
    C=best_params['C'],
    epsilon=best_params['epsilon'],
    gamma=best_params['gamma'],
    random_state=random_seed
)

best_model_svr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_svr = './svr.pkl'
joblib.dump(best_model_svr, model_svr)
print(f"Model saved to {model_svr}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_svr)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)