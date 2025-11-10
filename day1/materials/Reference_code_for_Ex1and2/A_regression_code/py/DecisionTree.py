# DTR Hyperopt
# 定义超参数空间
parameter_space_dtr = {
    'max_depth': hp.choice('max_depth', range(1,20)),  # 决策树深度1到20
    'min_samples_split': hp.uniform('min_samples_split', 0.01,1.0),
    'min_samples_leaf': hp.uniform('min_samples_leaf', 0.01,1.0),
    'criterion': hp.choice('criterion', ['squared_error', 'friedman_mse']),  # 回归任务的划分标准
}

# 定义目标函数
def dtr_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 DTR 模型
    model = DecisionTreeRegressor(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        criterion=params['criterion'],
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_dtr = Trials()
best_params_dtr = fmin(
    fn=dtr_objective,
    space=parameter_space_dtr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_dtr,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'max_depth' : best_params_dtr['max_depth'],
    'min_samples_split' : best_params_dtr['min_samples_split'],
    'min_samples_leaf' : best_params_dtr['min_samples_leaf'],
    'criterion' : ['squared_error', 'friedman_mse'][best_params_dtr['criterion']]
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 DTR 模型
best_model_dtr = DecisionTreeRegressor(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    criterion=best_params['criterion']
    random_state=random_seed
)

best_model_dtr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_dtr = './dtr.pkl'
joblib.dump(best_model_dtr, model_dtr)
print(f"Model saved to {model_dtr}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_dtr)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)