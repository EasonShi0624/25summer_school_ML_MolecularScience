# KNR Hyperopt
# 定义超参数空间
parameter_space_knr = {
    'n_neighbors': hp.quniform('n_neighbors', 1, 20,1),  #n_neighbors 从1 到 20
    'weights': hp.choice('weights',['uniform','distance']),  # 权重类型
    'p': hp.quniform('p', 1,2,1), # 距离度量参数（1: 曼哈顿距离，2: 欧氏距离）
    'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski'])  # 距离度量方法
}

# 定义目标函数
def knr_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 KNR 模型
    model = KNeighborsRegressor(
        n_neighbors=int(params['n_neighbors']), 
        weights=params['weights'], 
        p=int(params['p']),
        metric=params['metric'],
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_knr = Trials()
best_params_knr = fmin(
    fn=knr_objective,
    space=parameter_space_knr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_knr,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'n_neighbors' : int(best_params_knr['n_neighbors']),
    'weights' : ['uniform','distance'][best_params_knr['weights']],
    'p' : best_params_knr['p'],
    'metric' : ['euclidean', 'manhattan', 'minkowski'][best_params_knr['metric']]
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 KNR 模型
best_model_knr = KNeighborsRegressor(
    n_neighbors=best_params['n_neighbors'],
    weights=best_params['weights'],
    p=int(best_params['p']),
    metric=best_params['metric'],
    random_state=random_seed
)

best_model_knr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_knr = './knr.pkl'
joblib.dump(best_model_knr, model_knr)
print(f"Model saved to {model_knr}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_knr)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)