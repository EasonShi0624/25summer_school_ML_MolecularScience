# 随机森林 Hyperopt
# 定义超参数搜索空间

parameter_space_rf = {
    'n_estimators': hp.quniform('n_estimators', 30, 80, 1),  # 使用 quniform 生成整数范围
    'max_depth': hp.quniform('max_depth', 3, 7, 1), # max_depth_rf),  # 树的最大深度
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),  # 分裂所需最小样本数（整数）
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),  # 叶节点最小样本数（整数）
    'max_features': hp.choice('max_features', ['sqrt', 'log2', 2, 3, 4, 5]) # 控制每棵树在分裂时考虑的最大特征数量
}

# 定义目标函数
def rf_objective(params):
    # 处理 max_depth 中的 None 值
    # params['max_depth'] = None if params['max_depth'] == 0 else params['max_depth']
    # 打印当前参数
    print("Parameters: ", params)
    # 创建随机森林回归模型
    model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']), 
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        random_state=random_seed
    )

    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE

    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数优化
trials_rf = Trials()
best_params_rf = fmin(    
    fn=rf_objective,                # 优化的目标函数    
    space=parameter_space_rf,       # 搜索空间    
    algo=tpe.suggest,               # 贝叶斯优化算法    
    max_evals=200,                  # 最大评估次数
    trials=trials_rf,
    rstate=np.random.default_rng(random_seed)
)
print("Best parameters : ", best_params_rf)

# 处理 hp.choice 返回的索引值
best_params_rf['n_estimators'] = int(best_params_rf['n_estimators'])
best_params_rf['max_depth'] = int(best_params_rf['max_depth'])
best_params_rf['max_features'] = ['sqrt', 'log2', 2, 3, 4, 5][best_params_rf['max_features']]

# 使用最佳参数创建随机森林回归模型
best_model_rf = RandomForestRegressor(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    min_samples_split=int(best_params_rf['min_samples_split']),
    min_samples_leaf=int(best_params_rf['min_samples_leaf']),
    max_features=best_params_rf['max_features'],
    random_state=random_seed
)

# 在训练集上训练模型
best_model_rf.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_rf = './RF_2.pkl'
joblib.dump(best_model_rf, model_rf)
print(f"Model saved to {model_rf}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_rf)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)