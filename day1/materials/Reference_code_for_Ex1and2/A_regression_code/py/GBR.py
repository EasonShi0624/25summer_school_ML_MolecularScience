# GBR Hyperopt
# 定义超参数搜索空间

parameter_space_gbr = {
    'n_estimators': hp.quniform('n_estimators', 30, 80, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-1)),
    'max_depth': hp.quniform('max_depth', 3, 7, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 5, 1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2', 2, 3, 4]),
    'subsample': hp.uniform('subsample', 0.7, 1)   
}
def gbr_objective(params):
    # 处理 max_depth 中的 None 值
    #params['max_depth'] = None if params['max_depth'] == 0 else params['max_depth']
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    # 打印当前参数
    print("Parameters: ", params)
    model = GradientBoostingRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        max_features=params['max_features'],
        subsample=params['subsample'],
        random_state=random_seed
    )
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}
    
# 运行超参数搜索
trials_gbr = Trials()
best_params_gbr = fmin(
    fn=gbr_objective,
    space=parameter_space_gbr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_gbr,
    rstate=np.random.default_rng(random_seed)
)
# 打印最佳参数
print("Best parameters: ", best_params_gbr)
# 处理返回的索引值
best_params_gbr['n_estimators'] = int(best_params_gbr['n_estimators'])
best_params_gbr['max_depth'] = int(best_params_gbr['max_depth'])
best_params_gbr['max_features'] = ['sqrt', 'log2', 2, 3, 4][best_params_gbr['max_features']]
# 打印处理后的最佳参数
print("Best parameters: ", best_params_gbr)
# 使用最佳参数训练模型
best_model_gbr = GradientBoostingRegressor(
    n_estimators=best_params_gbr['n_estimators'],
    learning_rate=best_params_gbr['learning_rate'],
    max_depth=best_params_gbr['max_depth'],
    min_samples_split=int(best_params_gbr['min_samples_split']),
    min_samples_leaf=int(best_params_gbr['min_samples_leaf']),
    max_features=best_params_gbr['max_features'],
    subsample=best_params_gbr['subsample'],
    random_state=random_seed
)
best_model_gbr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_gbr = './out/GBR.pkl'
joblib.dump(best_model_gbr, model_gbr)
# print(f"Model saved to {model_gbr}")
# # 查看加载模型的参数
# loaded_model = joblib.load(model_gbr)
# print("Loaded model parameters: ", loaded_model.get_params())
# # 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)