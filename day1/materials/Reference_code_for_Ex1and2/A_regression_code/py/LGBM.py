# LGBM Hyperopt
# 定义超参数空间
parameter_space_lgbm = {
    'num_leaves': hp.choice('num_leaves', range(20, 150, 10)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'max_depth': hp.choice('max_depth', range(3, 12)),
    'min_child_samples': hp.choice('min_child_samples', range(5, 50, 5)),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(10)),
}

# 定义目标函数
def lgbm_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 LGBM 模型
    model = LGBMRegressor(
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_samples=params['min_child_samples'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        num_threads=1,
        verbose=-1
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_lgbm = Trials()
best_params_lgbm = fmin(
    fn=lgbm_objective,
    space=parameter_space_lgbm,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_lgbm,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
   'num_leaves': range(20, 150, 10)[best_params_lgbm['num_leaves']],
    'learning_rate': best_params_lgbm['learning_rate'],
    'max_depth': range(3, 12)[best_params_lgbm['max_depth']],
    'min_child_samples': range(5, 50, 5)[best_params_lgbm['min_child_samples']],
    'subsample': best_params_lgbm['subsample'],
    'colsample_bytree': best_params_lgbm['colsample_bytree'],
    'reg_alpha': best_params_lgbm['reg_alpha'],
    'reg_lambda': best_params_lgbm['reg_lambda']
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 LGBM 模型
best_model_lgbm = LGBMRegressor(
    num_leaves=best_params['num_leaves'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_child_samples=best_params['min_child_samples'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    num_threads=1
    random_state=random_seed
)

best_model_lgbm.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_lgbm = './lgbm.pkl'
joblib.dump(best_model_lgbm, model_lgbm)
print(f"Model saved to {model_lgbm}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_lgbm)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)