# CATBOOST Hyperopt
# 定义超参数空间
parameter_space_catboost = {
    'iterations': hp.choice('iterations', range(100, 1001, 100)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
    'depth': hp.choice('depth', range(4, 11)),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0.1, 10),
    'border_count': hp.choice('border_count', [32, 64, 128, 254]),
}

# 定义目标函数
def catboost_objective(params):
    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 CATBOOST 模型
    model = CatBoostRegressor(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        border_count=params['border_count'],
        verbose=0,            # 关闭训练日志
        random_state=random_seed,
        thread_count=1       # 1:control runing thread to 1 -1:使用所有CPU核心
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_catboost = Trials()
best_params_catboost = fmin(
    fn=catboost_objective,
    space=parameter_space_catboost,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_catboost,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'iterations': range(100, 1001, 100)[best_params_catboost['iterations']],
    'learning_rate': best_params_catboost['learning_rate'],
    'depth': range(4, 11)[best_params_catboost['depth']],
    'l2_leaf_reg': best_params_catboost['l2_leaf_reg'],
    'border_count': [32, 64, 128, 254][best_params_catboost['border_count']],
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 CATBOOST 模型
best_model_catboost = CatBoostRegressor(
        iterations=best_params['iterations'],
        learning_rate=best_params['learning_rate'],
        depth=best_params['depth'],
        l2_leaf_reg=best_params['l2_leaf_reg'],
        border_count=best_params['border_count'],
        thread_count=1,
        verbose=0,
        random_state=random_seed
)

best_model_catboost.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_catboost = './catboost.pkl'
joblib.dump(best_model_catboost, model_catboost)
print(f"Model saved to {model_catboost}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_catboost)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)