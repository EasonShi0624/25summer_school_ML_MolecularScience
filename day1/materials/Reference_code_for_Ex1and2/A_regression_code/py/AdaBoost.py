# ABR Hyperopt
# 定义超参数空间
parameter_space_abr = {
    'n_estimators': hp.choice('n_estimators', range(50, 500)),  # 弱学习器的数量
    'learning_rate': hp.loguniform('learning_rate', -5, 0),     # 学习率 (10^-5 到 10^0)
    'estimator': {
        'max_depth': hp.choice('max_depth', range(1, 10)),      # 决策树的最大深度
        'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),  # 最小样本分割比例
    }
}

# 定义目标函数
def abr_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 初始化模型
    # 创建基学习器（决策树）
    estimator = DecisionTreeRegressor(
        max_depth=params['estimator']['max_depth'],
        min_samples_split=params['estimator']['min_samples_split'],
        random_state=random_seed
    )
    # 创建 AdaBoostRegressor 模型
    model = AdaBoostRegressor(
        estimator=estimator,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_abr = Trials()
best_params_abr = fmin(
    fn=abr_objective,
    space=parameter_space_abr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_abr,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'n_estimators' : best_params_abr['n_estimators'],
    'learning_rate' : best_params_abr['learning_rate'],
    'estimator' : {
        'max_depth' : best_params_abr['max_depth'],
        'min_samples_split' : best_params_abr['min_samples_split'],
    }
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建基学习器（决策树）
best_estimator = DecisionTreeRegressor(
    max_depth=best_params['estimator']['max_depth'],
    min_samples_split=best_params['estimator']['min_samples_split'],
    random_state=random_seed
)
# 创建 AdaBoostRegressor 模型
best_model_abr = AdaBoostRegressor(
    estimator=best_estimator,
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    random_state=random_seed
)

best_model_abr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_abr = './abr.pkl'
joblib.dump(best_model_abr, model_abr)
print(f"Model saved to {model_abr}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_abr)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)
