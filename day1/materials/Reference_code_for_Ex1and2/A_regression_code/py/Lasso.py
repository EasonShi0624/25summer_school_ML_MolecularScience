# LASSO Hyperopt
# 定义超参数空间
parameter_space_lasso = {
    'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(10)),  # 
    'fit_intercept' : hp.choice('fit_intercept',[True,False]),  # 
    'selection': hp.choice('selection', ['cyclic','random']),  # g 坐标下降策略
}

# 定义目标函数
def lasso_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 LASSO 模型
    model = Lasso(
        alpha=params['alpha'],
        fit_intercept=params['fit_intercept'],
        selection=params['selection'],
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_lasso = Trials()
best_params_lasso = fmin(
    fn=lasso_objective,
    space=parameter_space_lasso,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_lasso,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'alpha' : best_params_lasso['alpha'],
    'fit_intercept' : [True,False][best_params_lasso['fit_intercept']],
    'selection' : ['cyclic','random'][best_params_lasso['selection']]
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 LASSO 模型
best_model_lasso = Lasso(
    alpha=best_params['alpha'],
    fit_intercept=best_params['fit_intercept'],
    selection=best_params['selection'],
    random_state=random_seed
)

best_model_lasso.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_lasso = './lasso.pkl'
joblib.dump(best_model_lasso, model_lasso)
print(f"Model saved to {model_lasso}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_lasso)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)