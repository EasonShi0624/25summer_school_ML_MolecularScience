# ET Hyperopt
# 定义超参数空间
parameter_space_et = {
    'n_estimators': hp.choice('n_estimators', range(50,300,50)),  # n_estimators 50-300
    'max_depth': hp.choice('max_depth', range(3, 20,2)),  # max_depth 3-20
    'min_samples_split': hp.uniform('min_samples_split', 0.01,1.0),  # min_samples_split 2-15
    'min_samples_leaf' : hp.uniform('min_samples_leaf', 0.01,1.0), #min_samples_leaf1-10
    'max_features' : hp.uniform('max_features',0.1,1.0),#特征采样比例
    'bootstrap' : hp.choice('bootstrap',[True,False])
}

# 定义目标函数
def et_objective(params):

    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 ET 模型
    model = ExtraTreesRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        bootstrap=params['bootstrap'],
        n_jobs=-1
        random_state=random_seed
    )
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_et = Trials()
best_params_et = fmin(
    fn=et_objective,
    space=parameter_space_et,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_et,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'n_estimators': range(50, 300, 50)[best_params_et['n_estimators']],
    'max_depth': range(3, 20, 2)[best_params_et['max_depth']],
    'min_samples_split': best_params_et['min_samples_split'],
    'min_samples_leaf': best_params_et['min_samples_leaf'],
    'max_features': best_params_et['max_features'],
    'bootstrap': [True, False][best_params_et['bootstrap']]
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 ET 模型
best_model_et = ExtraTreesRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=random_seed
)

best_model_et.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_et = './et.pkl'
joblib.dump(best_model_et, model_et)
print(f"Model saved to {model_et}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_et)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)