# KR Hyperopt
# 定义超参数空间
parameter_space_kr = {
    'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(10)), # 正则化强度（对数均匀分布）
    'kernel': hp.choice('kernel', ['rbf','poly']),   # 核函数类型
    'gamma': hp.loguniform('gamma', np.log(1e-3), np.log(10)),  #  RBF/多项式核的带宽
    'degree' : hp.choice('degree',[2,3,4]), # 多项式核的阶数（仅当kernel='poly'时生效）
    'coef0' : hp.uniform('coef0',0,1)# 多项式核的偏置项
}

# 定义目标函数
def kr_objective(params):
    # 打印当前参数
    print("Current parameters: ", params)
    
    # 创建 KR 模型
    model = KernelRidge(
        alpha=params['alpha'], 
        kernel=params['kernel'], 
        gamma=params['gamma'],
        degree=params['degree'] if params['kernel'] == 'poly' else 3,
        coef0=params['coef0'] if params['kernel'] == 'poly' else 1.0
        random_state=random_seed
    ) 
    
    # 使用交叉验证评估模型性能
    mse_scores = cross_val_score(model, X_train_CoRE_scaled, y_train_CoRE, cv=10, scoring='neg_mean_squared_error')
    mse = -np.mean(mse_scores)  # 转换为正数的 MSE
    # 返回 MSE，Hyperopt 会最小化该目标值
    return {'loss': mse, 'status': STATUS_OK}

# 运行超参数搜索
trials_kr = Trials()
best_params_kr = fmin(
    fn=kr_objective,
    space=parameter_space_kr,
    algo=tpe.suggest,
    max_evals=200,
    trials=trials_kr,
    rstate=np.random.default_rng(random_seed)
)

# 解码最佳超参数
best_params = {
    'alpha' : best_params_kr['alpha'],
    'kernel' : ['rbf','poly'][best_params_kr['kernel']],
    'gamma' : best_params_kr['gamma'],
    'degree' : best_params_kr['degree'] if best_params_kr['kernel'] == 1 else 3,
    'coef0': best_params_kr['coef0'] if best_params_kr['kernel'] == 1 else 1.0 
}

# 打印处理后的最佳参数
print("Best parameters: ", best_params)

# 使用最佳参数训练模型
# 创建 KR 模型
best_model_kr = KernelRidge(
    alpha=best_params['alpha'],
    kernel=best_params['kernel'],
    gamma=best_params['gamma'],
    degree=best_params['degree'] if best_params['kernel'] == 1 else 3,
    coef0=best_params['coef0'] if best_params['kernel'] == 1 else 1.0
    random_state=random_seed
)

best_model_kr.fit(X_train_CoRE_scaled, y_train_CoRE)

# 保存最优模型
model_kr = './kr.pkl'
joblib.dump(best_model_kr, model_kr)
print(f"Model saved to {model_kr}")
# 查看加载模型的参数
#loaded_model = joblib.load(model_kr)
#print("Loaded model parameters: ", loaded_model.get_params())
# 如果需要加载模型并进行预测
# loaded_model = joblib.load(model_filename)
# y_pred = loaded_model.predict(X_test_CoRE_scaled)
# print("Predictions: ", y_pred)