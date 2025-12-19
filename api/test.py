import joblib

# 1. 加载模型
model = joblib.load("models/global_best_model_optuna.pkl")

# 2. 看看 Pipeline 第一步 (ColumnTransformer) 里的配置
try:
    # 假设你的 pipeline 第一步通常是预处理
    preprocessor = model.named_steps['preprocessing'] # 或者是 model.steps[0][1]
    
    print("=== 数值特征 (Pipeline 认为的) ===")
    # 不同的 sklearn 写法可能不同，通常在 transformers_ 属性里
    for name, trans, cols in preprocessor.transformers_:
        print(f"步骤名称: {name}")
        print(f"涉及列名: {cols}")
        print("-" * 20)
except Exception as e:
    print(f"直接查看失败，尝试打印整个步骤: {e}")
    # 如果上面报错，直接打印这个看看
    print(model.steps[0])