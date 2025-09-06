from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def train_models(X_train, y_train):
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name} treinado com sucesso!")
    return models
