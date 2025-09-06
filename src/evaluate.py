from sklearn.metrics import roc_auc_score, f1_score, classification_report

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        f1 = f1_score(y_test, y_pred)
        print(f"\nModelo: {name}")
        print(f"AUC: {auc:.3f} | F1: {f1:.3f}")
        print(classification_report(y_test, y_pred))
