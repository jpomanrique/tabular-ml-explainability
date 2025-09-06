import shap
import matplotlib.pyplot as plt

def run_explainability(models, X_test):
    for name, model in models.items():
        print(f"\nInterpretando modelo: {name}")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(f"reports/{name}_shap_summary.png")
            plt.close()
        except:
            print(f"SHAP não compatível com {name}")
