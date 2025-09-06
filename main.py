import pandas as pd
from src.data_preprocessing import load_and_preprocess
from src.train_model import train_models
from src.evaluate import evaluate_models
from src.explainability import run_explainability

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/credit.csv")
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    run_explainability(models, X_test)

if __name__ == "__main__":
    main()
