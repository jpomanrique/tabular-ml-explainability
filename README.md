# 🧮 Tabular ML + Explainability
Este projeto implementa um pipeline completo de Machine Learning em dados tabulares, com foco em:
- Pré-processamento automatizado
- Treinamento e comparação de modelos
- Interpretabilidade com SHAP e LIME
- Dashboard interativo (Streamlit)

## 📊 Dataset
Utilizamos o [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit), um dataset de crédito com variáveis tabulares.

## 🚀 Pipeline
1. Pré-processamento (imputação, encoding, normalização)
2. Treinamento (Logistic Regression, Random Forest, XGBoost, LightGBM)
3. Avaliação (AUC, F1, ROC, Precision-Recall)
4. Interpretabilidade (SHAP, LIME)
5. Dashboard para visualização das previsões

## 🛠️ Stack
- Python 3.10+
- Scikit-learn
- XGBoost / LightGBM
- SHAP / LIME
- Pandas / Numpy
- Streamlit (para visualização)

## 📂 Estrutura
tabular-ml-explainability/
│── data/                # dados brutos e processados
│── notebooks/           # Jupyter notebooks exploratórios
│── src/                 # código modular
│── models/              # modelos treinados
│── reports/             # gráficos e dashboards
│── main.py              # pipeline principal
│── requirements.txt
│── README.md

## ▶️ Como rodar
```bash
python main.py
```

## 📌 Resultados esperados
- AUC ROC > 0.75
- Dashboard com insights de variáveis mais importantes
- Comparação entre modelos clássicos e gradient boosting
