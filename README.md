# 🧮 Tabular ML + Explainability

## Descrição do Projeto
Este projeto implementa um pipeline completo de **Machine Learning para dados tabulares**, com foco em:

- Pré-processamento automatizado de dados
- Treinamento e comparação de modelos (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Avaliação de performance (AUC ROC, F1-score, classification report)
- Interpretabilidade de modelos com **SHAP** e **LIME**
- Dashboard interativo em **Streamlit** para visualização de insights

O objetivo é demonstrar **habilidades avançadas de Data Science e ML** aplicadas a problemas do mundo real, usando dados de crédito.

---

## Tecnologias e Bibliotecas
- **Python 3.10+**
- Pandas, NumPy
- Scikit-learn
- XGBoost / LightGBM
- SHAP, LIME
- Matplotlib, Seaborn
- Streamlit
- Jupyter Notebook

---

## Estrutura do Repositório
```
tabular-ml-explainability/
│── data/                # dados brutos e processados (dataset credit.csv)
│── notebooks/           # notebooks exploratórios e baseline
│   └── 01_eda_baseline.ipynb
│── src/                 # código modular
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── explainability.py
│── models/              # modelos treinados (.pkl)
│── reports/             # gráficos, imagens e dashboards
│── main.py              # pipeline principal
│── requirements.txt     # dependências do projeto
│── README.md            # este arquivo de documentação
```
---

## Como Rodar

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/SEU_USUARIO/tabular-ml-explainability.git
cd tabular-ml-explainability
```

### 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Rodar pipeline principal
```bash
python main.py
```
- Executa todo o pipeline: pré-processamento, treinamento, avaliação e interpretabilidade.

### 4️⃣ Abrir notebook exploratório
```bash
jupyter notebook notebooks/01_eda_baseline.ipynb
```
- Explore dados, visualize gráficos e rode o modelo baseline.

### 5️⃣ Rodar dashboard interativo
```bash
streamlit run src/dashboard.py
```
> O dashboard permite visualizar previsões e variáveis importantes com gráficos interativos.

---

## Dataset
Usamos o dataset [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit), que contém dados tabulares de crédito.

---

## Resultados Esperados
- AUC ROC > 0.75 nos modelos principais  
- Identificação das variáveis mais importantes com SHAP  
- Dashboard interativo com insights de negócio  

---

## Sobre o Autor
**Seu Nome**  
- LinkedIn: [https://linkedin.com/in/JPOManrique](https://linkedin.com/in/JPOManrique)  
- GitHub: [https://github.com/JPOManrique](https://github.com/JPOManrique)  

Portfólio focado em **Machine Learning, Data Science e Inteligência Artificial**.