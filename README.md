# 🌊 Predicción de Probabilidad de Inundaciones

Modelo de **regresión** que estima la probabilidad de inundación de una región a partir de 20 variables climáticas, geográficas y de gestión (intensidad del monzón, deforestación, urbanización, sistemas de drenaje, vulnerabilidad costera, etc.).

## 📊 Dataset

[Flood Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset) — target continuo `FloodProbability`. Descargar `flood.csv` y colocarlo junto al script.

## 🔬 Metodología

- **EDA:** distribución del target, matriz de correlación completa y ranking de correlación de cada variable con `FloodProbability`.
- **Dos escenarios de variables:** (A) todas las predictoras vs. (B) un subconjunto de 10 variables climáticas y geográficas — para medir cuánta capacidad predictiva se conserva con un modelo más simple.
- **Preprocesamiento:** recorte de outliers por rango intercuartílico (IQR) y estandarización para los modelos lineales.
- **Modelado:** **Regresión Lineal**, **Ridge** (con `alpha` optimizado por `GridSearchCV`) y **Random Forest**, evaluados con RMSE, MAE y R².
- **Diagnóstico del mejor modelo:** curva de aprendizaje, gráfico de predicho vs. real y distribución de residuos.
- **Persistencia:** el mejor modelo se guarda con `joblib` y la comparación de escenarios se exporta a CSV.

## ▶️ Ejecución

```bash
pip install -r requirements.txt
python prediccion_inundaciones.py
```

Script con separadores `# %%`, ejecutable por celdas en Spyder, VS Code o Jupyter.

## 🛠️ Stack

Python · pandas · NumPy · scikit-learn · joblib · matplotlib · seaborn

---

👤 **Angel David Ortega Jaimes** — [LinkedIn](https://www.linkedin.com/in/angel-ortega-3899a52b4/) · [GitHub](https://github.com/mageangel)
