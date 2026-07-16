# -*- coding: utf-8 -*-
"""
Predicción de inundaciones a partir de variables climáticas y geográficas.
Dataset: Flood Prediction Dataset (Kaggle) -> flood.csv

Se comparan dos escenarios de variables (todas las columnas vs. un
subconjunto) usando Regresión Lineal, Ridge y Random Forest.
"""

# %% 1) Librerías y configuración --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     KFold, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# El CSV debe estar en la misma carpeta que este script.
# Descarga: https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset
RUTA_DATOS = "flood.csv"
RANDOM_STATE = 42
TARGET = "FloodProbability"

df = pd.read_csv(RUTA_DATOS)
print(f"Dimensiones del dataset: {df.shape}")


# %% 2) Análisis exploratorio -------------------------------------------------

# Distribución de la variable objetivo
plt.figure()
sns.histplot(df[TARGET], kde=True, bins=40, color="steelblue")
plt.title("Distribución de FloodProbability")
plt.xlabel("Probabilidad de inundación")
plt.tight_layout()
plt.show()

# Matriz de correlación entre todas las variables
matriz_corr = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(matriz_corr, cmap="coolwarm", center=0)
plt.title("Matriz de correlación")
plt.tight_layout()
plt.show()

# Correlación de cada variable con el target
corr_target = matriz_corr[TARGET].drop(TARGET).sort_values(ascending=False)
plt.figure()
corr_target.plot(kind="barh", color="darkorange")
plt.title("Correlación de cada variable con FloodProbability")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
print("\nCorrelación con el target:\n", corr_target)


# %% 3) Escenarios de variables ------------------------------------------------

# Escenario A: todas las variables predictoras
todas_las_variables = df.columns.drop(TARGET).tolist()

# Escenario B: subconjunto de variables climáticas y geográficas
variables_subconjunto = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement",
    "Deforestation", "Urbanization", "ClimateChange", "Siltation",
    "DrainageSystems", "CoastalVulnerability", "Watersheds",
]

print(f"\nEscenario A: {len(todas_las_variables)} variables")
print(f"Escenario B: {len(variables_subconjunto)} variables")


# %% 4) Función de entrenamiento y evaluación ----------------------------------

def entrenar_y_evaluar(columnas, etiqueta):
    """Entrena Regresión Lineal, Ridge y Random Forest con las columnas
    indicadas y devuelve la tabla de métricas y los objetos necesarios
    para las gráficas de diagnóstico."""
    X = df[columnas].copy()
    y = df[TARGET].copy()

    # Tratamiento de valores atípicos por recorte con el rango intercuartílico
    for col in columnas:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Escalado para los modelos lineales
    scaler = StandardScaler()
    X_train_esc = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=columnas, index=X_train.index)
    X_test_esc = pd.DataFrame(scaler.transform(X_test),
                              columns=columnas, index=X_test.index)

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    modelos = {}

    # Regresión Lineal
    lin = LinearRegression()
    lin.fit(X_train_esc, y_train)
    modelos["Regresión Lineal"] = lin

    # Ridge con búsqueda del hiperparámetro alpha por validación cruzada
    ridge_gs = GridSearchCV(Ridge(random_state=RANDOM_STATE),
                            {"alpha": [0.1, 1, 10]}, cv=kf,
                            scoring="neg_mean_squared_error", n_jobs=-1)
    ridge_gs.fit(X_train_esc, y_train)
    modelos["Ridge"] = ridge_gs.best_estimator_

    # Random Forest con hiperparámetros fijos
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    modelos["Random Forest"] = rf

    # Evaluación en el conjunto de prueba
    resultados = []
    for nombre, modelo in modelos.items():
        usa_escalado = nombre in ("Regresión Lineal", "Ridge")
        Xte = X_test_esc if usa_escalado else X_test
        pred = modelo.predict(Xte)
        resultados.append({
            "Escenario": etiqueta,
            "Modelo": nombre,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })

    tabla = pd.DataFrame(resultados).sort_values("RMSE")
    datos = (X_train, X_train_esc, X_test, X_test_esc, y_train, y_test)
    return tabla, modelos, datos


# %% 5) Entrenamiento de ambos escenarios --------------------------------------

tabla_A, modelos_A, _ = entrenar_y_evaluar(todas_las_variables,
                                           "A - Todas las variables")
tabla_B, modelos_B, datos_B = entrenar_y_evaluar(variables_subconjunto,
                                                 "B - Subconjunto de variables")
Xtr_B, Xtr_B_esc, Xte_B, Xte_B_esc, ytr_B, yte_B = datos_B

comparacion = pd.concat([tabla_A, tabla_B], ignore_index=True)
print("\nComparación de escenarios:\n", comparacion)

# Comparación gráfica del R2 por modelo y escenario
plt.figure()
sns.barplot(data=comparacion, x="Modelo", y="R2", hue="Escenario")
plt.title("R2 por modelo: todas las variables vs. subconjunto")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# %% 6) Diagnóstico del mejor modelo (Escenario B) ------------------------------

mejor_nombre = tabla_B.iloc[0]["Modelo"]
mejor_modelo = modelos_B[mejor_nombre]
usa_escalado = mejor_nombre in ("Regresión Lineal", "Ridge")
X_train_curva = Xtr_B_esc if usa_escalado else Xtr_B
X_test_pred = Xte_B_esc if usa_escalado else Xte_B

# Curva de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    mejor_modelo, X_train_curva, ytr_B,
    cv=3, scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1, random_state=RANDOM_STATE
)
train_rmse = np.sqrt(-train_scores).mean(axis=1)
val_rmse = np.sqrt(-val_scores).mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_rmse, "o-", label="RMSE entrenamiento")
plt.plot(train_sizes, val_rmse, "o-", label="RMSE validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("RMSE")
plt.title(f"Curva de aprendizaje - {mejor_nombre} (Escenario B)")
plt.legend()
plt.tight_layout()
plt.show()

# Valores predichos frente a valores reales
y_pred = mejor_modelo.predict(X_test_pred)
plt.figure()
plt.scatter(yte_B, y_pred, alpha=0.3, color="teal")
lims = [yte_B.min(), yte_B.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("Valor real")
plt.ylabel("Valor predicho")
plt.title(f"Predicho vs. real - {mejor_nombre} (Escenario B)")
plt.tight_layout()
plt.show()

# Distribución de los residuos
residuos = yte_B - y_pred
plt.figure()
sns.histplot(residuos, kde=True, color="purple")
plt.title(f"Distribución de residuos - {mejor_nombre} (Escenario B)")
plt.tight_layout()
plt.show()


# %% 7) Guardado de resultados ---------------------------------------------------

joblib.dump(mejor_modelo, "modelo_final_escenario_B.pkl")
comparacion.to_csv("comparacion_escenarios.csv", index=False)

print(f"\nModelo final: {mejor_nombre} (Escenario B) "
      f"con R2 = {tabla_B.iloc[0]['R2']:.4f}")
print("Archivos guardados: modelo_final_escenario_B.pkl, comparacion_escenarios.csv")
