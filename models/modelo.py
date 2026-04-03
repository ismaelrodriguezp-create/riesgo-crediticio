import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def entrenar_modelo():
    df = pd.read_csv("data/clientes.csv")
    X = df[["edad","ingresos","monto_prestamo","deuda_actual","historial_pagos","meses_empleo"]]
    y = df["default"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression(random_state=42)
    modelo.fit(X_train, y_train)

    acc = accuracy_score(y_test, modelo.predict(X_test))
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1])

    return modelo, scaler, acc, auc

def predecir(modelo, scaler, edad, ingresos, monto, deuda, historial, meses):
    X = pd.DataFrame([[edad, ingresos, monto, deuda, historial, meses]],
                     columns=["edad","ingresos","monto_prestamo","deuda_actual","historial_pagos","meses_empleo"])
    X_scaled = scaler.transform(X)
    prob = modelo.predict_proba(X_scaled)[0][1]
    return prob
