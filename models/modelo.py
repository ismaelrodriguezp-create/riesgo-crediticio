import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def entrenar_red_neuronal(df):
    """
    Entrena un Perceptrón Multicapa (MLP).
    Las redes neuronales superan a la regresión logística al capturar
    interacciones no lineales entre las variables financieras.
    """
    # Definimos las variables de entrada y el objetivo
    X = df[['ingresos', 'edad', 'deuda', 'score']]
    y = df['default']

    # Escalado de datos: Crucial para que la Red Neuronal converja correctamente
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Arquitectura: 12 neuronas en capa 1, 8 en capa 2.
    # Usamos activación ReLU para flexibilidad estadística.
    mlp = MLPClassifier(
        hidden_layer_sizes=(12, 8),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    mlp.fit(X_scaled, y)
    return mlp, scaler


def predecir_individual(modelo, scaler, datos_cliente):
    """
    Recibe una lista de datos y devuelve la probabilidad de impago.
    """
    df_cliente = pd.DataFrame([datos_cliente], columns=['ingresos', 'edad', 'deuda', 'score'])
    datos_escalados = scaler.transform(df_cliente)

    # Retorna la probabilidad del evento 1 (Default)
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    return probabilidad