import streamlit as st
import pandas as pd
import numpy as np
from modelo_riesgo import entrenar_red_neuronal, predecir_individual

# 1. Configuración de la Interfaz
st.set_page_config(page_title="AI Risk Management", layout="wide")


# --- Generador de Datos Sintéticos (Para que la app funcione siempre) ---
def generar_datos_demo():
    np.random.seed(42)
    n = 1000
    data = {
        'ingresos': np.random.normal(3500, 1200, n),
        'edad': np.random.randint(18, 75, n),
        'deuda': np.random.normal(1200, 900, n),
        'score': np.random.randint(300, 850, n)
    }
    df = pd.DataFrame(data)
    # Lógica de riesgo: si la deuda es > 50% del ingreso o score es muy bajo
    df['default'] = ((df['deuda'] / df['ingresos'] > 0.45) | (df['score'] < 450)).astype(int)
    return df


# --- Carga de Modelo (Optimizado con Cache) ---
@st.cache_resource
def iniciar_modelo():
    # Intenta cargar datos reales, si no, usa la simulación estadística
    try:
        df = pd.read_csv("data/clientes.csv")
    except:
        df = generar_datos_demo()

    modelo, scaler = entrenar_red_neuronal(df)
    return modelo, scaler


# Inicializamos el modelo una sola vez
mlp, scaler = iniciar_modelo()

# 2. Diseño del Dashboard
st.title("🛡️ Inteligencia de Riesgo: Redes Neuronales")
st.info("Modelo de Perceptrón Multicapa para la predicción de insolvencia financiera.")

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Perfil del Solicitante")
    ing = st.number_input("Ingresos Mensuales (S/)", value=4000)
    ed = st.slider("Edad del Cliente", 18, 80, 30)
    deu = st.number_input("Deuda Total Existente (S/)", value=500)
    sco = st.slider("Credit Score (Experian/Buro)", 300, 850, 720)

    boton_analizar = st.button("Ejecutar Modelo IA")

with col_der:
    if boton_analizar:
        # Ejecutamos la predicción con el archivo importado
        datos_usuario = [ing, ed, deu, sco]
        prob = predecir_individual(mlp, scaler, datos_usuario)

        # Semáforo de riesgo
        color = "green" if prob < 0.2 else "orange" if prob < 0.5 else "red"
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{prob * 100:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Probabilidad Estimada de Default</p>", unsafe_allow_html=True)

        st.progress(prob)

        if prob > 0.5:
            st.error("ALTO RIESGO: El modelo sugiere rechazar la solicitud.")
        elif prob > 0.2:
            st.warning("RIESGO MODERADO: Se recomienda análisis de comité.")
        else:
            st.success("APROBADO: El perfil cumple con los estándares de seguridad.")

# 3. Pie de página profesional
st.divider()
with st.expander("Detalles Técnicos de la Implementación"):
    st.write("""
    Este dashboard no utiliza una regresión lineal simple. Se ha implementado una **Red Neuronal Artificial (ANN)** con arquitectura de dos capas ocultas. El modelo utiliza el algoritmo de optimización **Adam** para ajustar 
    los pesos basándose en la entropía cruzada, permitiendo detectar patrones de comportamiento que los 
    métodos tradicionales ignoran.
    """)
