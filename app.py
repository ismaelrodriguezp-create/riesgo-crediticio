import streamlit as st
import pandas as pd
from models.modelo import entrenar_modelo
from components.kpis import mostrar_kpis
from components.scoring import mostrar_simulador
from components.graficos import grafico_distribucion, grafico_mora_edad, grafico_montecarlo

st.set_page_config(page_title="Riesgo Crediticio", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F2044 0%, #1B3A6B 100%); }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background: white; border-radius: 12px;
        padding: 4px; border: 1px solid #E2E8F0; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 20px;
        font-weight: 500; color: #64748B !important;
    }
    .stTabs [aria-selected="true"] { background: #2563EB !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
    <div style="display:flex; align-items:center; gap:14px;">
        <div style="background:linear-gradient(135deg,#1D4ED8,#0EA5E9); border-radius:14px;
                    padding:10px 14px; box-shadow:0 4px 14px rgba(37,99,235,0.3);">
            <span style="font-size:24px;">🏦</span>
        </div>
        <div>
            <h1 style="margin:0; font-size:26px; font-weight:700;">Dashboard de Riesgo Crediticio</h1>
            <p style="margin:0; color:#64748B; font-size:13px;">Credit Scoring · VaR · Simulación Monte Carlo</p>
        </div>
    </div>
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px; padding:8px 16px;">
        <p style="margin:0; color:#1D4ED8; font-size:12px; font-weight:600;">● MODELO ACTIVO</p>
    </div>
</div>
<hr style="border:none; border-top:1px solid #E2E8F0; margin:16px 0 20px;">
""", unsafe_allow_html=True)

modelo, scaler, acc, auc = entrenar_modelo()

with st.sidebar:
    st.markdown("<p style='font-size:11px; color:#94A3B8; font-weight:600; letter-spacing:0.08em;'>MODELO</p>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#0F2044; border-radius:12px; padding:16px; border:1px solid #1E3A5F;">
        <p style="margin:0 0 8px; font-size:12px; color:#94A3B8;">Regresión Logística</p>
        <p style="margin:0 0 4px; font-size:13px; color:#E2E8F0;">Accuracy: <b>{acc*100:.1f}%</b></p>
        <p style="margin:0; font-size:13px; color:#E2E8F0;">AUC-ROC: <b>{auc:.3f}</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E3A5F; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px; color:#94A3B8; font-weight:600; letter-spacing:0.08em;'>DATOS</p>", unsafe_allow_html=True)
    archivo = st.file_uploader("Subir cartera de clientes", type=["csv","xlsx"])

df = pd.read_csv(archivo) if archivo else pd.read_csv("data/clientes.csv")

mostrar_kpis(df)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Resumen Ejecutivo", "🎯 Calculadora de Riesgo", "📈 Simulación Monte Carlo"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribución de ingresos")
        grafico_distribucion(df)
    with col2:
        st.markdown("#### Mora por rango de edad")
        grafico_mora_edad(df)

with tab2:
    mostrar_simulador(modelo, scaler)

with tab3:
    fig, var_95, perdida_max = grafico_montecarlo()
    col1, col2, col3 = st.columns(3)
    col1.metric("💼 Portafolio inicial", "S/ 100,000")
    col2.metric("📉 VaR al 95%", f"S/ {var_95:,.0f}")
    col3.metric("🚨 Pérdida máxima esperada", f"S/ {perdida_max:,.0f}", delta=f"-{perdida_max/1000:.1f}k", delta_color="inverse")
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 El VaR al 95% indica que en el peor 5% de escenarios, el portafolio perdería más de esta cantidad en un año.")

st.markdown("""
<hr style="border:none; border-top:1px solid #E2E8F0; margin-top:2rem;">
<p style="text-align:center; color:#94A3B8; font-size:12px;">
    Dashboard de Riesgo Crediticio · Python & Streamlit · Desarrollado por Ismael Rodriguez
</p>
""", unsafe_allow_html=True)
