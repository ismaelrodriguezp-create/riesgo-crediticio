import streamlit as st

def mostrar_simulador(modelo, scaler):
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:24px;
                border:1px solid #E2E8F0; box-shadow:0 2px 12px rgba(37,99,235,0.07);">
        <p style="margin:0 0 20px; font-size:15px; font-weight:600; color:#0F2044;">
            🎯 Simulador de Riesgo Crediticio
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        edad      = st.slider("Edad del cliente", 18, 70, 30)
        ingresos  = st.number_input("Ingresos mensuales (S/)", 500, 50000, 3000, step=500)
        monto     = st.number_input("Monto del préstamo (S/)", 1000, 100000, 10000, step=1000)
    with col2:
        deuda     = st.number_input("Deuda actual (S/)", 0, 50000, 2000, step=500)
        historial = st.selectbox("Historial de pagos", [0, 1], format_func=lambda x: "✅ Sin atrasos" if x == 0 else "⚠️ Con atrasos")
        meses     = st.slider("Meses de empleo", 0, 300, 24)

    from models.modelo import predecir
    prob = predecir(modelo, scaler, edad, ingresos, monto, deuda, historial, meses)

    nivel = "ALTO" if prob > 0.6 else "MEDIO" if prob > 0.35 else "BAJO"
    color = "#EF4444" if prob > 0.6 else "#F59E0B" if prob > 0.35 else "#10B981"
    icono = "🔴" if prob > 0.6 else "🟡" if prob > 0.35 else "🟢"

    st.markdown(f"""
    <div style="margin-top:24px; background:{color}15; border:2px solid {color};
                border-radius:16px; padding:20px 24px; text-align:center;">
        <p style="margin:0; font-size:14px; color:#64748B;">Probabilidad de impago</p>
        <p style="margin:4px 0; font-size:48px; font-weight:700; color:{color};">{prob*100:.1f}%</p>
        <p style="margin:0; font-size:18px; font-weight:600; color:{color};">{icono} Riesgo {nivel}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    recomendacion = (
        "❌ **No recomendado:** Alto riesgo de impago. Considere rechazar o pedir garantías adicionales."
        if prob > 0.6 else
        "⚠️ **Precaución:** Riesgo moderado. Evalúe condiciones más estrictas o monto menor."
        if prob > 0.35 else
        "✅ **Aprobado:** Bajo riesgo de impago. Cliente con buen perfil crediticio."
    )
    st.markdown(f"<br><p style='font-size:14px;'>{recomendacion}</p>", unsafe_allow_html=True)
