import streamlit as st
import pandas as pd

def mostrar_kpis(df):
    total = len(df)
    mora = df["default"].sum()
    tasa_mora = (mora / total * 100)
    ingresos_prom = df["ingresos"].mean()
    monto_prom = df["monto_prestamo"].mean()

    col1, col2, col3, col4 = st.columns(4)

    def card(col, icono, titulo, valor, color="#0F2044"):
        col.markdown(f"""
        <div style="background:white; border-radius:16px; padding:18px 22px;
                    border:1px solid #E2E8F0; box-shadow:0 2px 12px rgba(37,99,235,0.07);">
            <p style="margin:0 0 4px; font-size:11px; color:#64748B; font-weight:600;
                      letter-spacing:0.06em; text-transform:uppercase;">{icono} {titulo}</p>
            <p style="margin:0; font-size:24px; font-weight:700; color:{color};">{valor}</p>
        </div>
        """, unsafe_allow_html=True)

    card(col1, "👥", "Total clientes",     f"{total}")
    card(col2, "🚨", "En mora",            f"{int(mora)}", "#EF4444")
    card(col3, "📊", "Tasa de mora",       f"{tasa_mora:.1f}%", "#EF4444" if tasa_mora > 30 else "#10B981")
    card(col4, "💰", "Ingreso promedio",   f"S/ {ingresos_prom:,.0f}")
