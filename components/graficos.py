import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def grafico_distribucion(df):
    fig = px.histogram(
        df, x="ingresos", color="default",
        color_discrete_map={0: "#2563EB", 1: "#EF4444"},
        labels={"default": "Mora", "ingresos": "Ingresos (S/)"},
        barmode="overlay", opacity=0.7,
        title="Distribución de ingresos por estado de mora"
    )
    fig.update_layout(
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="", orientation="h", y=-0.25),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.for_each_trace(lambda t: t.update(name="Sin mora" if t.name == "0" else "En mora"))
    st.plotly_chart(fig, use_container_width=True)

def grafico_mora_edad(df):
    df = df.copy()
    df["rango_edad"] = pd.cut(df["edad"],
                               bins=[18,25,35,45,55,70],
                               labels=["18-25","26-35","36-45","46-55","56-70"])
    resumen = df.groupby("rango_edad")["default"].mean().reset_index()
    resumen["tasa"] = resumen["default"] * 100

    fig = go.Figure(go.Bar(
        x=resumen["rango_edad"].astype(str),
        y=resumen["tasa"],
        marker=dict(
            color=resumen["tasa"],
            colorscale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
            line=dict(width=0)
        ),
        text=resumen["tasa"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))
    fig.update_layout(
        title="Tasa de mora por rango de edad",
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#F1F5F9", title="Tasa de mora (%)"),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def grafico_montecarlo():
    np.random.seed(42)
    retornos = np.random.normal(-0.001, 0.02, (1000, 252))
    portafolio = 100000 * np.cumprod(1 + retornos, axis=1)
    valores_finales = portafolio[:, -1]
    var_95 = np.percentile(valores_finales, 5)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valores_finales, nbinsx=50,
        marker=dict(color="#2563EB", opacity=0.7, line=dict(width=0)),
        name="Escenarios"
    ))
    fig.add_vline(
        x=var_95, line_dash="dash", line_color="#EF4444", line_width=2,
        annotation_text=f"VaR 95%: S/ {var_95:,.0f}",
        annotation_position="top right",
        annotation_font_color="#EF4444"
    )
    fig.update_layout(
        title="Simulación Monte Carlo — Valor del portafolio (1,000 escenarios)",
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, title="Valor final del portafolio (S/)"),
        yaxis=dict(gridcolor="#F1F5F9", title="Frecuencia"),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    perdida_max = 100000 - var_95
    return fig, var_95, perdida_max

import pandas as pd
