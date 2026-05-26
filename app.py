"""
Resiembra Invernal de Rye Grass en Campos Deportivos — Argentina 
Análisis Exploratorio de Datos y Modelo Predictivo de Calidad de Césped
Autor: Ezequias Almitrani |Materia: Ciencia de Datos 2026
Profesor: Gonzalo Ducca
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG GENERAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rye Grass Argentina ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&family=Source+Sans+3:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    h1, h2, h3 { font-family: 'Montserrat', sans-serif !important; font-weight: 800 !important; }
    .stApp { background: #f4f7f0; }
    .hero-box {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(27,94,32,0.25);
    }
    .hero-box h1 { color: white !important; font-size: 2.2rem; margin: 0; }
    .hero-box p { color: #c8e6c9; font-size: 1.05rem; margin-top: 0.5rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1rem;
        text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #2e7d32;
    }
    .metric-card .val { font-size: 2rem; font-weight: 800; color: #1b5e20; }
    .metric-card .lab { font-size: 0.82rem; color: #555; margin-top: 0.2rem; }
    .section-title {
        font-family: 'Montserrat', sans-serif; font-weight: 800; color: #1b5e20;
        border-bottom: 3px solid #81c784; padding-bottom: 0.3rem;
        margin-top: 1.5rem; margin-bottom: 1rem;
    }
    .insight-box {
        background: #e8f5e9; border-left: 5px solid #43a047;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
        color: #1b5e20; font-size: 0.95rem;
    }
    .warning-box {
        background: #fff8e1; border-left: 5px solid #ffa000;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.8rem 0;
        color: #5f4300; font-size: 0.95rem;
    }
    .pred-alta { background:#e8f5e9; border:2px solid #43a047; border-radius:10px; padding:1rem; text-align:center; }
    .pred-media { background:#fff8e1; border:2px solid #ffa000; border-radius:10px; padding:1rem; text-align:center; }
    .pred-baja { background:#ffebee; border:2px solid #e53935; border-radius:10px; padding:1rem; text-align:center; }
    [data-testid="stSidebar"] { background: #1b5e20 !important; }
    [data-testid="stSidebar"] * { color: #e8f5e9 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("ryegrass_limpio.csv")

df = load_data()

# ─────────────────────────────────────────────
# ENTRENAMIENTO DEL MODELO
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    feat_cols = [
        "Variedad_Comun", "Provincia", "Mes", "Temperatura_C",
        "Precipitacion_mm", "Humedad_Pct", "Tipo_Suelo", "pH_Suelo",
        "Fertilizante", "Frecuencia_Riego", "Densidad_Siembra_kg_100m2",
        "N_Fertilizante", "P_Fertilizante", "K_Fertilizante"
    ]
    cat_cols = ["Variedad_Comun", "Provincia", "Tipo_Suelo", "Fertilizante", "Frecuencia_Riego"]
    le_dict = {}
    X = df[feat_cols].copy()
    y = df["Categoria_Calidad"].copy()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Reg. Logística": Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    scores = {name: cross_val_score(m, X, y_enc, cv=5, scoring="accuracy")
              for name, m in models.items()}
    return rf, le_dict, le_y, X_test, y_test, feat_cols, cat_cols, scores, X, y_enc

rf_model, le_dict, le_y, X_test, y_test, feat_cols, cat_cols, cv_scores, X_full, y_full = train_model(df)

# ─────────────────────────────────────────────
# COLORES
# ─────────────────────────────────────────────
VERDE = "#2e7d32"; VERDE_CLR = "#81c784"; AMARILLO = "#f9a825"; ROJO = "#c62828"
PALETTE = ["#2e7d32","#81c784","#f9a825","#1565c0","#6a1b9a","#e65100"]
def color_cat(cat):
    return {"Alta": VERDE, "Media": AMARILLO, "Baja": ROJO}.get(cat, "#888")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Rye Grass AR")
    st.markdown("---")
    seccion = st.radio("Navegación", [
        "Inicio", "EDA — Exploración",
        "Modelo ML", "Predictor", "Conclusiones",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"• {len(df)} observaciones")
    st.markdown(f"• {df['Provincia'].nunique()} provincias")
    st.markdown(f"• {df.shape[1]} variables")
    st.markdown("---")
    st.markdown("**Autor:** Ezequias")
    st.markdown("**Curso:** Ciencia de Datos 2025")

# ═══════════════════════════════════════════════════════
# INICIO
# ═══════════════════════════════════════════════════════
if seccion == "Inicio":
    st.markdown("""
    <div class='hero-box'>
        <h1>Resiembra Invernal de Rye Grass</h1>
        <p>Análisis Exploratorio y Modelo Predictivo de Calidad de Césped · Argentina </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    acc = accuracy_score(y_test, rf_model.predict(X_test))
    for col, val, lab in zip(
        [c1,c2,c3,c4],
        [len(df), df['Provincia'].nunique(), df.shape[1], f"{acc:.0%}"],
        ["Observaciones","Provincias","Variables","Accuracy RF"]
    ):
        col.markdown(f"<div class='metric-card'><div class='val'>{val}</div><div class='lab'>{lab}</div></div>", unsafe_allow_html=True)

    st.markdown("<p class='section-title'>¿De qué trata el proyecto?</p>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("""
        En Argentina, durante el invierno el césped base —
        **Bermuda** (*Cynodon dactylon*) o **Kikuyo** (*Pennisetum clandestinum*) —
        entra en dormición vegetativa y pierde cobertura.

        Para mantener las condiciones de juego, los campos deportivos realizan una
        **resiembra invernal con Rye Grass** (*Lolium spp y Lolium multiflorum*).

        Este proyecto analiza los **factores agronómicos y climáticos** que determinan
        la calidad de esa resiembra usando técnicas de EDA y aprendizaje automático.
        """)
        st.markdown("""<div class='insight-box'>
        <strong>Pregunta central:</strong> ¿En qué provincia argentina y bajo qué combinación
        de condiciones se obtiene la mayor calidad de césped al realizar resiembra invernal con
        <em>Lolium perenne</em> y <em>Lolium multiflorum</em>?
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("<p class='section-title'>Objetivos</p>", unsafe_allow_html=True)
        for o in [
            "Comparar variedades Perenne vs Anual por provincia",
            "Identificar rango óptimo de temperatura y precipitación",
            "Determinar fertilizante y riego óptimos",
            "Analizar el mes óptimo de siembra por región",
            "Construir modelo ML (Alta/Media/Baja)",
            "Generar recomendaciones por región",
        ]: st.markdown(f"{o}")

    st.markdown("<p class='section-title'>Recomendaciones por Región</p>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Región": ["Pampeana","Metropolitana","Cuyo","NEA","NOA","Patagonia"],
        "Variedad": ["L. perenne","L. perenne","L. perenne","L. perenne","L. multiflorum","L. perenne"],
        "Mes óptimo": ["Abril","Abril","Mayo","Abril","Mayo-Junio","Marzo-Abril"],
        "Fertilizante": ["NPK 15-15-15","NPK 15-15-15","NPK 15-15-15","NPK 15-15-15","NPK 20-10-10","NPK 15-15-15"],
        "Nota": ["Zona óptima del país","Como Pampeana","Riego diario obligatorio",
                 "Monitorear hongos","Tolera calor","Anticipar heladas"],
    }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# EDA
# ═══════════════════════════════════════════════════════
elif seccion == "🔍 EDA — Exploración":
    st.markdown("<h2 class='section-title'>Análisis Exploratorio de Datos</h2>", unsafe_allow_html=True)
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📊 Distribución","Variedades","🗺️ Provincias",
        "🌡️ Clima","🧪 Suelo & Fertilizante","🔗 Correlaciones"])

    with tab1:
        st.markdown("### Distribución de Calidad Visual (Escala NTEP 1-9)")
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            counts = df["Categoria_Calidad"].value_counts()
            bars = ax.bar(counts.index, counts.values,
                          color=[color_cat(c) for c in counts.index],
                          edgecolor="white", linewidth=1.5, width=0.5)
            for bar,val in zip(bars,counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                        f'{val}\n({val/len(df):.0%})', ha='center', fontsize=10, fontweight='bold')
            ax.set_title("Categoría de Calidad", fontsize=13, fontweight='bold', color=VERDE)
            ax.set_xlabel("Categoría"); ax.set_ylabel("Cantidad")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            ax.hist(df["Calidad_Visual"], bins=16, color=VERDE, edgecolor="white", alpha=0.85)
            ax.axvline(df["Calidad_Visual"].mean(), color=AMARILLO, linewidth=2.5,
                       label=f'Media: {df["Calidad_Visual"].mean():.1f}')
            ax.axvline(df["Calidad_Visual"].median(), color=ROJO, linewidth=2.5, linestyle='--',
                       label=f'Mediana: {df["Calidad_Visual"].median():.1f}')
            ax.legend(); ax.set_title("Histograma Calidad Visual (NTEP)", fontweight='bold', color=VERDE)
            ax.set_xlabel("Puntaje NTEP"); ax.set_ylabel("Frecuencia")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        st.markdown("""<div class='insight-box'>
        <strong>Insight:</strong> El 90% de las observaciones tienen calidad "Alta" y el 10% "Media".
        No hay registros "Baja", indicando un dataset sesgado hacia condiciones favorables.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Comparación de Variedades de Rye Grass")
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            var_grp = df.groupby("Variedad_Comun")["Calidad_Visual"].mean().sort_values(ascending=False)
            ax.barh(var_grp.index, var_grp.values,
                    color=[VERDE if i==0 else VERDE_CLR for i in range(len(var_grp))],
                    edgecolor='white')
            for i,val in enumerate(var_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontweight='bold')
            ax.set_title("Calidad Media por Variedad", fontweight='bold', color=VERDE)
            ax.set_xlabel("Puntaje NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            for var,grp in df.groupby("Variedad_Comun"):
                ax.hist(grp["Calidad_Visual"], bins=12, alpha=0.65, label=var, edgecolor='white')
            ax.set_title("Distribución por Variedad", fontweight='bold', color=VERDE)
            ax.set_xlabel("Calidad Visual"); ax.set_ylabel("Frecuencia"); ax.legend()
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

        fig,ax = plt.subplots(figsize=(10,4))
        zonas_var = df.groupby(["Zona_Climatica","Variedad_Comun"])["Calidad_Visual"].mean().unstack()
        x = np.arange(len(zonas_var)); w = 0.35
        for i,(col,color) in enumerate(zip(zonas_var.columns,[VERDE,VERDE_CLR])):
            ax.bar(x+i*w, zonas_var[col], w, label=col, color=color, edgecolor='white')
        ax.set_xticks(x+w/2); ax.set_xticklabels(zonas_var.index, rotation=15)
        ax.set_title("Calidad por Variedad y Zona Climática", fontweight='bold', color=VERDE)
        ax.set_ylabel("Calidad Visual promedio"); ax.legend()
        ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()
        st.markdown("""<div class='insight-box'>
        📌 <strong>Insight:</strong> <em>Lolium perenne</em> supera al Anual en zonas templadas.
        En el NOA, el Rye Grass Anual es comparable o superior por su tolerancia al calor.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Resultados por Provincia y Zona")
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(7,5))
            prov_grp = df.groupby("Provincia")["Calidad_Visual"].mean().sort_values()
            colors_p = [VERDE if v>=7.5 else VERDE_CLR if v>=7.0 else AMARILLO for v in prov_grp.values]
            ax.barh(prov_grp.index, prov_grp.values, color=colors_p, edgecolor='white')
            for i,val in enumerate(prov_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.axvline(prov_grp.mean(), color=ROJO, linewidth=1.5, linestyle='--',
                       label=f'Media: {prov_grp.mean():.2f}')
            ax.legend(); ax.set_title("Calidad Media por Provincia", fontweight='bold', color=VERDE)
            ax.set_xlabel("Puntaje NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(7,5))
            zona_grp = df.groupby("Zona_Climatica")["Calidad_Visual"].mean().sort_values()
            ax.barh(zona_grp.index, zona_grp.values, color=PALETTE[:len(zona_grp)], edgecolor='white')
            for i,val in enumerate(zona_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.set_title("Calidad Media por Zona Climática", fontweight='bold', color=VERDE)
            ax.set_xlabel("Puntaje NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()

        st.markdown("### Mes Óptimo de Siembra por Provincia (Mapa de Calor)")
        fig,ax = plt.subplots(figsize=(12,4))
        mes_prov = df.groupby(["Nombre_Mes","Provincia"])["Calidad_Visual"].mean().unstack()
        orden = [m for m in ["Marzo","Abril","Mayo","Junio","Julio","Agosto"] if m in mes_prov.index]
        mes_prov = mes_prov.reindex(orden)
        im = ax.imshow(mes_prov.T.values, cmap="Greens", aspect="auto", vmin=6.5, vmax=9)
        ax.set_xticks(range(len(mes_prov.index))); ax.set_xticklabels(mes_prov.index)
        ax.set_yticks(range(len(mes_prov.columns))); ax.set_yticklabels(mes_prov.columns, fontsize=8)
        plt.colorbar(im, ax=ax, label="Calidad NTEP")
        for i in range(len(mes_prov.columns)):
            for j in range(len(mes_prov.index)):
                val = mes_prov.T.values[i,j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7)
        ax.set_title("Mapa de Calor: Mes × Provincia", fontweight='bold', color=VERDE)
        fig.patch.set_facecolor("#f4f7f0")
        st.pyplot(fig); plt.close()

    with tab4:
        st.markdown("### Influencia del Clima en la Calidad")
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            for cat,color in [("Alta",VERDE),("Media",AMARILLO)]:
                sub = df[df["Categoria_Calidad"]==cat]
                ax.scatter(sub["Temperatura_C"], sub["Calidad_Visual"],
                           alpha=0.4, color=color, label=cat, s=20)
            ax.set_title("Temperatura vs Calidad Visual", fontweight='bold', color=VERDE)
            ax.set_xlabel("Temperatura (°C)"); ax.set_ylabel("Calidad NTEP"); ax.legend()
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            for cat,color in [("Alta",VERDE),("Media",AMARILLO)]:
                sub = df[df["Categoria_Calidad"]==cat]
                ax.scatter(sub["Precipitacion_mm"], sub["Calidad_Visual"],
                           alpha=0.4, color=color, label=cat, s=20)
            ax.set_title("Precipitación vs Calidad Visual", fontweight='bold', color=VERDE)
            ax.set_xlabel("Precipitación (mm)"); ax.set_ylabel("Calidad NTEP"); ax.legend()
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        st.markdown("""<div class='insight-box'>
        <strong>Insight:</strong> El rango óptimo para <em>L. perenne</em> es 10-18 °C.
        Para <em>L. multiflorum</em> es 8-20 °C, mostrando mayor adaptabilidad térmica.
        </div>""", unsafe_allow_html=True)

    with tab5:
        st.markdown("### Tipo de Suelo, pH, Fertilizante y Riego")
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            suelo_grp = df.groupby("Tipo_Suelo")["Calidad_Visual"].mean().sort_values()
            ax.barh(suelo_grp.index, suelo_grp.values, color=VERDE, edgecolor='white', alpha=0.85)
            for i,val in enumerate(suelo_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.set_title("Calidad por Tipo de Suelo", fontweight='bold', color=VERDE)
            ax.set_xlabel("Calidad NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            for cat,color in [("Alta",VERDE),("Media",AMARILLO)]:
                sub = df[df["Categoria_Calidad"]==cat]
                ax.hist(sub["pH_Suelo"], bins=12, alpha=0.6, color=color, label=cat, edgecolor='white')
            ax.set_title("pH del Suelo por Categoría", fontweight='bold', color=VERDE)
            ax.set_xlabel("pH del Suelo"); ax.set_ylabel("Frecuencia"); ax.legend()
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(6,4))
            fert_grp = df.groupby("Fertilizante")["Calidad_Visual"].mean().sort_values()
            ax.barh(fert_grp.index, fert_grp.values, color=PALETTE[:len(fert_grp)], edgecolor='white')
            for i,val in enumerate(fert_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.set_title("Calidad por Fertilizante", fontweight='bold', color=VERDE)
            ax.set_xlabel("Calidad NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            fig,ax = plt.subplots(figsize=(6,4))
            riego_grp = df.groupby("Frecuencia_Riego")["Calidad_Visual"].mean().sort_values()
            ax.barh(riego_grp.index, riego_grp.values,
                    color=[VERDE,VERDE_CLR,AMARILLO], edgecolor='white')
            for i,val in enumerate(riego_grp.values):
                ax.text(val+0.02, i, f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
            ax.set_title("Calidad por Frecuencia de Riego", fontweight='bold', color=VERDE)
            ax.set_xlabel("Calidad NTEP promedio")
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        st.markdown("""<div class='insight-box'>
        <strong>Insight:</strong> NPK 15-15-15 es el fertilizante con mejores resultados en la mayoría
        de las zonas. El riego diario maximiza la calidad, especialmente en regiones áridas como Cuyo.
        </div>""", unsafe_allow_html=True)

    with tab6:
        st.markdown("### Matriz de Correlación — Variables Numéricas")
        num_cols = ["Temperatura_C","Precipitacion_mm","Humedad_Pct","pH_Suelo",
                    "Densidad_Siembra_kg_100m2","Calidad_Visual","Color","Densidad_Cesped",
                    "Textura","Resistencia_Enfermedades","Dias_Germinacion","Mes"]
        corr = df[num_cols].corr()
        fig,ax = plt.subplots(figsize=(10,8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, ax=ax, linewidths=0.5, square=True,
                    cbar_kws={"shrink":0.8}, annot_kws={"size":7})
        ax.set_title("Correlaciones entre Variables", fontweight='bold', color=VERDE, fontsize=13)
        fig.patch.set_facecolor("#f4f7f0"); plt.tight_layout()
        st.pyplot(fig); plt.close()
        top_corr = corr["Calidad_Visual"].drop("Calidad_Visual").abs().sort_values(ascending=False).head(5)
        st.markdown("**Variables más correlacionadas con Calidad Visual:**")
        for var,val in top_corr.items():
            st.markdown(f"• `{var}`: correlación = **{corr['Calidad_Visual'][var]:.3f}**")

# ═══════════════════════════════════════════════════════
# MODELO ML
# ═══════════════════════════════════════════════════════
elif seccion == "Modelo ML":
    st.markdown("<h2 class='section-title'>Modelo de Machine Learning</h2>", unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs([
        "Metodología","Comparación Modelos","Random Forest","Evaluación"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Variables de Entrada (X)")
            for f in feat_cols: st.markdown(f"• `{f}`")
        with col2:
            st.markdown("#### Parámetros")
            st.markdown("""
| Parámetro | Valor |
|---|---|
| Algoritmo | Random Forest |
| N° de árboles | 100 |
| Profundidad máx | 10 |
| Train / Test | 80% / 20% |
| Validación | 5-Fold CV |
| Variable objetivo | Categoría (Alta/Media/Baja) |
""")
        st.markdown("""<div class='insight-box'>
        <strong>Codificación:</strong> Variables categóricas con <code>LabelEncoder</code>.
        La Regresión Logística usa un Pipeline con <code>StandardScaler</code>.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Validación Cruzada 5-Fold — Comparación de Modelos")
        model_names = list(cv_scores.keys())
        means = [cv_scores[m].mean() for m in model_names]
        stds = [cv_scores[m].std() for m in model_names]
        fig,ax = plt.subplots(figsize=(8,4))
        bars = ax.bar(model_names, means, yerr=stds, capsize=6,
                      color=[ROJO,AMARILLO,VERDE,"#1565c0"], edgecolor='white', width=0.5)
        for bar,m,s in zip(bars,means,stds):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.005,
                    f'{m:.3f}', ha='center', fontweight='bold', fontsize=10)
        ax.set_ylim(0,1.12); ax.set_ylabel("Accuracy (5-Fold CV)")
        ax.set_title("Comparación de Modelos", fontweight='bold', color=VERDE)
        ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()
        st.dataframe(pd.DataFrame({
            "Modelo": model_names,
            "Accuracy Media": [f"{m:.4f}" for m in means],
            "Desv. Estándar": [f"{s:.4f}" for s in stds],
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Importancia de Variables — Random Forest")
        importances = rf_model.feature_importances_
        feat_imp = pd.Series(importances, index=feat_cols).sort_values()
        fig,ax = plt.subplots(figsize=(8,5))
        colors_fi = [VERDE if v>=feat_imp.quantile(0.75) else VERDE_CLR for v in feat_imp.values]
        ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='white')
        ax.axvline(feat_imp.mean(), color=ROJO, linestyle='--', linewidth=1.5, label='Media')
        for i,val in enumerate(feat_imp.values):
            ax.text(val+0.001, i, f'{val:.3f}', va='center', fontsize=8)
        ax.set_title("Importancia de Variables (RF)", fontweight='bold', color=VERDE)
        ax.set_xlabel("Importancia relativa"); ax.legend()
        ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        top3 = feat_imp.sort_values(ascending=False).head(3)
        st.markdown(f"""<div class='insight-box'>
        <strong>Top 3 variables:</strong>
        {", ".join([f"<strong>{v}</strong> ({s:.3f})" for v,s in top3.items()])}
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("#### Métricas en Test Set (20%)")
        y_pred = rf_model.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le_y.classes_, output_dict=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc_test:.4f}")
        c2.metric("Registros Test", len(y_test))
        c3.metric("Clases", len(le_y.classes_))
        report_df = pd.DataFrame(report).T.iloc[:-3].round(3)
        report_df.index.name = "Clase"
        st.dataframe(report_df.reset_index(), use_container_width=True, hide_index=True)
        st.markdown("#### Matriz de Confusión")
        fig,ax = plt.subplots(figsize=(5,4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le_y.classes_)
        disp.plot(ax=ax, colorbar=False, cmap="Greens")
        ax.set_title("Matriz de Confusión — RF", fontweight='bold', color=VERDE)
        fig.patch.set_facecolor("#f4f7f0"); st.pyplot(fig); plt.close()
        st.markdown("""<div class='warning-box'>
        <strong>Limitación:</strong> El dataset presenta desbalance de clases (90% Alta, 10% Media,
        0% Baja), lo que puede inflar el accuracy. Se recomienda SMOTE en producción.
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PREDICTOR
# ═══════════════════════════════════════════════════════
elif seccion == "Predictor":
    st.markdown("<h2 class='section-title'>Predictor de Calidad de Césped</h2>", unsafe_allow_html=True)
    st.markdown("Completá las condiciones de tu resiembra y el modelo predice la calidad esperada.")

    with st.form("predictor_form"):
        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown("**Variedad y Ubicación**")
            variedad = st.selectbox("Variedad", sorted(df["Variedad_Comun"].unique()))
            provincia = st.selectbox("Provincia", sorted(df["Provincia"].unique()))
            mes = st.slider("Mes de Siembra", 3, 8, 4,
                                   help="3=Marzo · 4=Abril · 5=Mayo · 6=Junio · 7=Julio · 8=Agosto")
        with col2:
            st.markdown("**Condiciones Climáticas**")
            temperatura = st.slider("Temperatura (°C)", 0.0, 30.0, 13.0, 0.5)
            precipitacion = st.slider("Precipitación (mm)", 0.0, 200.0, 50.0, 5.0)
            humedad = st.slider("Humedad (%)", 40.0, 100.0, 75.0, 1.0)
        with col3:
            st.markdown("**Manejo Agronómico**")
            tipo_suelo = st.selectbox("Tipo de Suelo", sorted(df["Tipo_Suelo"].unique()))
            ph = st.slider("pH del Suelo", 5.0, 8.5, 6.5, 0.1)
            fertilizante= st.selectbox("Fertilizante", sorted(df["Fertilizante"].unique()))
            riego = st.selectbox("Frecuencia de Riego", sorted(df["Frecuencia_Riego"].unique()))
            densidad = st.slider("Densidad Siembra (kg/100m²)", 2.0, 8.0, 4.0, 0.5)
        submitted = st.form_submit_button("🌱 Predecir Calidad", use_container_width=True)

    if submitted:
        fert_map = {"NPK 15-15-15":(15,15,15),"NPK 20-10-10":(20,10,10),
                    "Sin fertilizante":(0,0,0),"Urea (N puro)":(46,0,0)}
        n_f,p_f,k_f = fert_map.get(fertilizante,(15,15,15))
        input_data = pd.DataFrame([{
            "Variedad_Comun":variedad,"Provincia":provincia,"Mes":mes,
            "Temperatura_C":temperatura,"Precipitacion_mm":precipitacion,"Humedad_Pct":humedad,
            "Tipo_Suelo":tipo_suelo,"pH_Suelo":ph,"Fertilizante":fertilizante,
            "Frecuencia_Riego":riego,"Densidad_Siembra_kg_100m2":densidad,
            "N_Fertilizante":n_f,"P_Fertilizante":p_f,"K_Fertilizante":k_f,
        }])
        for col in cat_cols:
            le = le_dict[col]; val = input_data[col].values[0]
            input_data[col] = le.transform([val])[0] if val in le.classes_ else 0

        proba = rf_model.predict_proba(input_data)[0]
        pred_enc = rf_model.predict(input_data)[0]
        pred_label = le_y.inverse_transform([pred_enc])[0]
        confianza = max(proba)

        emoji_map = {"Alta":"","Media":"","Baja":""}
        css_map = {"Alta":"pred-alta","Media":"pred-media","Baja":"pred-baja"}
        st.markdown(f"""
        <div class='{css_map[pred_label]}'>
            <h2>{emoji_map[pred_label]} Calidad Predicha: {pred_label}</h2>
            <p style='font-size:1.1rem'>Confianza del modelo: <strong>{confianza:.1%}</strong></p>
        </div>""", unsafe_allow_html=True)

        col1,col2 = st.columns(2)
        with col1:
            fig,ax = plt.subplots(figsize=(5,3))
            prob_labels = le_y.inverse_transform(range(len(proba)))
            ax.bar(prob_labels, proba, color=[color_cat(l) for l in prob_labels],
                   edgecolor='white', width=0.4)
            for i,(label,p) in enumerate(zip(prob_labels,proba)):
                ax.text(i, p+0.01, f'{p:.1%}', ha='center', fontweight='bold')
            ax.set_ylim(0,1.15); ax.set_ylabel("Probabilidad")
            ax.set_title("Probabilidad por Clase", fontweight='bold', color=VERDE)
            ax.set_facecolor("#f4f7f0"); fig.patch.set_facecolor("#f4f7f0")
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            st.pyplot(fig); plt.close()
        with col2:
            st.markdown("**Condiciones ingresadas:**")
            meses_nombre = {3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto"}
            for k,v in {
                "Variedad":variedad,"Provincia":provincia,
                "Mes":f"{mes} ({meses_nombre[mes]})","Temperatura":f"{temperatura}°C",
                "Precipitación":f"{precipitacion} mm","Humedad":f"{humedad}%",
                "Suelo":tipo_suelo,"pH":ph,"Fertilizante":fertilizante,
                "Riego":riego,"Densidad":f"{densidad} kg/100m²",
            }.items(): st.markdown(f"• **{k}:** {v}")

# ═══════════════════════════════════════════════════════
# CONCLUSIONES
# ═══════════════════════════════════════════════════════
elif seccion == "📋 Conclusiones":
    st.markdown("<h2 class='section-title'>Conclusiones y Recomendaciones</h2>", unsafe_allow_html=True)

    st.markdown("### Hallazgos Principales")
    for k,v in {
        "Variedad óptima": "L. perenne en zonas templadas; L. multiflorum en zonas cálidas (NOA)",
        "Temperatura óptima": "10-18 °C para Perenne; 8-20 °C para Anual",
        "Mes óptimo (Pampeana)": "Abril",
        "Fertilizante": "NPK 15-15-15 para la mayoría; NPK 20-10-10 en el NOA",
        "Riego": "Diario en todas las zonas; cada 2 días donde hay más lluvia",
        "Variables más importantes": "Temperatura, mes de resiembra y tipo de fertilizante",
    }.items(): st.markdown(f"**{k}:** {v}")

    st.markdown("### 📋 Recomendaciones por Región")
    st.dataframe(pd.DataFrame({
        "Región": ["Pampeana","Metropolitana","Cuyo","NEA","NOA","Patagonia"],
        "Variedad": ["L. perenne","L. perenne","L. perenne","L. perenne","L. multiflorum","L. perenne"],
        "Mes óptimo": ["Abril","Abril","Mayo","Abril","Mayo-Junio","Marzo-Abril"],
        "Fertilizante": ["NPK 15-15-15"]*4+["NPK 20-10-10","NPK 15-15-15"],
        "Nota": ["Zona óptima del país","Como Pampeana","Riego diario obligatorio",
                         "Monitorear hongos","Tolera calor; germina rápido","Anticipar heladas"],
    }), use_container_width=True, hide_index=True)

    st.markdown("### Limitaciones del Estudio")
    for t,d in [
        ("Cobertura parcial","Solo 10 de las 24 provincias argentinas incluidas."),
        ("Datos climáticos históricos","Normales SMN 1981-2010, ~1.5°C por debajo del clima actual."),
        ("Variables no incluidas","Patógenos, compactación del suelo, tráfico en la cancha."),
        ("Desbalance de clases","90% Alta, 10% Media, 0% Baja — accuracy puede estar inflado."),
    ]: st.markdown(f"**{t}:** {d}")

    st.markdown("### Fuentes")
    for k,v in {
        "Características agronómicas":"turfland.com.ar y picasso.com.ar",
        "Datos climáticos":"Servicio Meteorológico Nacional (SMN) — datos.gob.ar",
        "Tipos de suelo":"INTA Argentina",
        "Fertilización":"diariodecuyo.com.ar y agroconsultoraplus.com",
        "Estándar NTEP":"National Turfgrass Evaluation Program 2022-2024",
    }.items(): st.markdown(f"• **{k}:** {v}")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#555;font-size:0.9rem;padding:1rem;'>
    🌱 <strong>Resiembra Invernal de Rye Grass — Argentina </strong><br>
    Proyecto académico · Ciencia de Datos · <strong>Ezequias</strong> · 2026
    </div>""", unsafe_allow_html=True)
