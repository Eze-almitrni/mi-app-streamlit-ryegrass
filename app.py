"""
Resiembra Invernal de Rye Grass en Campos Deportivos — Argentina 2024/2025
Aplicacion Streamlit: EDA interactivo + Modelo Predictivo
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Configuracion de pagina ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Rye Grass — Campos Deportivos Argentina",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .titulo-principal { font-size: 2rem; font-weight: 800; color: #1b4332; }
    .subtitulo        { font-size: 1rem; color: #555; margin-bottom: 1rem; }
    .metrica-label    { font-size: 0.85rem; color: #666; }
    .resultado-alta   { background:#d8f3dc; color:#1b4332; padding:18px; border-radius:10px; font-size:1.4rem; font-weight:700; text-align:center; }
    .resultado-media  { background:#fff3cd; color:#856404; padding:18px; border-radius:10px; font-size:1.4rem; font-weight:700; text-align:center; }
    .resultado-baja   { background:#f8d7da; color:#721c24; padding:18px; border-radius:10px; font-size:1.4rem; font-weight:700; text-align:center; }
    .info-box         { background:#e8f4f8; border-left:4px solid #2196F3; padding:12px 16px; border-radius:6px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)


# ── Carga y preparacion de datos (cacheado) ────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv("ryegrass_limpio.csv")
    return df

@st.cache_resource
def entrenar_modelo(df):
    features = [
        'Variedad_Comun', 'Provincia', 'Mes', 'Temperatura_C',
        'Precipitacion_mm', 'Humedad_Pct', 'Tipo_Suelo', 'pH_Suelo',
        'Fertilizante', 'Frecuencia_Riego', 'Densidad_Siembra_kg_100m2',
        'N_Fertilizante', 'P_Fertilizante', 'K_Fertilizante'
    ]
    target = 'Categoria_Calidad'

    # Verificar columnas NPK; si no existen, calcular desde datos disponibles
    for col in ['N_Fertilizante', 'P_Fertilizante', 'K_Fertilizante']:
        if col not in df.columns:
            df[col] = df['Calidad_Visual'] * np.random.uniform(0.8, 1.2, len(df))

    df_ml = df[[c for c in features if c in df.columns] + [target]].copy()
    features_ok = [c for c in features if c in df.columns]

    les = {}
    cols_cat = ['Variedad_Comun', 'Provincia', 'Tipo_Suelo', 'Fertilizante', 'Frecuencia_Riego']
    for col in cols_cat:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            les[col] = le

    le_target = LabelEncoder()
    df_ml[target] = le_target.fit_transform(df_ml[target])

    X = df_ml[features_ok]
    y = df_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    acc = modelo.score(X_test, y_test)

    return modelo, les, le_target, features_ok, acc, df_ml


# ── Sidebar — navegacion ──────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg", use_column_width=True)
st.sidebar.markdown("## Navegacion")
seccion = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Analisis Exploratorio (EDA)", "Predictor de Calidad", "Recomendaciones por Region"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Proyecto:** Resiembra Invernal de Rye Grass  \n**Autor:** Ezequias  \n**Curso:** Ciencia de Datos 2025")


# ── Cargar datos ──────────────────────────────────────────────────────────────
try:
    df = cargar_datos()
    modelo, les, le_target, features_ok, acc_modelo, df_ml = entrenar_modelo(df.copy())
    datos_ok = True
except FileNotFoundError:
    datos_ok = False


# ════════════════════════════════════════════════════════════════════════════════
# INICIO
# ════════════════════════════════════════════════════════════════════════════════
if seccion == "Inicio":
    st.markdown('<p class="titulo-principal">Resiembra Invernal de Rye Grass</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo">Campos Deportivos — Argentina 2024/2025 | Analisis Exploratorio + Modelo Predictivo</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    if datos_ok:
        col1.metric("Registros analizados", f"{len(df):,}")
        col2.metric("Provincias", df['Provincia'].nunique())
        col3.metric("Variedades", df['Variedad_Comun'].nunique())
        col4.metric("Precision del modelo", f"{acc_modelo:.1%}")
    else:
        col1.metric("Registros", "510")
        col2.metric("Provincias", "10")
        col3.metric("Variedades", "2")
        col4.metric("Algoritmo", "Random Forest")

    st.markdown("---")

    st.markdown("""
    ### Sobre este proyecto

    En Argentina, durante el invierno el cesped base de las canchas de futbol — generalmente **Bermuda** o **Kikuyo** —
    entra en dormicion y pierde su capacidad de cobertura. Para mantener las condiciones de juego durante toda la temporada,
    los campos deportivos realizan una **resiembra invernal con Rye Grass** (*Lolium spp.*) y (*Lolium multiflorum*)

    Este proyecto analiza los factores que determinan la calidad de esa resiembra y construye un modelo de
    Machine Learning capaz de predecir si el resultado sera de calidad **Alta**, **Media** o **Baja**.

    ### Como usar esta aplicacion

    Usa el menu de la izquierda para navegar entre las secciones:

    - **Analisis Exploratorio (EDA):** explora graficos interactivos sobre temperatura, variedad, provincia, fertilizante y mas.
    - **Predictor de Calidad:** ingresa las condiciones de tu cancha y obtene una prediccion del modelo.
    - **Recomendaciones por Region:** tabla de recomendaciones para las 6 regiones historico-geograficas de Argentina.
    """)

    if not datos_ok:
        st.warning("No se encontro el archivo `ryegrass_limpio.csv`. Coloca el archivo en la misma carpeta que `app.py` para habilitar los graficos y el predictor.")


# ════════════════════════════════════════════════════════════════════════════════
# EDA
# ════════════════════════════════════════════════════════════════════════════════
elif seccion == "Analisis Exploratorio (EDA)":
    st.markdown('<p class="titulo-principal">Analisis Exploratorio de Datos</p>', unsafe_allow_html=True)
    st.markdown("Exploracion interactiva de los factores que influyen en la calidad de la resiembra.")
    st.markdown("---")

    if not datos_ok:
        st.error("No se encontro `ryegrass_limpio.csv`. Por favor coloca el archivo en la misma carpeta que `app.py`.")
        st.stop()

    # Filtros globales en sidebar
    st.sidebar.markdown("### Filtros del EDA")
    variedades = st.sidebar.multiselect(
        "Variedad", df['Variedad_Comun'].unique().tolist(),
        default=df['Variedad_Comun'].unique().tolist()
    )
    provincias = st.sidebar.multiselect(
        "Provincia", sorted(df['Provincia'].unique().tolist()),
        default=sorted(df['Provincia'].unique().tolist())
    )

    df_f = df[df['Variedad_Comun'].isin(variedades) & df['Provincia'].isin(provincias)]

    if df_f.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()

    # Selector de grafico
    grafico = st.selectbox("Selecciona el grafico a visualizar:", [
        "1. Distribucion de Calidad Visual",
        "2. Comparacion entre Variedades",
        "3. Calidad por Provincia",
        "4. Efecto de la Temperatura",
        "5. Efecto de la Precipitacion y el Riego",
        "6. Efecto del Fertilizante",
        "7. Mejor Mes para Resembrar",
        "8. Tipo de Suelo y pH",
        "9. Matriz de Correlacion",
    ])

    fig, ax = plt.subplots(figsize=(12, 5))

    # ── Grafico 1 ─────────────────────────────────────────────────────────────
    if grafico.startswith("1."):
        st.markdown('<div class="info-box">Distribucion de los puntajes de calidad visual en toda la muestra. El minimo aceptable para una cancha de futbol es 6 sobre 9 (escala NTEP).</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].hist(df_f['Calidad_Visual'], bins=20, color='#2d6a4f', edgecolor='white')
        axes[0].axvline(6, color='red', linestyle='--', label='Minimo aceptable (6)')
        axes[0].set_xlabel('Calidad Visual (1-9)')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribucion de Calidad Visual')
        axes[0].legend()
        cats = df_f['Categoria_Calidad'].value_counts()
        colores = {'Alta': '#2d6a4f', 'Media': '#e9c46a', 'Baja': '#e63946'}
        axes[1].pie(cats, labels=cats.index, autopct='%1.1f%%',
                    colors=[colores.get(c, '#999') for c in cats.index],
                    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[1].set_title('Proporcion por Categoria de Calidad')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 2 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("2."):
        st.markdown('<div class="info-box">Comparacion de calidad visual entre Lolium perenne (Perenne) y Lolium multiflorum (Anual). El boxplot muestra la mediana, el rango intercuartil y los valores atipicos.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colores_var = {'Rye Grass Perenne': '#2d6a4f', 'Rye Grass Anual': '#e9c46a'}
        sns.boxplot(data=df_f, x='Variedad_Comun', y='Calidad_Visual', ax=axes[0],
                    palette=colores_var)
        axes[0].set_title('Calidad Visual por Variedad')
        axes[0].set_xlabel('Variedad')
        axes[0].set_ylabel('Calidad Visual (1-9)')
        sns.violinplot(data=df_f, x='Variedad_Comun', y='Calidad_Visual', ax=axes[1],
                       palette=colores_var, inner='box')
        axes[1].set_title('Distribucion de Calidad por Variedad')
        axes[1].set_xlabel('Variedad')
        axes[1].set_ylabel('Calidad Visual (1-9)')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 3 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("3."):
        st.markdown('<div class="info-box">Calidad visual promedio por provincia para cada variedad. Permite identificar que zonas son mas favorables segun el tipo de Rye Grass elegido.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, var, color in zip(axes, ['Rye Grass Perenne', 'Rye Grass Anual'], ['#2d6a4f', '#e9a010']):
            sub = df_f[df_f['Variedad_Comun'] == var].groupby('Provincia')['Calidad_Visual'].mean().sort_values()
            sub.plot(kind='barh', ax=ax, color=color, edgecolor='white')
            ax.set_title(f'Calidad por Provincia — {var}')
            ax.set_xlabel('Calidad Visual Promedio')
            ax.axvline(6, color='red', linestyle='--', linewidth=1, label='Minimo (6)')
            ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 4 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("4."):
        st.markdown('<div class="info-box">Relacion entre temperatura media y calidad visual. El rango optimo para el Perenne es 10-18°C; para el Anual 8-20°C. Fuera de ese rango la calidad tiende a disminuir.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        rangos = {'Rye Grass Perenne': (10, 18), 'Rye Grass Anual': (8, 20)}
        for ax, (var, (t_min, t_max)) in zip(axes, rangos.items()):
            sub = df_f[df_f['Variedad_Comun'] == var]
            ax.scatter(sub['Temperatura_C'], sub['Calidad_Visual'],
                       alpha=0.5, color='#2d6a4f' if 'Perenne' in var else '#e9a010', s=20)
            ax.axvspan(t_min, t_max, alpha=0.12, color='green', label=f'Rango optimo ({t_min}-{t_max}°C)')
            ax.set_title(f'Temperatura vs Calidad — {var}')
            ax.set_xlabel('Temperatura (°C)')
            ax.set_ylabel('Calidad Visual')
            ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 5 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("5."):
        st.markdown('<div class="info-box">El Rye Grass es una de las especies con mayor requerimiento hidrico. Se analiza la influencia de la precipitacion mensual y la frecuencia de riego sobre la calidad visual.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(df_f['Precipitacion_mm'], df_f['Calidad_Visual'], alpha=0.4, color='#1565c0', s=20)
        axes[0].set_title('Precipitacion vs Calidad Visual')
        axes[0].set_xlabel('Precipitacion mensual (mm)')
        axes[0].set_ylabel('Calidad Visual')
        sns.boxplot(data=df_f, x='Frecuencia_Riego', y='Calidad_Visual', ax=axes[1],
                    order=['Diario', 'Cada 2 dias', 'Cada 3 dias'], palette='Blues_r')
        axes[1].set_title('Frecuencia de Riego vs Calidad Visual')
        axes[1].set_xlabel('Frecuencia de Riego')
        axes[1].set_ylabel('Calidad Visual')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 6 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("6."):
        st.markdown('<div class="info-box">Comparacion de la calidad visual promedio segun el tipo de fertilizante aplicado. El NPK 15-15-15 balanceado tiende a dar los mejores resultados en la zona Pampeana.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        orden = df_f.groupby('Fertilizante')['Calidad_Visual'].mean().sort_values(ascending=False).index
        sns.barplot(data=df_f, x='Fertilizante', y='Calidad_Visual', order=orden,
                    ax=axes[0], palette='Greens_r')
        axes[0].set_title('Calidad por Tipo de Fertilizante')
        axes[0].set_xlabel('Fertilizante')
        axes[0].set_ylabel('Calidad Visual Promedio')
        axes[0].tick_params(axis='x', rotation=30)
        sns.boxplot(data=df_f, x='Fertilizante', y='Calidad_Visual', order=orden,
                    ax=axes[1], palette='Greens_r')
        axes[1].set_title('Distribucion de Calidad por Fertilizante')
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 7 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("7."):
        st.markdown('<div class="info-box">Calidad promedio segun el mes de resiembra. El mes optimo en la zona Pampeana es Abril, cuando la temperatura baja de 18°C. Sembrar muy temprano o muy tarde reduce la calidad.</div>', unsafe_allow_html=True)
        meses_orden = ['Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto']
        meses_num   = {3:'Marzo', 4:'Abril', 5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto'}
        df_f2 = df_f.copy()
        df_f2['Nombre_Mes'] = df_f2['Mes'].map(meses_num)
        fig, ax = plt.subplots(figsize=(12, 5))
        for var, color in zip(df_f2['Variedad_Comun'].unique(), ['#2d6a4f', '#e9a010']):
            sub = df_f2[df_f2['Variedad_Comun'] == var].groupby('Nombre_Mes')['Calidad_Visual'].mean()
            sub = sub.reindex([m for m in meses_orden if m in sub.index])
            ax.plot(sub.index, sub.values, marker='o', label=var, color=color, linewidth=2)
        ax.set_title('Calidad Visual Promedio por Mes de Resiembra')
        ax.set_xlabel('Mes')
        ax.set_ylabel('Calidad Visual Promedio')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 8 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("8."):
        st.markdown('<div class="info-box">El tipo de suelo y el pH determinan la disponibilidad de nutrientes. El rango optimo de pH para el Rye Grass es 6.0-7.0 segun INTA y picasso.com.ar.</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        orden_s = df_f.groupby('Tipo_Suelo')['Calidad_Visual'].mean().sort_values(ascending=False).index
        sns.barplot(data=df_f, x='Tipo_Suelo', y='Calidad_Visual', order=orden_s,
                    ax=axes[0], palette='YlGn')
        axes[0].set_title('Calidad por Tipo de Suelo')
        axes[0].tick_params(axis='x', rotation=30)
        axes[1].scatter(df_f['pH_Suelo'], df_f['Calidad_Visual'], alpha=0.4, color='#6a4c93', s=20)
        axes[1].axvspan(6.0, 7.0, alpha=0.15, color='green', label='pH optimo (6.0-7.0)')
        axes[1].set_title('pH del Suelo vs Calidad Visual')
        axes[1].set_xlabel('pH del Suelo')
        axes[1].set_ylabel('Calidad Visual')
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)

    # ── Grafico 9 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("9."):
        st.markdown('<div class="info-box">Mapa de calor con el coeficiente de correlacion de Pearson entre todas las variables numericas. Valores cercanos a +1 indican relacion positiva; cercanos a -1, relacion inversa.</div>', unsafe_allow_html=True)
        cols_num = [c for c in ['Calidad_Visual', 'Temperatura_C', 'Precipitacion_mm',
                                 'Humedad_Pct', 'pH_Suelo', 'Densidad_Siembra_kg_100m2'] if c in df_f.columns]
        corr = df_f[cols_num].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    ax=ax, square=True, linewidths=0.5)
        ax.set_title('Matriz de Correlacion entre Variables Numericas')
        plt.tight_layout()
        st.pyplot(fig)

    plt.close('all')


# ════════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
elif seccion == "Predictor de Calidad":
    st.markdown('<p class="titulo-principal">Predictor de Calidad de Resiembra</p>', unsafe_allow_html=True)
    st.markdown("Ingresa las condiciones de tu cancha y el modelo predice si la resiembra sera de calidad Alta, Media o Baja.")
    st.markdown("---")

    if not datos_ok:
        st.error("No se encontro `ryegrass_limpio.csv`. Coloca el archivo en la misma carpeta que `app.py`.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Variables de manejo")
        variedad = st.selectbox("Variedad de Rye Grass",
            sorted(df['Variedad_Comun'].unique().tolist()))
        fertilizante = st.selectbox("Fertilizante",
            sorted(df['Fertilizante'].unique().tolist()))
        frecuencia_riego = st.selectbox("Frecuencia de Riego",
            sorted(df['Frecuencia_Riego'].unique().tolist()))
        densidad = st.slider("Densidad de Siembra (kg/100m2)", 4.0, 12.0, 7.5, 0.5)

    with col2:
        st.markdown("#### Ubicacion y fecha")
        provincia = st.selectbox("Provincia",
            sorted(df['Provincia'].unique().tolist()))
        mes_nombre = st.selectbox("Mes de Resiembra",
            ['Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto'])
        mes_num = {'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8}[mes_nombre]

        st.markdown("#### Suelo")
        tipo_suelo = st.selectbox("Tipo de Suelo",
            sorted(df['Tipo_Suelo'].unique().tolist()))
        ph = st.slider("pH del Suelo", 4.5, 8.5, 6.5, 0.1)

    with col3:
        st.markdown("#### Clima (SMN — datos historicos)")
        temp = st.slider("Temperatura media (°C)", 0.0, 30.0,
            float(df[df['Provincia'] == provincia]['Temperatura_C'].median()), 0.5)
        precip = st.slider("Precipitacion mensual (mm)", 0.0, 200.0,
            float(df[df['Provincia'] == provincia]['Precipitacion_mm'].median()), 5.0)
        humedad = st.slider("Humedad relativa (%)", 30.0, 100.0,
            float(df[df['Provincia'] == provincia]['Humedad_Pct'].median()), 1.0)

    st.markdown("---")

    if st.button("Predecir Calidad", type="primary", use_container_width=True):

        # Calcular NPK del fertilizante
        npk = df[df['Fertilizante'] == fertilizante][
            ['N_Fertilizante', 'P_Fertilizante', 'K_Fertilizante']
        ].mean() if all(c in df.columns for c in ['N_Fertilizante', 'P_Fertilizante', 'K_Fertilizante']) \
        else pd.Series({'N_Fertilizante': 15, 'P_Fertilizante': 15, 'K_Fertilizante': 15})

        caso = pd.DataFrame([{
            'Variedad_Comun'            : variedad,
            'Provincia'                 : provincia,
            'Mes'                       : mes_num,
            'Temperatura_C'             : temp,
            'Precipitacion_mm'          : precip,
            'Humedad_Pct'               : humedad,
            'Tipo_Suelo'                : tipo_suelo,
            'pH_Suelo'                  : ph,
            'Fertilizante'              : fertilizante,
            'Frecuencia_Riego'          : frecuencia_riego,
            'Densidad_Siembra_kg_100m2' : densidad,
            'N_Fertilizante'            : npk.get('N_Fertilizante', 15),
            'P_Fertilizante'            : npk.get('P_Fertilizante', 15),
            'K_Fertilizante'            : npk.get('K_Fertilizante', 15),
        }])

        caso_ok = caso[[c for c in features_ok if c in caso.columns]]
        cols_cat = ['Variedad_Comun', 'Provincia', 'Tipo_Suelo', 'Fertilizante', 'Frecuencia_Riego']
        for col in cols_cat:
            if col in caso_ok.columns and col in les:
                try:
                    caso_ok[col] = les[col].transform(caso_ok[col].astype(str))
                except ValueError:
                    caso_ok[col] = 0

        pred  = modelo.predict(caso_ok)[0]
        proba = modelo.predict_proba(caso_ok)[0]
        categoria = le_target.inverse_transform([pred])[0]
        prob_max  = proba.max()
        confianza = 'Alta' if prob_max >= 0.85 else 'Media' if prob_max >= 0.70 else 'Baja'
        clases    = le_target.classes_

        # Mostrar resultado
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            css_class = f"resultado-{categoria.lower()}"
            st.markdown(f'<div class="{css_class}">Calidad predicha<br>{categoria}<br><small>Confianza: {confianza} ({prob_max:.0%})</small></div>',
                        unsafe_allow_html=True)

        with col_res2:
            st.markdown("**Probabilidad por categoria:**")
            for clase, prob in zip(clases, proba):
                color = '#2d6a4f' if clase == 'Alta' else '#e9c46a' if clase == 'Media' else '#e63946'
                st.markdown(f"{clase}: **{prob:.1%}**")
                st.progress(float(prob))

        # Recomendacion
        st.markdown("---")
        st.markdown("#### Recomendacion")
        if categoria == 'Alta':
            st.success("Las condiciones ingresadas son favorables para una resiembra exitosa. Mantene el plan de manejo seleccionado.")
        elif categoria == 'Media':
            st.warning("La calidad esperada es aceptable pero hay margen de mejora. Revisa la temperatura y considera aumentar la frecuencia de riego o cambiar el fertilizante.")
        else:
            st.error("Las condiciones no son favorables. Considera cambiar el mes de resiembra, el fertilizante o la variedad segun tu region.")


# ════════════════════════════════════════════════════════════════════════════════
# RECOMENDACIONES
# ════════════════════════════════════════════════════════════════════════════════
elif seccion == "Recomendaciones por Region":
    st.markdown('<p class="titulo-principal">Recomendaciones por Region Historico-Geografica</p>', unsafe_allow_html=True)
    st.markdown("Condiciones optimas de resiembra segun las 6 grandes regiones historico-geograficas de Argentina.")
    st.markdown("---")

    recomendaciones = pd.DataFrame([
        {
            'Region'        : 'Pampeana',
            'Provincias'    : 'Buenos Aires, Cordoba, Santa Fe, La Pampa',
            'Variedad'      : 'Lolium perenne',
            'Mes optimo'    : 'Abril',
            'Fertilizante'  : 'NPK 15-15-15',
            'Riego'         : 'Diario',
            'Temperatura'   : '10-18 °C',
            'Nota'          : 'Zona optima del pais para resiembra invernal',
        },
        {
            'Region'        : 'Metropolitana',
            'Provincias'    : 'Ciudad Autonoma de Buenos Aires',
            'Variedad'      : 'Lolium perenne',
            'Mes optimo'    : 'Abril',
            'Fertilizante'  : 'NPK 15-15-15',
            'Riego'         : 'Diario',
            'Temperatura'   : '10-18 °C',
            'Nota'          : 'Mismas condiciones que la zona Pampeana',
        },
        {
            'Region'        : 'Cuyo',
            'Provincias'    : 'Mendoza, San Juan, San Luis',
            'Variedad'      : 'Lolium perenne',
            'Mes optimo'    : 'Mayo',
            'Fertilizante'  : 'NPK 15-15-15',
            'Riego'         : 'Diario',
            'Temperatura'   : '8-16 °C',
            'Nota'          : 'Clima seco: el riego diario es indispensable',
        },
        {
            'Region'        : 'NEA',
            'Provincias'    : 'Entre Rios, Corrientes, Misiones, Chaco, Formosa',
            'Variedad'      : 'Lolium perenne',
            'Mes optimo'    : 'Abril',
            'Fertilizante'  : 'NPK 15-15-15',
            'Riego'         : 'Cada 2 dias',
            'Temperatura'   : '12-20 °C',
            'Nota'          : 'Alta humedad: monitorear presencia de hongos',
        },
        {
            'Region'        : 'NOA',
            'Provincias'    : 'Tucuman, Salta, Jujuy, Catamarca, La Rioja, Santiago del Estero',
            'Variedad'      : 'Lolium multiflorum',
            'Mes optimo'    : 'Mayo - Junio',
            'Fertilizante'  : 'NPK 20-10-10',
            'Riego'         : 'Diario',
            'Temperatura'   : '12-20 °C',
            'Nota'          : 'El Anual tolera mejor las temperaturas elevadas',
        },
        {
            'Region'        : 'Patagonia',
            'Provincias'    : 'Neuquen, Rio Negro, Chubut, Santa Cruz, Tierra del Fuego',
            'Variedad'      : 'Lolium perenne',
            'Mes optimo'    : 'Marzo - Abril',
            'Fertilizante'  : 'NPK 15-15-15',
            'Riego'         : 'Cada 2 dias',
            'Temperatura'   : '6-14 °C',
            'Nota'          : 'Anticipar la siembra por las heladas tempranas',
        },
    ])

    # Filtro por region
    region_sel = st.multiselect("Filtrar por region:", recomendaciones['Region'].tolist(),
                                 default=recomendaciones['Region'].tolist())
    df_rec = recomendaciones[recomendaciones['Region'].isin(region_sel)]

    st.dataframe(df_rec.set_index('Region'), use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("#### Detalle por Region")

    for _, row in df_rec.iterrows():
        with st.expander(f"{row['Region']} — {row['Variedad']}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Variedad", row['Variedad'].replace('Lolium ', 'L. '))
            c2.metric("Mes optimo", row['Mes optimo'])
            c3.metric("Fertilizante", row['Fertilizante'])
            c1.metric("Riego", row['Riego'])
            c2.metric("Temperatura", row['Temperatura'])
            st.info(f"Provincias: {row['Provincias']}")
            st.caption(f"Nota: {row['Nota']}")

    st.markdown("---")
    st.markdown("""
    **Fuentes:** SMN Argentina, INTA, turfland.com.ar, picasso.com.ar, NTEP 2022-2024.  
    **Nota:** Las recomendaciones para NEA y Patagonia se basan en criterios agronomicos de las fuentes consultadas.
    El dataset de entrenamiento cubre principalmente las regiones Pampeana, Cuyo y NOA.
    """)
