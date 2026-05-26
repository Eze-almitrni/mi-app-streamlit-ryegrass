"""
Resiembra Invernal de Rye Grass en Campos Deportivos — Argentina 2024/2025
Aplicacion Streamlit: EDA interactivo + Modelo Predictivo
"""

import streamlit as st
import pandas as pd
import numpy as np
# graficos nativos streamlit
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
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg", use_container_width=True)
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
    los campos deportivos realizan una **resiembra invernal con Rye Grass** (*Lolium spp.*).

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

    colores_var = {'Rye Grass Perenne': '#2d6a4f', 'Rye Grass Anual': '#e9a010'}

    # ── Grafico 1 ─────────────────────────────────────────────────────────────
    if grafico.startswith("1."):
        st.markdown('<div class="info-box">Distribucion de los puntajes de calidad visual en toda la muestra. El minimo aceptable para una cancha de futbol es 6 sobre 9 (escala NTEP).</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_f, x='Calidad_Visual', nbins=20,
                               color_discrete_sequence=['#2d6a4f'],
                               title='Distribucion de Calidad Visual',
                               labels={'Calidad_Visual': 'Calidad Visual (1-9)', 'count': 'Frecuencia'})
            fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Minimo aceptable (6)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cats = df_f['Categoria_Calidad'].value_counts().reset_index()
            cats.columns = ['Categoria', 'Cantidad']
            colores_pie = {'Alta': '#2d6a4f', 'Media': '#e9c46a', 'Baja': '#e63946'}
            fig2 = px.pie(cats, names='Categoria', values='Cantidad',
                          color='Categoria', color_discrete_map=colores_pie,
                          title='Proporcion por Categoria de Calidad')
            st.plotly_chart(fig2, use_container_width=True)

    # ── Grafico 2 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("2."):
        st.markdown('<div class="info-box">Comparacion de calidad visual entre Lolium perenne (Perenne) y Lolium multiflorum (Anual).</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df_f, x='Variedad_Comun', y='Calidad_Visual',
                         color='Variedad_Comun', color_discrete_map=colores_var,
                         title='Calidad Visual por Variedad',
                         labels={'Variedad_Comun': 'Variedad', 'Calidad_Visual': 'Calidad Visual (1-9)'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.violin(df_f, x='Variedad_Comun', y='Calidad_Visual',
                             color='Variedad_Comun', color_discrete_map=colores_var,
                             box=True, title='Distribucion de Calidad por Variedad',
                             labels={'Variedad_Comun': 'Variedad', 'Calidad_Visual': 'Calidad Visual (1-9)'})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Grafico 3 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("3."):
        st.markdown('<div class="info-box">Calidad visual promedio por provincia para cada variedad.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for col_ui, var, color in zip([col1, col2], ['Rye Grass Perenne', 'Rye Grass Anual'], ['#2d6a4f', '#e9a010']):
            sub = df_f[df_f['Variedad_Comun'] == var].groupby('Provincia')['Calidad_Visual'].mean().reset_index()
            sub = sub.sort_values('Calidad_Visual')
            fig = px.bar(sub, x='Calidad_Visual', y='Provincia', orientation='h',
                         title=f'Calidad por Provincia — {var}',
                         color_discrete_sequence=[color],
                         labels={'Calidad_Visual': 'Calidad Visual Promedio'})
            fig.add_vline(x=6, line_dash="dash", line_color="red", annotation_text="Minimo (6)")
            with col_ui:
                st.plotly_chart(fig, use_container_width=True)

    # ── Grafico 4 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("4."):
        st.markdown('<div class="info-box">Relacion entre temperatura media y calidad visual. El rango optimo para el Perenne es 10-18°C; para el Anual 8-20°C.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        rangos = {'Rye Grass Perenne': (10, 18), 'Rye Grass Anual': (8, 20)}
        for col_ui, (var, (t_min, t_max)) in zip([col1, col2], rangos.items()):
            sub = df_f[df_f['Variedad_Comun'] == var]
            color = '#2d6a4f' if 'Perenne' in var else '#e9a010'
            fig = px.scatter(sub, x='Temperatura_C', y='Calidad_Visual',
                             title=f'Temperatura vs Calidad — {var}',
                             color_discrete_sequence=[color], opacity=0.5,
                             labels={'Temperatura_C': 'Temperatura (°C)', 'Calidad_Visual': 'Calidad Visual'})
            fig.add_vrect(x0=t_min, x1=t_max, fillcolor="green", opacity=0.1,
                          annotation_text=f"Optimo ({t_min}-{t_max}°C)")
            with col_ui:
                st.plotly_chart(fig, use_container_width=True)

    # ── Grafico 5 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("5."):
        st.markdown('<div class="info-box">El Rye Grass es una de las especies con mayor requerimiento hidrico.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df_f, x='Precipitacion_mm', y='Calidad_Visual',
                             title='Precipitacion vs Calidad Visual',
                             color_discrete_sequence=['#1565c0'], opacity=0.4,
                             labels={'Precipitacion_mm': 'Precipitacion mensual (mm)', 'Calidad_Visual': 'Calidad Visual'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            orden_riego = ['Diario', 'Cada 2 dias', 'Cada 3 dias']
            df_riego = df_f[df_f['Frecuencia_Riego'].isin(orden_riego)]
            fig2 = px.box(df_riego, x='Frecuencia_Riego', y='Calidad_Visual',
                          category_orders={'Frecuencia_Riego': orden_riego},
                          title='Frecuencia de Riego vs Calidad Visual',
                          color_discrete_sequence=['#1565c0'],
                          labels={'Frecuencia_Riego': 'Frecuencia de Riego', 'Calidad_Visual': 'Calidad Visual'})
            st.plotly_chart(fig2, use_container_width=True)

    # ── Grafico 6 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("6."):
        st.markdown('<div class="info-box">Comparacion de la calidad visual promedio segun el tipo de fertilizante aplicado.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        orden = df_f.groupby('Fertilizante')['Calidad_Visual'].mean().sort_values(ascending=False).index.tolist()
        with col1:
            media_fert = df_f.groupby('Fertilizante')['Calidad_Visual'].mean().reset_index()
            media_fert = media_fert.sort_values('Calidad_Visual', ascending=False)
            fig = px.bar(media_fert, x='Fertilizante', y='Calidad_Visual',
                         title='Calidad por Tipo de Fertilizante',
                         color_discrete_sequence=['#2d6a4f'],
                         labels={'Calidad_Visual': 'Calidad Visual Promedio'})
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.box(df_f, x='Fertilizante', y='Calidad_Visual',
                          category_orders={'Fertilizante': orden},
                          title='Distribucion de Calidad por Fertilizante',
                          color_discrete_sequence=['#2d6a4f'],
                          labels={'Calidad_Visual': 'Calidad Visual'})
            fig2.update_xaxes(tickangle=30)
            st.plotly_chart(fig2, use_container_width=True)

    # ── Grafico 7 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("7."):
        st.markdown('<div class="info-box">Calidad promedio segun el mes de resiembra. El mes optimo en la zona Pampeana es Abril.</div>', unsafe_allow_html=True)
        meses_orden = ['Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto']
        meses_num = {3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto'}
        df_f2 = df_f.copy()
        df_f2['Nombre_Mes'] = df_f2['Mes'].map(meses_num)
        fig = go.Figure()
        for var, color in zip(df_f2['Variedad_Comun'].unique(), ['#2d6a4f', '#e9a010']):
            sub = df_f2[df_f2['Variedad_Comun'] == var].groupby('Nombre_Mes')['Calidad_Visual'].mean()
            sub = sub.reindex([m for m in meses_orden if m in sub.index])
            fig.add_trace(go.Scatter(x=sub.index, y=sub.values, mode='lines+markers',
                                     name=var, line=dict(color=color, width=2),
                                     marker=dict(size=8)))
        fig.update_layout(title='Calidad Visual Promedio por Mes de Resiembra',
                          xaxis_title='Mes', yaxis_title='Calidad Visual Promedio')
        st.plotly_chart(fig, use_container_width=True)

    # ── Grafico 8 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("8."):
        st.markdown('<div class="info-box">El tipo de suelo y el pH determinan la disponibilidad de nutrientes. Rango optimo de pH para Rye Grass: 6.0-7.0.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            orden_s = df_f.groupby('Tipo_Suelo')['Calidad_Visual'].mean().sort_values(ascending=False).index.tolist()
            media_suelo = df_f.groupby('Tipo_Suelo')['Calidad_Visual'].mean().reset_index()
            media_suelo = media_suelo.sort_values('Calidad_Visual', ascending=False)
            fig = px.bar(media_suelo, x='Tipo_Suelo', y='Calidad_Visual',
                         title='Calidad por Tipo de Suelo',
                         color_discrete_sequence=['#6a4c93'],
                         labels={'Calidad_Visual': 'Calidad Visual Promedio', 'Tipo_Suelo': 'Tipo de Suelo'})
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(df_f, x='pH_Suelo', y='Calidad_Visual',
                              title='pH del Suelo vs Calidad Visual',
                              color_discrete_sequence=['#6a4c93'], opacity=0.4,
                              labels={'pH_Suelo': 'pH del Suelo', 'Calidad_Visual': 'Calidad Visual'})
            fig2.add_vrect(x0=6.0, x1=7.0, fillcolor="green", opacity=0.1,
                           annotation_text="pH optimo (6.0-7.0)")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Grafico 9 ─────────────────────────────────────────────────────────────
    elif grafico.startswith("9."):
        st.markdown('<div class="info-box">Mapa de calor con el coeficiente de correlacion de Pearson entre todas las variables numericas.</div>', unsafe_allow_html=True)
        cols_num = [c for c in ['Calidad_Visual', 'Temperatura_C', 'Precipitacion_mm',
                                 'Humedad_Pct', 'pH_Suelo', 'Densidad_Siembra_kg_100m2'] if c in df_f.columns]
        corr = df_f[cols_num].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdYlGn',
                        zmin=-1, zmax=1, title='Matriz de Correlacion entre Variables Numericas',
                        aspect='auto')
        st.plotly_chart(fig, use_container_width=True)


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
        variedad = st.selectbox("Variedad de Rye Grass", sorted(df['Variedad_Comun'].unique().tolist()))
        fertilizante = st.selectbox("Fertilizante", sorted(df['Fertilizante'].unique().tolist()))
        frecuencia_riego = st.selectbox("Frecuencia de Riego", sorted(df['Frecuencia_Riego'].unique().tolist()))
        densidad = st.slider("Densidad de Siembra (kg/100m2)", 4.0, 12.0, 7.5, 0.5)

    with col2:
        st.markdown("#### Ubicacion y fecha")
        provincia = st.selectbox("Provincia", sorted(df['Provincia'].unique().tolist()))
        mes_nombre = st.selectbox("Mes de Resiembra", ['Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto'])
        mes_num = {'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8}[mes_nombre]
        st.markdown("#### Suelo")
        tipo_suelo = st.selectbox("Tipo de Suelo", sorted(df['Tipo_Suelo'].unique().tolist()))
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

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            css_class = f"resultado-{categoria.lower()}"
            st.markdown(f'<div class="{css_class}">Calidad predicha<br>{categoria}<br><small>Confianza: {confianza} ({prob_max:.0%})</small></div>',
                        unsafe_allow_html=True)

        with col_res2:
            st.markdown("**Probabilidad por categoria:**")
            fig_proba = px.bar(
                x=list(clases), y=list(proba),
                color=list(clases),
                color_discrete_map={'Alta': '#2d6a4f', 'Media': '#e9c46a', 'Baja': '#e63946'},
                labels={'x': 'Categoria', 'y': 'Probabilidad'},
                title='Probabilidad por Categoria'
            )
            fig_proba.update_layout(showlegend=False, yaxis_tickformat='.0%')
            st.plotly_chart(fig_proba, use_container_width=True)

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
        {'Region': 'Pampeana', 'Provincias': 'Buenos Aires, Cordoba, Santa Fe, La Pampa',
         'Variedad': 'Lolium perenne', 'Mes optimo': 'Abril', 'Fertilizante': 'NPK 15-15-15',
         'Riego': 'Diario', 'Temperatura': '10-18 °C', 'Nota': 'Zona optima del pais para resiembra invernal'},
        {'Region': 'Metropolitana', 'Provincias': 'Ciudad Autonoma de Buenos Aires',
         'Variedad': 'Lolium perenne', 'Mes optimo': 'Abril', 'Fertilizante': 'NPK 15-15-15',
         'Riego': 'Diario', 'Temperatura': '10-18 °C', 'Nota': 'Mismas condiciones que la zona Pampeana'},
        {'Region': 'Cuyo', 'Provincias': 'Mendoza, San Juan, San Luis',
         'Variedad': 'Lolium perenne', 'Mes optimo': 'Mayo', 'Fertilizante': 'NPK 15-15-15',
         'Riego': 'Diario', 'Temperatura': '8-16 °C', 'Nota': 'Clima seco: el riego diario es indispensable'},
        {'Region': 'NEA', 'Provincias': 'Entre Rios, Corrientes, Misiones, Chaco, Formosa',
         'Variedad': 'Lolium perenne', 'Mes optimo': 'Abril', 'Fertilizante': 'NPK 15-15-15',
         'Riego': 'Cada 2 dias', 'Temperatura': '12-20 °C', 'Nota': 'Alta humedad: monitorear presencia de hongos'},
        {'Region': 'NOA', 'Provincias': 'Tucuman, Salta, Jujuy, Catamarca, La Rioja, Santiago del Estero',
         'Variedad': 'Lolium multiflorum', 'Mes optimo': 'Mayo - Junio', 'Fertilizante': 'NPK 20-10-10',
         'Riego': 'Diario', 'Temperatura': '12-20 °C', 'Nota': 'El Anual tolera mejor las temperaturas elevadas'},
        {'Region': 'Patagonia', 'Provincias': 'Neuquen, Rio Negro, Chubut, Santa Cruz, Tierra del Fuego',
         'Variedad': 'Lolium perenne', 'Mes optimo': 'Marzo - Abril', 'Fertilizante': 'NPK 15-15-15',
         'Riego': 'Cada 2 dias', 'Temperatura': '6-14 °C', 'Nota': 'Anticipar la siembra por las heladas tempranas'},
    ])

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
