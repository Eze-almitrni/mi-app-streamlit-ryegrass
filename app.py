# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------
st.set_page_config(
    page_title="Análisis Ryegrass - Argentina",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌿 Análisis Interactivo de Resiembra Invernal de Ryegrass")
st.subheader("Producción y Calidad de Campos Deportivos en Argentina")

# --------------------------
# CARGA DE DATOS
# --------------------------
st.sidebar.header("📂 Carga de Datos")

# Cargar archivos CSV
try:
    # Cargar los archivos que tenés en tu repositorio
    df_limpio = pd.read_csv("ryegrass_limpio.csv")
    df_completo = pd.read_csv("ryegrass_argentina_final.csv")
    
    st.sidebar.success("✅ Datos cargados correctamente")
    
except FileNotFoundError:
    st.error("❌ Error: No se encontraron los archivos CSV. Asegurate de que estén en el repositorio.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")
    st.stop()

# --------------------------
# MENU DE NAVEGACIÓN
# --------------------------
menu = st.sidebar.selectbox(
    "📋 Seleccionar sección",
    ["Inicio", "Análisis Exploratorio", "Visualizaciones", "Predicción de Calidad", "Información"]
)

# --------------------------
# SECCIÓN 1: INICIO
# --------------------------
if menu == "Inicio":
    st.header("Bienvenido al Proyecto")
    st.write("""
    Esta aplicación permite analizar datos de producción y calidad de Ryegrass en diferentes regiones de Argentina.
    Podrás explorar información, ver gráficos interactivos y predecir la calidad del cultivo según diferentes variables.
    """)
    
    # Mostrar resumen de datos
    st.subheader("📊 Resumen de los Datos")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de registros", len(df_limpio))
    with col2:
        st.metric("Variedades distintas", df_limpio["Variedad_Comun"].nunique())
    with col3:
        st.metric("Provincias", df_limpio["Provincia"].nunique())
    
    st.dataframe(df_limpio.head())

# --------------------------
# SECCIÓN 2: ANÁLISIS EXPLORATORIO
# --------------------------
elif menu == "Análisis Exploratorio":
    st.header("🔍 Análisis Exploratorio de Datos")
    
    # Filtros interactivos
    col1, col2 = st.columns(2)
    with col1:
        provincia = st.selectbox("Seleccionar Provincia", df_limpio["Provincia"].unique())
    with col2:
        variedad = st.selectbox("Seleccionar Variedad", df_limpio["Variedad_Comun"].unique())
    
    # Filtrar datos
    df_filtrado = df_limpio[(df_limpio["Provincia"] == provincia) & (df_limpio["Variedad_Comun"] == variedad)]
    
    st.subheader(f"Datos para {variedad} en {provincia}")
    st.dataframe(df_filtrado)
    
    # Estadísticas descriptivas
    st.subheader("📈 Estadísticas Descriptivas")
    st.dataframe(df_filtrado.describe())

# --------------------------
# SECCIÓN 3: VISUALIZACIONES
# --------------------------
elif menu == "Visualizaciones":
    st.header("📊 Visualizaciones de Datos")
    
    tipo_grafico = st.selectbox(
        "Seleccionar tipo de gráfico",
        ["Distribución de Calidad", "Calidad por Provincia", "Relación Temperatura-Calidad", "Relación Precipitación-Calidad"]
    )
    
    # Configurar estilo de gráficos
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if tipo_grafico == "Distribución de Calidad":
        st.subheader("Distribución de la Calidad Visual")
        sns.countplot(data=df_limpio, x="Calidad_Visual", palette="viridis", ax=ax)
        ax.set_xlabel("Nivel de Calidad")
        ax.set_ylabel("Cantidad de Registros")
        
    elif tipo_grafico == "Calidad por Provincia":
        st.subheader("Calidad Promedio por Provincia")
        calidad_provincia = df_limpio.groupby("Provincia")["Calidad_Visual"].mean().sort_values()
        calidad_provincia.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_xlabel("Provincia")
        ax.set_ylabel("Calidad Promedio")
        plt.xticks(rotation=45)
        
    elif tipo_grafico == "Relación Temperatura-Calidad":
        st.subheader("Relación entre Temperatura y Calidad")
        sns.scatterplot(data=df_limpio, x="Temperatura_C", y="Calidad_Visual", hue="Variedad_Comun", s=100, ax=ax)
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Calidad Visual")
        
    elif tipo_grafico == "Relación Precipitación-Calidad":
        st.subheader("Relación entre Precipitación y Calidad")
        sns.scatterplot(data=df_limpio, x="Precipitacion_mm", y="Calidad_Visual", hue="Tipo_Suelo", s=100, ax=ax)
        ax.set_xlabel("Precipitación (mm)")
        ax.set_ylabel("Calidad Visual")
    
    # Mostrar gráfico
    st.pyplot(fig)

# --------------------------
# SECCIÓN 4: PREDICCIÓN DE CALIDAD
# --------------------------
elif menu == "Predicción de Calidad":
    st.header("🤖 Predicción de Calidad del Cultivo")
    
    # Preparar datos para el modelo
    # Codificar variables categóricas
    le_provincia = LabelEncoder()
    le_variedad = LabelEncoder()
    le_calidad = LabelEncoder()
    
    df_limpio["Provincia_Cod"] = le_provincia.fit_transform(df_limpio["Provincia"])
    df_limpio["Variedad_Cod"] = le_variedad.fit_transform(df_limpio["Variedad_Comun"])
    df_limpio["Calidad_Cod"] = le_calidad.fit_transform(df_limpio["Calidad_Visual"])
    
    # Seleccionar características y variable objetivo
    caracteristicas = [
        "Temperatura_C", "Precipitacion_mm", "Humedad_Pct", 
        "N_Fertilizante", "P_Fertilizante", "K_Fertilizante",
        "Frecuencia_Riego", "Densidad_Siembra_kg_100m2",
        "Provincia_Cod", "Variedad_Cod"
    ]
    objetivo = "Calidad_Cod"
    
    X = df_limpio[caracteristicas]
    y = df_limpio[objetivo]
    
    # Dividir datos en entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    
    # Mostrar precisión del modelo
    precision = accuracy_score(y_prueba, modelo.predict(X_prueba))
    st.metric("Precisión del Modelo", f"{precision:.2%}")
    
    # Formulario para ingresar datos
    st.subheader("Ingresar datos para predecir")
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperatura (°C)", min_value=0.0, max_value=35.0, value=18.0)
        precipitacion = st.slider("Precipitación (mm)", min_value=0.0, max_value=300.0, value=80.0)
        humedad = st.slider("Humedad (%)", min_value=0.0, max_value=100.0, value=70.0)
        n_fertilizante = st.slider("Nitrógeno (kg/ha)", min_value=0, max_value=60, value=25)
    
    with col2:
        p_fertilizante = st.slider("Fósforo (kg/ha)", min_value=0, max_value=40, value=12)
        k_fertilizante = st.slider("Potasio (kg/ha)", min_value=0, max_value=40, value=10)
        riego = st.selectbox("Frecuencia de Riego", df_limpio["Frecuencia_Riego"].unique())
        densidad = st.slider("Densidad de Siembra (kg/100m²)", min_value=1.0, max_value=7.0, value=4.0)
    
    # Codificar variables categóricas para la predicción
    provincia_cod = le_provincia.transform([df_limpio["Provincia"].iloc[0]])[0]
    variedad_cod = le_variedad.transform([df_limpio["Variedad_Comun"].iloc[0]])[0]
    
    # Crear datos de entrada
    datos_entrada = pd.DataFrame([[
        temp, precipitacion, humedad, n_fertilizante, p_fertilizante, k_fertilizante,
        riego, densidad, provincia_cod, variedad_cod
    ]], columns=caracteristicas)
    
    # Botón para predecir
    if st.button("Predecir Calidad"):
        prediccion_cod = modelo.predict(datos_entrada)[0]
        prediccion_calidad = le_calidad.inverse_transform([prediccion_cod])[0]
        
        st.success(f"✅ La calidad predicha es: **{prediccion_calidad}**")
        
        # Mostrar probabilidades
        probabilidades = modelo.predict_proba(datos_entrada)[0]
        clases = le_calidad.classes_
        
        st.subheader("Probabilidades por nivel de calidad:")
        for clase, prob in zip(clases, probabilidades):
            st.write(f"- {clase}: {prob:.2%}")

# --------------------------
# SECCIÓN 5: INFORMACIÓN
# --------------------------
elif menu == "Información":
    st.header("ℹ️ Información del Proyecto")
    st.write("""
    **Tema:** Resiembra Invernal de Ryegrass en Campos Deportivos de Argentina
    **Objetivo:** Analizar factores que influyen en la calidad del cultivo y predecir resultados
    
    **Variables incluidas:**
    - Datos climáticos: Temperatura, Precipitación, Humedad
    - Datos de cultivo: Fertilizantes, Riego, Densidad de siembra
    - Datos geográficos: Provincia, Tipo de suelo
    - Resultados: Calidad visual del cultivo
    """)
    
    st.subheader("📁 Archivos en el repositorio:")
    st.write("""
    - `ryegrass_limpio.csv`: Datos limpios y organizados
    - `ryegrass_argentina_final.csv`: Datos completos con información detallada
    - `app.py`: Código de la aplicación
    - `requirements.txt`: Librerías necesarias para ejecutar la app
    """)

# Pie de página
st.markdown("---")
st.markdown("Proyecto realizado para la cátedra de Ciencia de Datos - Argentina")
