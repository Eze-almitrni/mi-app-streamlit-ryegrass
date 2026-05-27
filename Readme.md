# 🌱 Resiembra Invernal de Rye Grass — Argentina 2024/2025

**Análisis Exploratorio de Datos y Modelo Predictivo de Calidad de Césped**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📋 Descripción

En Argentina, durante el invierno el césped base de las canchas de fútbol entra en dormición vegetativa. Para mantener las condiciones de juego, los campos deportivos realizan una **resiembra invernal con Rye Grass** (*Lolium spp.*).

Este proyecto analiza los factores agronómicos y climáticos que determinan la calidad de esa resiembra, usando EDA y Machine Learning.

## 🚀 Ver la app en Streamlit Cloud

👉 **[Abrir App](https://share.streamlit.io)** *(reemplazá con tu URL de Streamlit Cloud)*

## 🗂️ Estructura del repositorio

```
ryegrass-campos-deportivos/
├── app.py # App principal Streamlit
├── ryegrass_limpio.csv # Dataset limpio (500 observaciones)
├── ryegrass_argentina_final.csv # Dataset original
├── requirements.txt # Dependencias Python
├── .streamlit/
│ └── config.toml # Configuración de tema Streamlit
└── README.md # Este archivo
```

## 📊 Secciones de la App

| Sección | Contenido |
|---|---|
| 🏠 Inicio | Overview del proyecto, métricas clave, recomendaciones por región |
| 🔍 EDA | 6 tabs con 12+ gráficos exploratorios |
| 🤖 Modelo ML | Comparación de 4 modelos, importancia de variables, matriz de confusión |
| 🎯 Predictor | Formulario interactivo para predecir calidad de resiembra |
| 📋 Conclusiones | Hallazgos, limitaciones y fuentes |

## 🤖 Modelo de Machine Learning

- **Algoritmo:** Random Forest Classifier (100 árboles, profundidad máx. 10)
- **Variables:** 14 features (climáticas, agronómicas y edáficas)
- **Objetivo:** Categoría de calidad (Alta / Media / Baja)
- **Evaluación:** Accuracy, clasificación report, validación cruzada 5-Fold, matriz de confusión
- **Comparación:** Dummy Classifier, Regresión Logística, Random Forest, Gradient Boosting

## 📦 Instalación local

```bash
git clone https://github.com/TU_USUARIO/ryegrass-campos-deportivos.git
cd ryegrass-campos-deportivos
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy en Streamlit Cloud

1. Subí el repositorio a GitHub (público)
2. Entrá a [share.streamlit.io](https://share.streamlit.io)
3. Conectá tu cuenta de GitHub
4. Seleccioná el repositorio y `app.py` como archivo principal
5. ¡Listo! La app se despliega automáticamente

## 📈 Dataset

| Característica | Detalle |
|---|---|
| Registros | 500 observaciones (limpio) |
| Provincias | 10 argentinas |
| Variables | 27 columnas |
| Variable objetivo | Categoría Calidad (Alta / Media / Baja) |
| Período | Temporada invernal 2024/2025 (Marzo–Agosto) |

## ⚠️ Limitaciones


1. Cobertura de 10/24 provincias argentinas
2. Datos climáticos históricos (SMN 1981-2010)
3. Desbalance de clases: 90% Alta, 10% Media, 0% Baja

## 📚 Fuentes

- Características agronómicas: [turfland.com.ar](https://turfland.com.ar) y [picasso.com.ar](https://picasso.com.ar)
- Datos climáticos: [SMN Argentina](https://www.smn.gob.ar)
- Tipos de suelo: [INTA Argentina](https://www.inta.gob.ar)
- Estándar NTEP: [ntep.org](https://www.ntep.org)

---

**Autor:** Ezequias | Ciencia de Datos 2026
