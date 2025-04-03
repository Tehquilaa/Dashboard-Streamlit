import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from components.headers import get_main_title, get_intro_highlight, get_section_header
from components.trajectory_viz import display_trajectory_visualization

# Configuración de la página con tema personalizado
st.set_page_config(
    page_title="Predicción de la Dinámica del Balín",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔮"
)

# Cargar CSS desde archivo externo
def load_css(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cargar nuestro CSS personalizado
load_css("styles/main.css")

# Función para cargar animaciones Lottie desde archivo local
def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_neural = load_lottiefile("animations/ai_animated.json")
lottie_chaos = load_lottiefile("animations/chaos.json")  

# Layout con dos columnas principales
col1, col2 = st.columns([2, 1])

with col1:
    # Título principal con ícono (usando componente)
    st.markdown(get_main_title(), unsafe_allow_html=True)
    st.markdown(get_intro_highlight(), unsafe_allow_html=True)

with col2:
    # Animación en la columna derecha
    st_lottie(lottie_neural, height=250, key="ia_animation")

st.markdown("---")

# Sección 1: Introducción
st.markdown(get_section_header("1", "📊", "Introducción"), unsafe_allow_html=True)

# Crear columnas para el diseño
col_left, col_text, col_animation = st.columns([0.1, 2.5, 3])

with col_text:
    # Texto centrado en la columna del medio
    st.markdown("""
    <div class="justified-text margin-top-intro">
    Los sistemas caóticos, como el movimiento de un balín bajo la influencia de un campo magnético,
    representan un reto en la predicción debido a su alta sensibilidad a condiciones iniciales.
    Este proyecto utiliza redes neuronales —incluyendo arquitecturas LSTM, GRU y modelos densos—
    para modelar la dinámica caótica a partir de datos experimentales.
    </div>
    """, unsafe_allow_html=True)

with col_animation:
    # Animación en la columna derecha
    st_lottie(
        lottie_chaos,
        height=250,
        key="chaos_animation",
        quality="high"
    )

# Sección 2: Metodología con visualización de trayectoria
st.markdown(get_section_header("2", "🔍", "Antecedentes y Planteamiento del Problema"), unsafe_allow_html=True)

antec_text_col, antec_viz_col = st.columns([3, 5])

with antec_text_col:
    st.markdown("""
    <div class="justified-text margin-top-antecedentes">
    La predicción de trayectorias en sistemas caóticos es complicada por la naturaleza no lineal y la sensibilidad a las condiciones iniciales. Los métodos tradicionales basados en ecuaciones diferenciales tienen limitaciones, lo que ha impulsado el uso de técnicas de machine learning para capturar patrones complejos en datos experimentales.
    El problema se centra en predecir la trayectoria de un balín, cuyos datos experimentales comprenden 1020 puntos por muestra, con 195 muestras distribuidas en 5 carpetas (12G, 20G, 30G, 50G y 70G) y frecuencias entre 1Hz y 35Hz.
    <br><br>
    En este proyecto, adoptamos un enfoque basado en deep learning, comparando diferentes arquitecturas de redes neuronales.
    </div>
    """, unsafe_allow_html=True)

with antec_viz_col:
    
    display_trajectory_visualization(antec_viz_col)

# Sección 3: Datos Experimentales (con video a la derecha)
st.markdown(get_section_header("3", "🧪", "Datos Experimentales"), unsafe_allow_html=True)

# Nuevas columnas específicas para datos experimentales y video
datos_col, video_col = st.columns([3, 2])

with datos_col:
    st.markdown("""
    <div class="justified-text margin-top-intro">
    Los datos provienen de un experimento físico controlado y se estructuran de la siguiente forma:
    <ul>
      <li><strong style="color:#4b6cb7;">Total de muestras:</strong> 195 (35 por cada campo magnético).</li>
      <li><strong style="color:#4b6cb7;">Puntos por muestra:</strong> 1020 registros de coordenadas (XM, YM).</li>
      <li><strong style="color:#4b6cb7;">Variables experimentales:</strong> Campo Magnético (12G a 70G) y Frecuencia (1Hz a 35Hz).</li>
    </ul>
    <br>
    El preprocesamiento incluye:
    <ul>
      <li>Lectura y consolidación de archivos txt (cada uno con 2 columnas: XM y YM).</li>
      <li>Extracción de metadatos (campo magnético y frecuencia, a partir del nombre de carpeta y archivo).</li>
      <li>Normalización de los datos con la técnica Min-Max, preservando la forma de la distribución.</li>
      <li>División del dataset en subconjuntos: entrenamiento (70%), validación (20%) y prueba (10%).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with video_col:
    # Video y visualizaciones en la columna derecha
    st.markdown("<h3 style='text-align:center; color:#4b6cb7;'>Demo del Experimento</h3>", unsafe_allow_html=True)
    
  
    st.video(
        "video/experimento.mp4",
        format="video/mp4", 
        autoplay=True,
        loop=True,
        start_time=3,
    ) 
    

    with st.expander("ℹ️ Más información sobre el experimento"):
        st.markdown("""
        Este video muestra el comportamiento del balín bajo un campo electromagnetico de 12 G y con una frecuencia de 1Hz.
        El experimento fue realizado utilizando un electroimán controlado por corriente alterna,
        permitiendo generar campos magnéticos variables con frecuencias programables.
        """)

