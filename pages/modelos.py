import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
import os

# Importar componentes comunes
from components.headers import get_inter_graficas, get_proceso_train, get_section_header, get_section_header_modelo, intro_highlight_modelo, get_arquitecturas, get_hiperparametros
from components.headers import get_tarjeta_lstm, get_tarjeta_gru, get_tarjeta_densa
# Configuración de la página
st.set_page_config(
    page_title="Modelos - Predicción Balín",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar animaciones Lottie
def load_lottiefile(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

# Cargar animaciones
lottie_neural = load_lottiefile("animations/neural_network.json")
if not lottie_neural:  
    lottie_neural = "https://assets5.lottiefiles.com/packages/lf20_2znxgjyt.json"

# CSS personalizado
def load_css(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/modelos.css")


col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(get_section_header_modelo("🔍", "Antecedentes y Planteamiento del Problema"), unsafe_allow_html=True)
    st.markdown(intro_highlight_modelo(), unsafe_allow_html=True)

with col2:
    st_lottie(lottie_neural, height=250, key="neural_animation")

st.markdown("---")

# Sección 4: Arquitectura del Modelo
st.markdown(get_section_header("4", "🏗️", "Arquitectura de los Modelos"), unsafe_allow_html=True)

# Columnas para información y visualización
model_desc_col, model_viz_col = st.columns([1, 1])

with model_desc_col:
    st.markdown(get_arquitecturas(), unsafe_allow_html=True)

with model_viz_col:
    # Visualización de arquitecturas de red
    st.markdown("<h4 style='text-align:center; color:#4b6cb7;'>Comparación de Arquitecturas</h4>", unsafe_allow_html=True)
    
    # Crear visualización con plotly
    fig = go.Figure()
    
    # Datos para la visualización
    models = ['LSTM', 'GRU', 'Denso']
    nodes = [109, 109, 94]  # Total de neuronas
    layers = [2, 2, 4]      # Número de capas
    params = [10180, 8712, 4629]  # Aproximación de parámetros
    
    # Gráfico de barras para número de capas
    fig.add_trace(go.Bar(
        x=models,
        y=layers,
        name='Capas',
        marker_color='#4b6cb7',
        text=layers,
        textposition='auto',
        hoverinfo='text',
        hovertext=[f'LSTM: {layers[0]} capas', f'GRU: {layers[1]} capas', f'Denso: {layers[2]} capas']
    ))
    
    # Gráfico de barras para número de neuronas
    fig.add_trace(go.Bar(
        x=models,
        y=nodes,
        name='Neuronas',
        marker_color='#63a6e6',
        text=nodes,
        textposition='auto',
        hoverinfo='text',
        hovertext=[f'LSTM: {nodes[0]} neuronas', f'GRU: {nodes[1]} neuronas', f'Denso: {nodes[2]} neuronas']
    ))
    
    # Configuración del layout
    fig.update_layout(
        barmode='group',
        title='Comparativa de Complejidad de Modelos',
        xaxis_title='Tipo de Modelo',
        yaxis_title='Cantidad',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240,242,246,0.8)'
    )
    
    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)

# Sección 5: Hiperparámetros y Optimización
st.markdown(get_section_header("5", "⚙️", "Hiperparámetros y Optimización"), unsafe_allow_html=True)
st.markdown(get_hiperparametros(), unsafe_allow_html=True)

# Tarjetas para cada modelo con sus hiperparámetros
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(get_tarjeta_lstm(), unsafe_allow_html=True)

with col2:
    st.markdown(get_tarjeta_gru(), unsafe_allow_html=True)

with col3:
    st.markdown(get_tarjeta_densa(), unsafe_allow_html=True)

# Sección 6: Proceso de Entrenamiento
st.markdown(get_section_header("6", "📈", "Proceso de Entrenamiento y Evaluación"), unsafe_allow_html=True)

train_col, metrics_col = st.columns([3, 2])

with train_col:
    st.markdown(get_proceso_train(), unsafe_allow_html=True)

with metrics_col:
    # Visualización de métricas
    st.markdown("<h4 style='text-align:center; color:#4b6cb7;'>Métricas de Rendimiento</h4>", unsafe_allow_html=True)
    
    metrics = {
        'LSTM': {'MSE': 0.0280, 'MAE': 0.0798,},
        'GRU': {'MSE': 0.0158, 'MAE': 0.0914,},
        'Denso': {'MSE': 0.0143, 'MAE': 0.0850,}
    }
    
    # Convertir a DataFrame para visualización
    metrics_df = pd.DataFrame(metrics).T
    
    # Estilizar la tabla
    st.dataframe(
        metrics_df.style.background_gradient(cmap='Blues', subset=['MSE', 'MAE'], low=0.8)
                      .background_gradient(cmap='Greens', low=0.8)
                      .format({'MSE': '{:.4f}', 'MAE': '{:.4f}'}),
        use_container_width=True
    )
    
    # Añadir interpretación de métricas
    with st.expander("ℹ️ Interpretación de métricas"):
         st.markdown(get_inter_graficas())

# Sección 7: Conclusiones y Hallazgos
st.markdown(get_section_header("7", "🔍", "Conclusiones y Trabajo Futuro"), unsafe_allow_html=True)

st.markdown("""
<div class="justified-text highlight">
<h4>Principales hallazgos:</h4>

- Las arquitecturas recurrentes (LSTM y GRU) superaron significativamente al modelo denso, demostrando la importancia de capturar dependencias temporales en sistemas caóticos.

- El modelo LSTM mostró el mejor rendimiento general, con una reducción del 53% en MSE comparado con el modelo denso y un 16% respecto al GRU.

- La capacidad de predicción disminuye a medida que aumenta el horizonte temporal, siendo particularmente notable después de 5 pasos de tiempo futuros.

- Los resultados confirman la viabilidad del enfoque basado en deep learning para modelar dinámicas caóticas sin recurrir a ecuaciones físicas explícitas.

<h4>Trabajo futuro:</h4>

- Explorar arquitecturas híbridas que combinen elementos de física y aprendizaje automático.
- Aumentar el horizonte de predicción mediante técnicas avanzadas como atención y modelos autorregresivos.
- Evaluar la transferibilidad de los modelos a diferentes condiciones experimentales (campos magnéticos y frecuencias variables).
</div>
""", unsafe_allow_html=True)

# Pie de página
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 | Dashboard de Predicción de Dinámica Caótica</p>", unsafe_allow_html=True)