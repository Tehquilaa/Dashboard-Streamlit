import streamlit as st
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import json
import os
import streamlit.components.v1 as components 
from components.headers import get_main_title, get_intro_highlight, get_section_header, get_introduccion, get_antecedentes, get_datos_experimentales
from components.viz.trajectory_viz import display_trajectory_visualization


# Configuración de la página con tema personalizado
st.set_page_config(
    page_title="Predicción de la Dinámica del Balín",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

st.sidebar.caption("© 2025 | Desarrollado por Aldo")

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
    st.markdown(get_introduccion(), unsafe_allow_html=True)

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
    st.markdown(get_antecedentes(), unsafe_allow_html=True)

with antec_viz_col:
    
    display_trajectory_visualization(antec_viz_col)

# Sección 3: Datos Experimentales (con video a la derecha)
st.markdown(get_section_header("3", "🧪", "Datos Experimentales"), unsafe_allow_html=True)

# Nuevas columnas específicas para datos experimentales y video
datos_col, video_col = st.columns([3, 2])

with datos_col:
    st.markdown(get_datos_experimentales(), unsafe_allow_html=True)

with video_col:
    # Video y visualizaciones en la columna derecha
    st.markdown("<h3 style='text-align:center; color:#4b6cb7;'>Demo del Experimento</h3>", unsafe_allow_html=True)
    
  
    st.video(
        "video/experimento.mp4",
        format="video/mp4", 
        start_time=3,
    ) 
    

    with st.expander("ℹ️ Más información sobre el experimento"):
        st.markdown("""
        Este video muestra el comportamiento del balín bajo un campo electromagnetico de 12 G y con una frecuencia de 1Hz.
        El experimento fue realizado utilizando un electroimán controlado por corriente alterna,
        permitiendo generar campos magnéticos variables con frecuencias programables.
        """)



def load_footer():
    # Construir ruta absoluta al archivo HTML combinado
    base_path = os.path.dirname(__file__)
    html_path = os.path.join(base_path, "styles", "footer.html")

    footer_content = ""
    try:
        # Cargar el archivo HTML (que ahora incluye el CSS)
        with open(html_path, "r", encoding="utf-8") as f:
            footer_content = f.read()
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del footer en {html_path}")
        return "" # Retorna vacío si no se encuentra
    except Exception as e:
        st.error(f"Error al leer el archivo del footer: {e}")
        return "" # Retorna vacío si hay error

    return footer_content

st.markdown("---")
footer_code = load_footer()
if footer_code:
    components.html(footer_code, height=180,
                   scrolling=False, width=1000, )
    