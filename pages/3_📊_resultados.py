import streamlit as st
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components
import os
from components.headers import get_section_header
from components.utils import load_lottiefile, load_css, apply_default_css
from components.viz.training_viz import display_training_history_section
from components.viz.prediction_viz import display_trajectory_comparison_section, display_realtime_prediction_section

# Configuración de la página
st.set_page_config(
    page_title="Resultados - Predicción Balín",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)
st.sidebar.caption("© 2025 | Desarrollado por Aldo")

# Cargar estilos y configurar aspecto visual
load_css("styles/resultados.css")
apply_default_css()

# Cargar animaciones
lottie_results = load_lottiefile("animations/graphic.json")
if not lottie_results:  # URL de respaldo
    lottie_results = "https://assets5.lottiefiles.com/packages/lf20_2znxgjyt.json"

# Encabezado principal con animación
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1>📊 Evaluación y Resultados</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="justified-text highlight">
    Esta sección presenta los resultados detallados del entrenamiento y evaluación de los modelos de deep learning.
    Se incluyen visualizaciones interactivas del proceso de entrenamiento, predicciones en tiempo real y
    métricas comparativas entre las distintas arquitecturas implementadas.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_results, height=250, key="results_animation")

st.markdown("---")

# Sección 1: Historia del Entrenamiento
st.markdown(get_section_header("1", "📈", "Historial de Entrenamiento"), unsafe_allow_html=True)
display_training_history_section()

# Sección 2: Visualización de Trayectorias Predichas
st.markdown(get_section_header("2", "🎯", "Visualización de Trayectorias Predichas"), unsafe_allow_html=True)
display_trajectory_comparison_section()

# Sección 3: final
st.markdown(get_section_header("3", "🔍", "Conclusiones y Trabajo Futuro"), unsafe_allow_html=True)

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

def load_footer():
    # Obtener el directorio de la página actual (pages/)
    current_dir = os.path.dirname(__file__)
    
    # Subir un nivel para llegar al directorio raíz del proyecto
    project_root = os.path.dirname(current_dir)
    
    # Construir la ruta al footer desde el directorio raíz
    html_path = os.path.join(project_root, "styles", "footer.html")

    footer_content = ""
    try:
        # Cargar el archivo HTML
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