import streamlit as st
import json
import os

def load_lottiefile(filepath):
    """Carga un archivo de animación Lottie desde una ruta"""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None

def load_css(css_file):
    """Carga un archivo CSS y lo aplica al dashboard"""
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def apply_default_css():
    """Aplica estilos CSS básicos para el dashboard"""
    st.markdown("""
    <style>
        .justified-text {
            text-align: justify;
            text-justify: inter-word;
        }
        .highlight {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .model-card {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #4b6cb7;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #4b6cb7;
        }
    </style>
    """, unsafe_allow_html=True)
