import streamlit as st
import pandas as pd

st.set_page_config(page_title="Modelos - Predicción Balín", layout="wide")

st.title("Arquitectura de Modelos")

# Sección 4: Arquitectura del Modelo
st.header("4. Arquitectura del Modelo")
st.write("""
Se evaluaron tres tipos de modelos:
- **Redes LSTM:** Ideales para capturar dependencias a largo plazo en series temporales.
- **Redes GRU:** Variante de las LSTM con menor costo computacional.
- **Modelo Denso (Feedforward):** Usado como referencia para comparar el desempeño.
""")
st.write("""
Para la red LSTM se configuraron dos capas con 80 y 29 neuronas, respectivamente, y para la GRU se utilizaron 62 y 47 neuronas.
""")

# Sección 5: Hiperparámetros y Optimización
st.header("5. Hiperparámetros y Optimización")
st.write("Valores obtenidos tras Random Search (se aplicaron estrategias como early stopping para evitar sobreajuste):")
hyperparams = {
    "Modelo LSTM": "Dropout: 0.226, Learning Rate: 0.006, Capas: [80, 29], Optimizer: adamax",
    "Modelo GRU": "Dropout: 0.41, Learning Rate: 0.0004, Capas: [62, 47], Optimizer: adamax",
    "Modelo Denso": "Dropout: 0.57, Learning Rate: 0.007, Capas: [56, 16, 11, 11], Optimizer: adamax"
}

