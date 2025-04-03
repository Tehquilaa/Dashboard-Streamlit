import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resultados - Predicción Balín", layout="wide")

st.title("Evaluación y Resultados")

# Selector para secciones específicas
seccion = st.sidebar.radio("Selecciona sección:", ["Evaluación", "Predicción", "Conclusiones"])

if seccion == "Evaluación":
    # Sección 6: Evaluación del Modelo
    st.header("6. Evaluación del Modelo")
    st.write("""
    Se entrenaron los modelos durante 100 épocas con un tamaño de batch de 16. Se implementó early stopping para detener el entrenamiento 
    cuando la métrica de validación (MSE) no mejoraba tras 10 épocas consecutivas.  
    Los errores en el conjunto de prueba son:
    - **Modelo LSTM:** MSE ≈ 0.0280, MAE ≈ 0.0799
    - **Modelo GRU:** MSE ≈ 0.0187, MAE ≈ 0.0892
    - **Modelo Denso:** MSE ≈ 0.0144, MAE ≈ 0.0850
    """)
    st.write("""
    A continuación se muestra una gráfica simulada del error (MSE) durante el entrenamiento:
    """)
    # Gráfica simulada de error
    fig_err, ax_err = plt.subplots()
    epochs = np.arange(1, 101)
    train_error = np.exp(-epochs/30) + 0.02*np.random.randn(100)
    val_error = np.exp(-epochs/28) + 0.025*np.random.randn(100)
    ax_err.plot(epochs, train_error, label='Entrenamiento', color='blue')
    ax_err.plot(epochs, val_error, label='Validación', color='orange')
    ax_err.set_xlabel("Épocas")
    ax_err.set_ylabel("Error Cuadrático Medio (MSE)")
    ax_err.legend()
    st.pyplot(fig_err)

elif seccion == "Predicción":
    # Sección 7: Predicción
    st.header("7. Predicción")
    st.write("""
    Ingrese valores de las características (por ejemplo, coordenadas normalizadas) para obtener una salida indicativa de la trayectoria.
    """)
    valor1 = st.number_input("Valor de la Característica 1", value=0.0)
    valor2 = st.number_input("Valor de la Característica 2", value=0.0)

    if st.button("Predecir"):
        # Simulación simple: la predicción depende de la suma de las características
        pred = "Trayectoria Compleja" if (valor1 + valor2) > 1 else "Trayectoria Simple"
        st.success(f"Predicción simulada: {pred}")

else:  # Conclusiones
    # Sección 8: Conclusiones y Futuras Líneas de Trabajo
    st.header("8. Conclusiones")
    st.write("""
    El uso de redes neuronales para predecir la dinámica de un balín bajo un campo magnético armónico permite abordar 
    sistemas caóticos mediante métodos basados en datos.  
    Los resultados preliminares muestran diferencias en el desempeño entre las arquitecturas evaluadas, y la metodología aplicada 
    permite optimizar los modelos mediante técnicas de hiperparámetros y validación temprana.  
    Futuras investigaciones podrían integrar conocimientos físicos para mejorar la interpretabilidad y precisión de las predicciones.
    """)