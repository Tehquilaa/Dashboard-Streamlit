import streamlit as st
import matplotlib.pyplot as plt
from data.simulation import load_training_history

def display_training_history_tab(model_type, key_suffix=""):
    """
    Muestra el historial de entrenamiento para un modelo específico
    
    Args:
        model_type: Tipo de modelo ('lstm', 'gru', o 'dense')
        key_suffix: Sufijo para las claves de los componentes de Streamlit
    """
    # Título según el modelo
    title_map = {
        'lstm': "LSTM", 
        'gru': "GRU", 
        'dense': "Red Densa"
    }
    
    # Obtener datos de entrenamiento
    history_data = load_training_history(model_type=model_type)
    
    st.markdown(f"<h3 style='text-align:center; color:#4b6cb7;'>Evolución del Entrenamiento - {title_map.get(model_type, model_type)}</h3>", 
                unsafe_allow_html=True)
    
    # Crear figura con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Añadir las curvas de pérdida
    ax.plot(history_data['epochs'], history_data['loss'], label="Pérdida (entrenamiento)", color="blue")
    ax.plot(history_data['epochs'], history_data['val_loss'], label="Pérdida (validación)", color="orange")
    
    # Configurar el diseño de la gráfica
    ax.set_title("Historia del Entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error (MSE)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)

def display_training_history_section():
    """Muestra la sección completa de historiales de entrenamiento con pestañas"""
    # Crear pestañas para cada modelo
    model_tabs = st.tabs(["Modelo LSTM", "Modelo GRU", "Modelo Denso"])
    
    with model_tabs[0]:  # LSTM
        display_training_history_tab('lstm')
    
    with model_tabs[1]:  # GRU
        display_training_history_tab('gru')
    
    with model_tabs[2]:  # Denso
        display_training_history_tab('dense')