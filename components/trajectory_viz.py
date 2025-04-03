import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Función para cargar los datos del CSV
@st.cache_data  # Cachear para mejorar rendimiento
def load_trajectory_data(file_path="data/12G-1Hz.csv"):
    try:
        # Intenta cargar desde el archivo CSV
        if os.path.exists(file_path):
            # Carga el CSV con pandas
            data = pd.read_csv(file_path)
            return data["XM"].values, data["YM"].values
        else:
            return generate_sample_data(1020)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return generate_sample_data(1020)

# Función para generar datos de muestra si no hay reales
def generate_sample_data(n_points=1020):
    t = np.linspace(0, 8 * np.pi, n_points)
    r = 0.2 * (1 + np.sin(t)) + 0.1 * np.random.randn(n_points)
    x = r * np.cos(t) 
    y = r * np.sin(t)
    return x, y

# Función principal para visualizar trayectoria en Streamlit
def display_trajectory_visualization(container):
    # Título de la visualización
    container.markdown("<h4 style='text-align:center; color:#4b6cb7;'>Visualización de la Trayectoria</h4>", unsafe_allow_html=True)
    
    # Control deslizante para ajustar cuántos puntos mostrar
    num_points = container.slider("Puntos a visualizar", 10, 1020, 200, 10)
    
    # Cargar datos
    x_coords, y_coords = load_trajectory_data()
    
    # Crear la visualización
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Mostrar solo el número de puntos seleccionado
    points_to_show = min(num_points, len(x_coords))
    
    # Añadir línea suave de trayectoria
    ax.plot(x_coords[:points_to_show], y_coords[:points_to_show], 
            'k-', alpha=0.3, linewidth=0.8)
    
    # Colorear puntos según el tiempo
    scatter = ax.scatter(x_coords[:points_to_show], y_coords[:points_to_show], 
                         c=range(points_to_show), cmap='viridis', 
                         s=15, alpha=0.8)
    
    # Marcar puntos de inicio y actual
    if points_to_show > 0:
        # Punto inicial
        ax.scatter(x_coords[0], y_coords[0], 
                   color='green', s=120, label='Inicio', 
                   zorder=5, edgecolor='white')
        
        # Punto actual
        ax.scatter(x_coords[points_to_show-1], y_coords[points_to_show-1], 
                   color='red', s=120, label='Posición actual', 
                   zorder=5, edgecolor='white')
    
    # Configuración del gráfico
    ax.set_title(f"Visualización de {points_to_show} puntos de la trayectoria")
    ax.set_xlabel("Posición X")
    ax.set_ylabel("Posición Y")
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right')
    
    # Ajustar el gráfico para mostrar bien la proporción
    ax.set_aspect('equal')
    plt.tight_layout()
    
    # Mostrar el gráfico en Streamlit
    container.pyplot(fig)
    
    # Información adicional sobre los datos
    with container.expander("ℹ️ Detalles sobre la visualización"):
        container.markdown("""
        Esta visualización muestra la trayectoria del balín bajo un campo magnético de 12G y frecuencia de 1Hz.
        - **Punto verde**: Posición inicial
        - **Punto rojo**: Posición actual según el número de puntos seleccionados
        - **Gradiente de colores**: Representa el tiempo (azul → amarillo)
        
        Usa el control deslizante para ver cómo se construye la trayectoria punto a punto.
        """)