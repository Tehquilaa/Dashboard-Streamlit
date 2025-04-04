import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.simulation import generate_trajectory_data, generate_realtime_prediction

def display_trajectory_comparison_section():
    """Muestra la sección de comparación de trayectorias predichas"""
    # Crear columnas para controles y visualización
    pred_controls_col, pred_viz_col = st.columns([1, 2])
    
    # Cargar datos de trayectorias
    trajectory_data = generate_trajectory_data()
    
    with pred_controls_col:
        st.markdown("<h4 style='color:#4b6cb7;'>Controles de Visualización</h4>", unsafe_allow_html=True)
        
        # Selector de modelo
        modelo_seleccionado = st.radio("Selecciona el modelo a visualizar:",
                                   ["LSTM", "GRU", "Red Densa", "Comparar todos"])
        
        # Selector de campos y frecuencias
        campo_idx = st.selectbox("Campo magnético:", 
                              ["12G", "20G", "30G", "50G", "70G"], index=0)
        
        freq_idx = st.slider("Frecuencia:", 
                           min_value=1, max_value=35, value=10, step=1,
                           help="Frecuencia del campo magnético (Hz)")
        
        # Parámetros de visualización
        st.markdown("<h5 style='color:#4b6cb7;'>Opciones de visualización</h5>", unsafe_allow_html=True)
        
        show_points = st.checkbox("Mostrar puntos", value=True)
        show_errors = st.checkbox("Visualizar error", value=True)
        
        # Info sobre la precisión
        with st.expander("ℹ️ Métricas de error"):
            st.markdown(f"""
            **{campo_idx}, {freq_idx}Hz:**
            
            **LSTM:**
            - MSE: 0.0132
            - MAE: 0.0872
            
            **GRU:**
            - MSE: 0.0158
            - MAE: 0.0914
            
            **Red Densa:**
            - MSE: 0.0283
            - MAE: 0.1247
            """)
    
    with pred_viz_col:
        st.markdown("<h4 style='text-align:center; color:#4b6cb7;'>Predicciones de Trayectoria</h4>", unsafe_allow_html=True)
        
        # Crear visualización interactiva con Plotly
        fig = go.Figure()
        
        # Mostrar datos según selección
        if modelo_seleccionado == "LSTM" or modelo_seleccionado == "Comparar todos":
            # Trayectoria real
            fig.add_trace(
                go.Scatter(
                    x=trajectory_data['real']['x'], 
                    y=trajectory_data['real']['y'],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Trayectoria Real'
                )
            )
            
            # Trayectoria predicha por LSTM
            fig.add_trace(
                go.Scatter(
                    x=trajectory_data['lstm']['x'], 
                    y=trajectory_data['lstm']['y'],
                    mode='lines' if not show_points else 'lines+markers',
                    line=dict(color='#4b6cb7', width=2),
                    marker=dict(size=6),
                    name='Predicción LSTM'
                )
            )
            
            # Visualización del error
            if show_errors:
                # Crear líneas para representar el error entre puntos reales y predichos
                for i in range(len(trajectory_data['real']['x'])):
                    if i % 10 == 0:  # Mostrar solo algunos errores para no sobrecargar
                        fig.add_trace(
                            go.Scatter(
                                x=[trajectory_data['real']['x'][i], trajectory_data['lstm']['x'][i]],
                                y=[trajectory_data['real']['y'][i], trajectory_data['lstm']['y'][i]],
                                mode='lines',
                                line=dict(color='rgba(75, 108, 183, 0.3)', width=1),
                                showlegend=False
                            )
                        )
        
        if modelo_seleccionado == "GRU" or modelo_seleccionado == "Comparar todos":
            # Trayectoria real (si no se muestra antes)
            if modelo_seleccionado != "Comparar todos":
                fig.add_trace(
                    go.Scatter(
                        x=trajectory_data['real']['x'], 
                        y=trajectory_data['real']['y'],
                        mode='lines',
                        line=dict(color='black', width=2),
                        name='Trayectoria Real'
                    )
                )
            
            # Trayectoria predicha por GRU
            fig.add_trace(
                go.Scatter(
                    x=trajectory_data['gru']['x'], 
                    y=trajectory_data['gru']['y'],
                    mode='lines' if not show_points else 'lines+markers',
                    line=dict(color='#ff7043', width=2),
                    marker=dict(size=6),
                    name='Predicción GRU'
                )
            )
            
            # Visualización del error para GRU
            if show_errors and modelo_seleccionado != "Comparar todos":
                for i in range(len(trajectory_data['real']['x'])):
                    if i % 10 == 0:
                        fig.add_trace(
                            go.Scatter(
                                x=[trajectory_data['real']['x'][i], trajectory_data['gru']['x'][i]],
                                y=[trajectory_data['real']['y'][i], trajectory_data['gru']['y'][i]],
                                mode='lines',
                                line=dict(color='rgba(255, 112, 67, 0.3)', width=1),
                                showlegend=False
                            )
                        )
        
        if modelo_seleccionado == "Red Densa" or modelo_seleccionado == "Comparar todos":
            # Trayectoria real (si no se muestra antes)
            if modelo_seleccionado != "Comparar todos":
                fig.add_trace(
                    go.Scatter(
                        x=trajectory_data['real']['x'], 
                        y=trajectory_data['real']['y'],
                        mode='lines',
                        line=dict(color='black', width=2),
                        name='Trayectoria Real'
                    )
                )
            
            # Trayectoria predicha por Red Densa
            fig.add_trace(
                go.Scatter(
                    x=trajectory_data['dense']['x'], 
                    y=trajectory_data['dense']['y'],
                    mode='lines' if not show_points else 'lines+markers',
                    line=dict(color='#4caf50', width=2),
                    marker=dict(size=6),
                    name='Predicción Red Densa'
                )
            )
            
            # Visualización del error para Red Densa
            if show_errors and modelo_seleccionado != "Comparar todos":
                for i in range(len(trajectory_data['real']['x'])):
                    if i % 10 == 0:
                        fig.add_trace(
                            go.Scatter(
                                x=[trajectory_data['real']['x'][i], trajectory_data['dense']['x'][i]],
                                y=[trajectory_data['real']['y'][i], trajectory_data['dense']['y'][i]],
                                mode='lines',
                                line=dict(color='rgba(76, 175, 80, 0.3)', width=1),
                                showlegend=False
                            )
                        )
        
        # Configuración del layout
        fig.update_layout(
            title=f"Trayectoria predicha ({campo_idx}, {freq_idx}Hz)",
            xaxis_title="Posición X",
            yaxis_title="Posición Y",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            hovermode='closest',
            plot_bgcolor='rgba(240,242,246,0.8)',
            # Mantener la misma escala en ambos ejes para mejor visualización
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        )
        
        # Mostrar figura
        st.plotly_chart(fig, use_container_width=True)
    
        # Información sobre la visualización
        st.caption("La visualización muestra la trayectoria real (negro) y las trayectorias predichas por los modelos. Las líneas entre puntos representan el error de predicción en cada punto.")

def display_realtime_prediction_section():
    """Muestra la sección de predicción en tiempo real"""
    st.markdown("""
    <div class="justified-text">
    Para entender mejor cómo funcionan los modelos en la práctica, puedes probar a predecir la trayectoria
    ingresando coordenadas iniciales. El sistema tomará estos valores y generará una secuencia
    que representa la trayectoria predicha del balín.
    </div>
    """, unsafe_allow_html=True)
    
    # Crear columnas para los inputs y resultados
    input_col, result_col = st.columns([1, 2])
    
    with input_col:
        st.markdown("<h4 style='color:#4b6cb7;'>Parámetros de entrada</h4>", unsafe_allow_html=True)
        
        st.markdown("**Condiciones iniciales (secuencia):**")
        
        # Crear 5 filas de coordenadas X, Y de entrada
        coords = []
        for i in range(5):
            col_x, col_y = st.columns(2)
            with col_x:
                x = st.number_input(f"X{i+1}", value=round(float(np.sin(i*0.5)), 2), format="%.2f", key=f"x{i}")
            with col_y:
                y = st.number_input(f"Y{i+1}", value=round(float(np.cos(i*0.5)), 2), format="%.2f", key=f"y{i}")
            coords.append((x, y))
        
        # Seleccionar modelo para la predicción
        prediction_model = st.selectbox("Modelo a utilizar:",
                                     ["LSTM", "GRU", "Red Densa"],
                                     index=0)
        
        # Configuración de la predicción
        st.markdown("**Configuración:**")
        steps = st.slider("Pasos a predecir:", 5, 30, 10)
        
        # Botón para ejecutar la predicción
        predict_btn = st.button("Generar Predicción", type="primary")
    
    with result_col:
        st.markdown("<h4 style='text-align:center; color:#4b6cb7;'>Resultado de la predicción</h4>", unsafe_allow_html=True)
        
        # Si se presiona el botón, mostrar la predicción
        if predict_btn:
            # Obtener predicciones
            prediction_data = generate_realtime_prediction(coords, steps, prediction_model)
            
            # Visualizar la trayectoria predicha
            fig = go.Figure()
            
            # Puntos iniciales (entrada)
            fig.add_trace(
                go.Scatter(
                    x=prediction_data['initial']['x'],
                    y=prediction_data['initial']['y'],
                    mode='lines+markers',
                    line=dict(color='black', width=2),
                    marker=dict(size=8, color='black'),
                    name='Puntos iniciales'
                )
            )
            
            # Trayectoria predicha
            model_color = "#4b6cb7" if prediction_model == "LSTM" else "#ff7043" if prediction_model == "GRU" else "#4caf50"
            
            fig.add_trace(
                go.Scatter(
                    x=prediction_data['predicted']['x'],
                    y=prediction_data['predicted']['y'],
                    mode='lines+markers',
                    line=dict(color=model_color, width=2),
                    marker=dict(size=8, 
                                color=list(range(len(prediction_data['predicted']['x']))),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(
                                    title=dict(
                                        text="Tiempo",
                                        side="right"
                                    )
                                )),
                    name=f'Predicción {prediction_model}'
                )
            )
            
            # Configuración del layout
            fig.update_layout(
                title=f"Trayectoria predicha con modelo {prediction_model}",
                xaxis_title="Posición X",
                yaxis_title="Posición Y",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400,
                hovermode='closest',
                plot_bgcolor='rgba(240,242,246,0.8)',
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            # Mostrar figura
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla con valores predichos
            st.markdown("<h5 style='color:#4b6cb7;'>Valores predichos:</h5>", unsafe_allow_html=True)
            
            # Crear DataFrame con los resultados
            df_results = pd.DataFrame({
                'Paso': [f"t+{i+1}" for i in range(steps)],
                'Coordenada X': [round(x, 3) for x in prediction_data['predicted']['x']],
                'Coordenada Y': [round(y, 3) for y in prediction_data['predicted']['y']],
                'Distancia al origen': [round(np.sqrt(x**2 + y**2), 3) for x, y in 
                                       zip(prediction_data['predicted']['x'], 
                                           prediction_data['predicted']['y'])],
            })
            
            st.dataframe(df_results, use_container_width=True)
        else:
            # Mensaje cuando no se ha hecho una predicción
            st.info("Ingresa los datos iniciales y haz clic en 'Generar Predicción' para ver los resultados.")
            
            # Imagen ilustrativa
            st.markdown("""
            <div style="display: flex; justify-content: center; margin-top: 30px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Lorenz_system_r28_s10_b2-6.png/320px-Lorenz_system_r28_s10_b2-6.png" 
                     alt="Ilustración Sistema Caótico" 
                     width="300">
            </div>
            <p style="text-align: center; font-size: 0.8rem; color: gray;">
            Ilustración: Trayectoria en un sistema caótico (Atractor de Lorenz)
            </p>
            """, unsafe_allow_html=True)