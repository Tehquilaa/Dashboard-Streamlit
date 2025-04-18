import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
import time
from streamlit_lottie import st_lottie
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components



# Configuración de la página
st.set_page_config(
    page_title="Predicciones - Predicción Balín",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔮"
)

st.sidebar.caption("© 2025 | Desarrollado por Aldo")

# Estilos CSS personalizados
def load_css(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cargar nuestro CSS personalizado
load_css("styles/prediccion.css")

# Función para cargar animaciones Lottie
def load_lottieurl(path):
    with open(path, "r") as f:
        return json.load(f)

# Cargar animaciones
try:
    lottie_model = load_lottieurl("assets/model_animation.json")
    lottie_prediction = load_lottieurl("assets/prediction_animation.json")
except:
    lottie_model = None
    lottie_prediction = None

# Título principal
st.markdown("<h1 class='main-header'>🔮 Predicción de Trayectorias</h1>", unsafe_allow_html=True)

# Cargar escaladores globales
with st.spinner("Inicializando componentes..."):
    try:
        scaler_X = joblib.load('data/parametros/scaler_X.pkl')
        scaler_Y = joblib.load('data/parametros/scaler_Y.pkl')
        SCALERS_LOADED = True
        st.markdown("<div class='success-box'>✅ Escaladores cargados correctamente</div>", unsafe_allow_html=True)
    except Exception as e:
        SCALERS_LOADED = False
        st.error(f"Error al cargar los escaladores: {e}")

# Función personalizada para compatibilidad
def mse(y_true, y_pred):
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

# Interfaz con pestañas
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("<h2 class='section-header'>1. 📂 Modelo</h2>", unsafe_allow_html=True)
    if lottie_model:
        st_lottie(lottie_model, height=150, key="model_animation")
    
    model_options = ["LSTM", "GRU", "Dense"]
    model_choice = st.selectbox("Seleccione arquitectura:", model_options, help="Tipo de red neuronal para predicción")
    
    # Mapeo de modelos
    model_files = {
        "LSTM": "models/lstm_model.h5",
        "GRU": "models/gru_model.h5",
        "Dense": "models/dense_model.h5"
    }
    
    load_model_btn = st.button("📥 Cargar Modelo", type="primary")
    
    if "model_info" not in st.session_state:
        st.session_state["model_info"] = None
    
    if load_model_btn:
        with st.spinner(f"Cargando modelo {model_choice}..."):
            try:
                model = tf.keras.models.load_model(model_files[model_choice], custom_objects={'mse': mse})
                st.session_state["model"] = model
                
                # Guardar información del modelo para mostrarla
                st.session_state["model_info"] = {
                    "tipo": model_choice,
                    "capas": len(model.layers),
                    "parámetros": model.count_params(),
                    "formato_entrada": model.input_shape,
                    "formato_salida": model.output_shape
                }
                
                st.success(f"✅ Modelo {model_choice} cargado exitosamente")
            except Exception as e:
                st.error(f"Error al cargar el modelo: {e}")
    
    # Mostrar información del modelo si está cargado
    if st.session_state["model_info"]:
        info = st.session_state["model_info"]
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown(f"**Modelo:** {info['tipo']}")
        st.markdown(f"**Capas:** {info['capas']}")
        st.markdown(f"**Parámetros:** {info['parámetros']:,}")
        st.markdown(f"**Entrada:** {info['formato_entrada']}")
        st.markdown(f"**Salida:** {info['formato_salida']}")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='section-header'>2. 📊 Datos de Test</h2>", unsafe_allow_html=True)
    
    # Usar directamente los datos de Test
    dataset_choice = "Test"
    file_path = "data/trajectorie/dataset_alta_test.csv"
    
    # Selector de trayectoria
    traj_idx = st.number_input(
        "Índice de trayectoria", 
        min_value=0, 
        max_value=2, 
        value=0,
        help="Índice de la trayectoria a predecir (0-2)"
    )
    
    load_data_btn = st.button("📥 Cargar Datos", type="primary")
    
    if load_data_btn:
        with st.spinner(f"Cargando datos de {dataset_choice}..."):
            try:
                # Cargar CSV seleccionado
                data_selected = pd.read_csv(file_path)
                
                # Extraer las coordenadas de la trayectoria seleccionada
                coords = []
                for i in range(1, 1021):  # Para 1020 puntos
                    col_x = f"xm{i}"
                    col_y = f"ym{i}"
                    if col_x in data_selected.columns and col_y in data_selected.columns:
                        x = data_selected[col_x].iloc[int(traj_idx)]
                        y = data_selected[col_y].iloc[int(traj_idx)]
                        coords.append([x, y])
                
                # Crear nuevo dataframe
                df_test = pd.DataFrame(coords, columns=["XM", "YM"])
                
                st.session_state["test_data"] = df_test
                st.session_state["dataset_choice"] = dataset_choice
                st.session_state["traj_idx"] = traj_idx
                
                st.success(f"✅ Datos cargados: {dataset_choice}, Trayectoria {traj_idx+1}")
                
            except Exception as e:
                st.error(f"Error al cargar datos: {e}")
    
    # Reemplazar la visualización de datos cargados con plotly:
    if "test_data" in st.session_state:
        df = st.session_state["test_data"]
        
        # Gráfico interactivo con Plotly
        fig_preview = go.Figure()
        
        # Trayectoria completa
        fig_preview.add_trace(go.Scatter(
            x=df["XM"], y=df["YM"],
            mode='lines',
            name="Trayectoria",
            line=dict(color='blue', width=2)
        ))
        
        # Puntos inicial y final
        fig_preview.add_trace(go.Scatter(
            x=[df["XM"].iloc[0]], y=[df["YM"].iloc[0]],
            mode='markers',
            name="Inicio",
            marker=dict(color='green', size=10, symbol='circle')
        ))
        
        fig_preview.add_trace(go.Scatter(
            x=[df["XM"].iloc[-1]], y=[df["YM"].iloc[-1]],
            mode='markers',
            name="Fin",
            marker=dict(color='red', size=10, symbol='circle')
        ))
        
        # Configuración
        fig_preview.update_layout(
            title=f"Trayectoria {st.session_state.get('traj_idx', 0)+1}",
            xaxis_title="Posición X",
            yaxis_title="Posición Y",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly"
        )
        
        # Mantener proporciones
        fig_preview.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig_preview, use_container_width=True)

with col3:
    st.markdown("<h2 class='section-header'>3. 🎯 Predicción</h2>", unsafe_allow_html=True)
    
    if lottie_prediction:
        st_lottie(lottie_prediction, height=150, key="prediction_animation")
    
    # Configuración simplificada
    window_size = st.slider(
        "Ventana de entrada (puntos)",
        min_value=100, 
        max_value=500,
        value=200,
        step=25,
        help="Número de puntos consecutivos utilizados como entrada"
    )
    
    # Normalización activada por defecto
    normalize = st.checkbox("Normalizar datos", value=True,
                           help="Aplicar normalización usando escaladores pre-entrenados (recomendado)")
    
    predict_btn = st.button("🔮 Generar Predicción", type="primary")
    
    if predict_btn:
        if "model" not in st.session_state:
            st.error("⚠️ Debes cargar un modelo primero")
        elif "test_data" not in st.session_state:
            st.error("⚠️ Debes cargar datos de prueba primero")
        else:
            with st.spinner("Generando predicción..."):
                try:
                    # Obtener datos y modelo
                    df = st.session_state["test_data"]
                    model = st.session_state["model"]
                    
                    
                    if "model" in st.session_state:
                        model = st.session_state["model"]
                        model_type = st.session_state["model_info"]["tipo"]
                        
                        # Extraer coordenadas
                        X_raw = df["XM"].values
                        Y_raw = df["YM"].values
                        
                        # Aplicar normalización usando el escalador original
                        if SCALERS_LOADED and normalize:
                            # Obtener n_coords_input del modelo
                            n_coords_input = window_size
                            
                            # Crear matriz de entrada con formato para el escalador
                            X_test = np.zeros((1, 1020))  # 1020 es el tamaño que espera el escalador
                            for i in range(min(window_size, 510)):
                                X_test[0, i*2] = X_raw[i]
                                X_test[0, i*2+1] = Y_raw[i]
                            
                            # Normalizar
                            X_test_scaled = scaler_X.transform(X_test)
                            
                            # Formatear los datos según el tipo de modelo
                            if model_type == "Dense":
                                # Para modelo Dense: usar directamente los datos normalizados (forma 2D)
                                input_sequence = X_test_scaled
                                st.info("Usando formato de datos para modelo Dense (aplanado)")
                            else:
                                # Para modelos LSTM/GRU: reshape a 3D (batch, timesteps, features)
                                input_sequence = np.zeros((1, window_size, 2))
                                for i in range(window_size):
                                    if i*2 < X_test_scaled.shape[1]:
                                        input_sequence[0, i, 0] = X_test_scaled[0, i*2]
                                        input_sequence[0, i, 1] = X_test_scaled[0, i*2+1]
                                st.info(f"Usando formato de datos para modelo {model_type} (secuencial)")
                            
                            # Realizar predicción
                            Y_pred_scaled = model.predict(input_sequence, verbose=0)
                            
                            # Desnormalizar
                            output_shape = Y_pred_scaled.shape[1]
                            output_scaled = np.zeros((1, 1020))  # Mismo tamaño que espera el escalador
                            for i in range(output_shape//2):
                                output_scaled[0, i*2] = Y_pred_scaled[0, i*2]
                                output_scaled[0, i*2+1] = Y_pred_scaled[0, i*2+1]
                            
                            Y_pred = scaler_Y.inverse_transform(output_scaled)
                            
                            # Extraer coordenadas
                            pred_x = []
                            pred_y = []
                            for i in range(output_shape//2):
                                if i*2 < Y_pred.shape[1]:
                                    pred_x.append(Y_pred[0, i*2])
                                    pred_y.append(Y_pred[0, i*2+1])
                    
                    # Obtener datos reales para comparar
                    real_x = X_raw[window_size:window_size+len(pred_x)]
                    real_y = Y_raw[window_size:window_size+len(pred_y)]
                    
                    # Guardar resultados
                    st.session_state["prediction_results"] = {
                        "input_x": X_raw[:window_size],
                        "input_y": Y_raw[:window_size],
                        "real_x": real_x,
                        "real_y": real_y,
                        "pred_x": np.array(pred_x),
                        "pred_y": np.array(pred_y),
                        "window_size": window_size
                    }
                    
                    st.success("✅ Predicción generada correctamente")
                    
                except Exception as e:
                    st.error(f"⚠️ Error en predicción: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Reemplazar la sección de visualización de resultados:
if "prediction_results" in st.session_state:
    st.markdown("<h2 class='section-header'>📊 Resultados de Predicción</h2>", unsafe_allow_html=True)
    
    results = st.session_state["prediction_results"]
    
    # Opciones de visualización
    viz_options = st.expander("🎨 Opciones de visualización")
    with viz_options:
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        
        with col_viz1:
            marker_size = st.slider("Tamaño de marcadores", 1, 10, 4)
            line_width = st.slider("Grosor de líneas", 1, 5, 2)
        
        with col_viz2:
            show_markers = st.checkbox("Mostrar marcadores", value=True)
            show_grid = st.checkbox("Mostrar cuadrícula", value=True)
            show_error_overlay = st.checkbox("Mostrar error superpuesto", value=False)
        
        with col_viz3:
            show_3d = st.checkbox("Vista 3D", value=False, 
                                  help="Mostrar visualización 3D de la trayectoria")
    
    # Dividir en dos columnas: gráfica y métricas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crear gráfico interactivo con Plotly
        fig = go.Figure()
        
        # Secuencia de entrada
        fig.add_trace(go.Scatter(
            x=results["input_x"], y=results["input_y"],
            mode='lines' if not show_markers else 'lines+markers',
            name="Secuencia de entrada",
            line=dict(color='black', width=line_width),
            marker=dict(size=marker_size)
        ))
        
        # Secuencia real
        fig.add_trace(go.Scatter(
            x=results["real_x"], y=results["real_y"],
            mode='lines' if not show_markers else 'lines+markers',
            name="Secuencia real",
            line=dict(color='blue', width=line_width),
            marker=dict(size=marker_size)
        ))
        
        # Secuencia predicha
        fig.add_trace(go.Scatter(
            x=results["pred_x"], y=results["pred_y"],
            mode='lines' if not show_markers else 'lines+markers',
            name="Secuencia predicha",
            line=dict(color='red', width=line_width, dash='dash'),
            marker=dict(size=marker_size)
        ))
        
        # Último punto de entrada
        fig.add_trace(go.Scatter(
            x=[results["input_x"][-1]], y=[results["input_y"][-1]],
            mode='markers',
            name="Último punto de entrada",
            marker=dict(color='black', size=12, symbol='circle')
        ))
        
        # Mejorar diseño
        traj_idx = st.session_state.get("traj_idx", 0)
        fig.update_layout(
            title=f"Predicción de Trayectoria {traj_idx+1}",
            xaxis_title="Posición X",
            yaxis_title="Posición Y",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            autosize=True,
            height=600,
            hovermode="closest",
            template="plotly"
        )
        
        # Mantener proporciones
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        
        # Añadir grid si está activado
        if show_grid:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        # Si está activada la visualización de error
        if show_error_overlay:
            error_points = []
            for i in range(len(results["real_x"])):
                err = np.sqrt((results["real_x"][i] - results["pred_x"][i])**2 + 
                             (results["real_y"][i] - results["pred_y"][i])**2)
                error_points.append(err)
            
            # Crear líneas coloreadas por segmento según el error
            for i in range(len(results["pred_x"])-1):
                error_value = error_points[i]
                color = px.colors.sample_colorscale('Jet', [error_points[i]/max(error_points)])[0]
                
                fig.add_trace(go.Scatter(
                    x=[results["pred_x"][i], results["pred_x"][i+1]],
                    y=[results["pred_y"][i], results["pred_y"][i+1]],
                    mode='lines',
                    line=dict(color=color, width=line_width*1.5),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f'Error: {error_value:.4f}'
                ))
            
            # Añadir barra de color
            fig.update_layout(
                coloraxis=dict(colorscale='Jet'),
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(
                    title="Error",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300,
                    yanchor="top", y=1,
                    xanchor="left", x=1.02,
                )
            )
        
        # Mostrar el gráfico interactivo
        st.plotly_chart(fig, use_container_width=True)

        # Vista adicional - Gráfica 3D (opcional)
        show_3d = st.checkbox("Mostrar vista 3D", value=False)
        if show_3d:
            # Tiempo como tercera dimensión
            t_input = np.arange(len(results["input_x"]))
            t_real = np.arange(len(results["input_x"]), len(results["input_x"]) + len(results["real_x"]))
            t_pred = np.arange(len(results["input_x"]), len(results["input_x"]) + len(results["pred_x"]))
            
            fig3d = go.Figure()
            
            # Entrada
            fig3d.add_trace(go.Scatter3d(
                x=results["input_x"], y=results["input_y"], z=t_input,
                mode='lines',
                name='Entrada',
                line=dict(color='black', width=4)
            ))
            
            # Real
            fig3d.add_trace(go.Scatter3d(
                x=results["real_x"], y=results["real_y"], z=t_real,
                mode='lines',
                name='Real',
                line=dict(color='blue', width=4)
            ))
            
            # Predicción
            fig3d.add_trace(go.Scatter3d(
                x=results["pred_x"], y=results["pred_y"], z=t_pred,
                mode='lines',
                name='Predicción',
                line=dict(color='red', width=4, dash='dash')
            ))
            
            fig3d.update_layout(
                title="Vista 3D de la trayectoria",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Paso temporal',
                    aspectmode='auto'
                ),
                height=600
            )
            
            st.plotly_chart(fig3d, use_container_width=True)

    # Columna de métricas
    with col2:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        
        # Métricas básicas
        mse_x = np.mean((results["real_x"] - results["pred_x"])**2)
        mse_y = np.mean((results["real_y"] - results["pred_y"])**2)
        mse_total = (mse_x + mse_y) / 2
        
        st.markdown("### 📏 Métricas de Error")
        st.metric("MSE Total", f"{mse_total:.5f}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("MSE (X)", f"{mse_x:.5f}")
        with col_b:
            st.metric("MSE (Y)", f"{mse_y:.5f}")
        
        # Errores por pasos de predicción
        st.markdown("### 📈 Error por paso")
        
        # Calcular error por paso de tiempo
        error_by_step = []
        for i in range(len(results["real_x"])):
            err_x = (results["real_x"][i] - results["pred_x"][i])**2
            err_y = (results["real_y"][i] - results["pred_y"][i])**2
            error_by_step.append(np.sqrt(err_x + err_y))
        
        # Gráfica de error interactiva
        error_fig = go.Figure()
        error_fig.add_trace(go.Scatter(
            x=list(range(len(error_by_step))),
            y=error_by_step,
            mode='lines',
            name='Error',
            line=dict(color='red', width=2)
        ))

        error_fig.update_layout(
            title="Error por paso de predicción",
            xaxis_title="Paso",
            yaxis_title="Error",
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            hovermode="closest"
        )

        st.plotly_chart(error_fig, use_container_width=True)
        
        # Información de la predicción
        st.markdown("### ℹ️ Información")
        st.markdown(f"**Modelo:** {st.session_state.get('model_info', {}).get('tipo', 'No especificado')}")
        st.markdown(f"**Ventana de entrada:** {results['window_size']} puntos")
        st.markdown(f"**Puntos predichos:** {len(results['pred_x'])}")
        
        # Botón para guardar resultados
        st.download_button(
            label="💾 Guardar resultados",
            data=pd.DataFrame({
                'real_x': results["real_x"],
                'real_y': results["real_y"],
                'pred_x': results["pred_x"],
                'pred_y': results["pred_y"],
                'error': error_by_step
            }).to_csv(),
            file_name=f"prediccion_trayectoria_{traj_idx+1}_{int(time.time())}.csv",
            mime="text/csv"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

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