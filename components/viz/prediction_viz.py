import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.simulation import generate_realtime_prediction
import tensorflow as tf
import joblib



def load_model_results(model_type):
    """Carga los resultados reales de predicción del modelo especificado"""
    # Mapeo de nombres de modelos a archivos
    model_files = {
        "LSTM": "models/lstm_model.h5",
        "GRU": "models/gru_model.h5",
        "Dense": "models/dense_model.h5"
    }
    
    # Cargar el modelo directamente
    model = tf.keras.models.load_model(model_files[model_type], compile=False)
    
    # Cargar datos de test y escaladores
    test_data = pd.read_csv("data/trajectorie/dataset_alta_test.csv")
    scaler_X = joblib.load('data/parametros/scaler_X.pkl')
    scaler_Y = joblib.load('data/parametros/scaler_Y.pkl')
    
    # Parámetros para evaluar el modelo
    traj_idx = 0
    n_coords_input = 510
    
    # Extraer coordenadas
    coords_x, coords_y = [], []
    for i in range(1, 1021):
        col_x, col_y = f"xm{i}", f"ym{i}"
        coords_x.append(test_data[col_x].iloc[traj_idx])
        coords_y.append(test_data[col_y].iloc[traj_idx])
    
    # Dividir en entrada y salida
    X_raw = coords_x[:n_coords_input]
    Y_raw = coords_y[:n_coords_input]
    real_x = coords_x[n_coords_input:]
    real_y = coords_y[n_coords_input:]
    
    # Preparar datos para el modelo
    X_test = np.zeros((1, 1020))
    for i in range(n_coords_input):
        X_test[0, i*2] = X_raw[i]
        X_test[0, i*2+1] = Y_raw[i]
    
    # Normalizar
    X_test_scaled = scaler_X.transform(X_test)
    
    # Ajustar formato según modelo
    if model_type == "Dense":
        input_data = X_test_scaled
    else:
        input_data = np.zeros((1, n_coords_input, 2))
        for i in range(n_coords_input):
            input_data[0, i, 0] = X_test_scaled[0, i*2]
            input_data[0, i, 1] = X_test_scaled[0, i*2+1]
    
    # Realizar predicción
    Y_pred_scaled = model.predict(input_data, verbose=0)
    
    # Desnormalizar
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    
    # Extraer coordenadas predichas
    pred_x, pred_y = [], []
    for i in range(n_coords_input):
        pred_x.append(Y_pred[0, i*2])
        pred_y.append(Y_pred[0, i*2+1])
    
    # Calcular métricas
    mse_x = np.mean((np.array(real_x) - np.array(pred_x))**2)
    mse_y = np.mean((np.array(real_y) - np.array(pred_y))**2)
    mse = (mse_x + mse_y) / 2
    
    mae_x = np.mean(np.abs(np.array(real_x) - np.array(pred_x)))
    mae_y = np.mean(np.abs(np.array(real_y) - np.array(pred_y)))
    mae = (mae_x + mae_y) / 2
    
    # Calcular error por punto y segmento
    error_by_point = [np.sqrt((real_x[i] - pred_x[i])**2 + (real_y[i] - pred_y[i])**2) 
                      for i in range(len(real_x))]
    
    n_points = len(error_by_point)
    segment_size = n_points // 3
    
    error_by_segment = {
        "inicial": np.mean(error_by_point[:segment_size]),
        "media": np.mean(error_by_point[segment_size:2*segment_size]),
        "final": np.mean(error_by_point[2*segment_size:])
    }
    
    # Retornar resultados
    return {
        "real_x": real_x,
        "real_y": real_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "metrics": {
            "mse": mse, "mae": mae,
            "mse_x": mse_x, "mse_y": mse_y,
            "mae_x": mae_x, "mae_y": mae_y,
            "max_error": max(error_by_point),
            "std_error": np.std(error_by_point)
        },
        "error_by_point": error_by_point,
        "error_by_segment": error_by_segment,
        "input_x": X_raw, "input_y": Y_raw
    }

def create_fallback_data(model_type):
    # Simulación de trayectoria
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.8
    
    # Datos iniciales (para todos los modelos)
    input_x = r * np.cos(theta[:50])
    input_y = r * np.sin(theta[:50])
    
    # Datos reales
    real_x = r * np.cos(theta[50:])
    real_y = r * np.sin(theta[50:])
    
    # Añadir diferentes niveles de ruido según el modelo
    noise_level = 0.05 if model_type == "LSTM" else 0.08 if model_type == "GRU" else 0.12
    
    # Datos predichos (con ruido para simular error)
    pred_x = real_x + noise_level * np.random.randn(len(real_x))
    pred_y = real_y + noise_level * np.random.randn(len(real_y))
    
    # Calcular error por punto
    error_by_point = []
    for i in range(len(real_x)):
        err = np.sqrt((real_x[i] - pred_x[i])**2 + (real_y[i] - pred_y[i])**2)
        error_by_point.append(err)
    
    # Crear objeto de resultados
    results = {
        "real_x": real_x,
        "real_y": real_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "metrics": {
            "mse": np.mean(np.power(error_by_point, 2)),
            "mae": np.mean(error_by_point),
            "mse_x": np.mean((real_x - pred_x)**2),
            "mse_y": np.mean((real_y - pred_y)**2),
            "mae_x": np.mean(np.abs(real_x - pred_x)),
            "mae_y": np.mean(np.abs(real_y - pred_y)),
            "max_error": max(error_by_point),
            "std_error": np.std(error_by_point)
        },
        "error_by_point": error_by_point,
        "error_by_segment": {
            "inicial": np.mean(error_by_point[:len(error_by_point)//3]),
            "media": np.mean(error_by_point[len(error_by_point)//3:2*len(error_by_point)//3]),
            "final": np.mean(error_by_point[2*len(error_by_point)//3:])
        },
        "input_x": input_x,
        "input_y": input_y
    }
    
    return results

def display_trajectory_comparison_section():
    """Muestra análisis comparativo avanzado de modelos de predicción"""
    st.markdown("<h3 style='color:#4b6cb7;'>📊 Análisis Comparativo de Modelos</h3>", unsafe_allow_html=True)
    
    # Cargar datos reales para cada modelo
    with st.spinner("Cargando datos de modelos..."):
        lstm_results = load_model_results("LSTM")
        gru_results = load_model_results("GRU")
        dense_results = load_model_results("Dense")
        
        # Crear dataframe de métricas con datos reales
        metrics_df = pd.DataFrame({
            "Modelo": ["LSTM", "GRU", "Dense"],
            "MSE": [lstm_results["metrics"]["mse"], 
                   gru_results["metrics"]["mse"], 
                   dense_results["metrics"]["mse"]],
            "MAE": [lstm_results["metrics"]["mae"], 
                   gru_results["metrics"]["mae"], 
                   dense_results["metrics"]["mae"]],
            "Error máximo": [lstm_results["metrics"]["max_error"], 
                           gru_results["metrics"]["max_error"], 
                           dense_results["metrics"]["max_error"]],
            "Desviación estándar": [lstm_results["metrics"]["std_error"], 
                                  gru_results["metrics"]["std_error"], 
                                  dense_results["metrics"]["std_error"]]
        })
    
    # Pestañas para diferentes tipos de análisis
    tab1, tab2, tab3 = st.tabs(["Comparativa de Errores", "Distribución Espacial", "Análisis por Región"])
    
    # Tab 1: Comparativa de errores
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Controles para el análisis
            st.markdown("#### Configuración")
            
            # En lugar de seleccionar un modelo, seleccionamos un tipo de análisis
            analisis_tipo = st.selectbox(
                "Tipo de análisis:",
                ["Error absoluto", "Error por componente", "Error acumulativo"]
            )
            
            # Métricas a visualizar
            metricas = st.multiselect(
                "Métricas:",
                ["MSE", "MAE", "Error máximo", "Desviación estándar"],
                default=["MSE", "MAE"]
            )
            
            # Tabla comparativa de métricas (ahora filtrada según la selección)
            st.markdown("#### Resumen de métricas")
            metrics_display = metrics_df.copy()
            # Asegurar que siempre tenemos la columna Modelo
            columnas_a_mostrar = ["Modelo"] + metricas
            # Formatear solo las columnas numéricas seleccionadas
            for col in metricas:
                if col in metrics_display.columns:
                    metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.4f}")

            # Mostrar solo las columnas seleccionadas
            st.dataframe(metrics_display[columnas_a_mostrar], use_container_width=True)
            
            # Destacar mejor modelo
            mejor_modelo = metrics_df.iloc[metrics_df["MSE"].argmin()]["Modelo"]
            st.success(f"✅ Mejor modelo por MSE: **{mejor_modelo}**")
        
        with col2:
            # Gráfico avanzado basado en la selección
            st.markdown("#### Visualización de error")
            
            # Crear figura para el análisis de error
            fig = go.Figure()
            
            # Usar datos reales
            if analisis_tipo == "Error absoluto":
                # Error absoluto por paso
                fig.add_trace(go.Scatter(
                    x=list(range(len(lstm_results["error_by_point"]))), 
                    y=lstm_results["error_by_point"], 
                    mode='lines+markers', name='LSTM', 
                    line=dict(color='#4b6cb7')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gru_results["error_by_point"]))), 
                    y=gru_results["error_by_point"], 
                    mode='lines+markers', name='GRU', 
                    line=dict(color='#ff7043')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(dense_results["error_by_point"]))), 
                    y=dense_results["error_by_point"], 
                    mode='lines+markers', name='Dense', 
                    line=dict(color='#4caf50')
                ))
                
                fig.update_layout(
                    title="Error absoluto por paso de predicción",
                    xaxis_title="Paso de predicción",
                    yaxis_title="Error absoluto",
                )
                
            elif analisis_tipo == "Error por componente":
                # Error desglosado por componente X e Y
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Error en X", "Error en Y"))
                
                # Calcular errores por componente
                lstm_error_x = np.abs(np.array(lstm_results["real_x"]) - np.array(lstm_results["pred_x"]))
                lstm_error_y = np.abs(np.array(lstm_results["real_y"]) - np.array(lstm_results["pred_y"]))
                
                gru_error_x = np.abs(np.array(gru_results["real_x"]) - np.array(gru_results["pred_x"]))
                gru_error_y = np.abs(np.array(gru_results["real_y"]) - np.array(gru_results["pred_y"]))
                
                dense_error_x = np.abs(np.array(dense_results["real_x"]) - np.array(dense_results["pred_x"]))
                dense_error_y = np.abs(np.array(dense_results["real_y"]) - np.array(dense_results["pred_y"]))
                
                # Componente X
                fig.add_trace(go.Scatter(
                    x=list(range(len(lstm_error_x))), 
                    y=lstm_error_x, 
                    mode='lines', name='LSTM - X', 
                    line=dict(color='#4b6cb7')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gru_error_x))), 
                    y=gru_error_x, 
                    mode='lines', name='GRU - X', 
                    line=dict(color='#ff7043')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(dense_error_x))), 
                    y=dense_error_x, 
                    mode='lines', name='Dense - X', 
                    line=dict(color='#4caf50')
                ), row=1, col=1)
                
                # Componente Y
                fig.add_trace(go.Scatter(
                    x=list(range(len(lstm_error_y))), 
                    y=lstm_error_y, 
                    mode='lines', name='LSTM - Y', 
                    line=dict(color='#4b6cb7', dash='dash')
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gru_error_y))), 
                    y=gru_error_y, 
                    mode='lines', name='GRU - Y', 
                    line=dict(color='#ff7043', dash='dash')
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(dense_error_y))), 
                    y=dense_error_y, 
                    mode='lines', name='Dense - Y', 
                    line=dict(color='#4caf50', dash='dash')
                ), row=1, col=2)
                
                fig.update_layout(
                    title="Error por componente (X e Y)",
                    xaxis_title="Paso de predicción",
                    yaxis_title="Error medio",
                )
                
            else:  # Error acumulativo
                # Error acumulado (suma del error a lo largo de la trayectoria)
                lstm_error_acum = np.cumsum(lstm_results["error_by_point"])
                gru_error_acum = np.cumsum(gru_results["error_by_point"])
                dense_error_acum = np.cumsum(dense_results["error_by_point"])
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(lstm_error_acum))), 
                    y=lstm_error_acum, 
                    mode='lines+markers', name='LSTM', 
                    fill='tozeroy', line=dict(color='#4b6cb7')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(gru_error_acum))), 
                    y=gru_error_acum, 
                    mode='lines+markers', name='GRU', 
                    fill='tozeroy', line=dict(color='#ff7043')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(dense_error_acum))), 
                    y=dense_error_acum, 
                    mode='lines+markers', name='Dense', 
                    fill='tozeroy', line=dict(color='#4caf50')
                ))
                
                fig.update_layout(
                    title="Error acumulativo a lo largo de la trayectoria",
                    xaxis_title="Paso de predicción",
                    yaxis_title="Error acumulado",
                )
            
            fig.update_layout(
                height=500,
                hovermode='closest',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Distribución espacial del error
    with tab2:
        st.markdown("#### Mapa de calor del error")
        st.markdown("""
        Este análisis muestra dónde se concentra el error de predicción en el espacio 2D de la trayectoria.
        Las regiones en rojo indican áreas con mayor error de predicción.
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            modelo_mapa = st.radio("Modelo a analizar:", ["LSTM", "GRU", "Dense"])
            
            # Seleccionar los datos según el modelo elegido
            model_data = lstm_results if modelo_mapa == "LSTM" else gru_results if modelo_mapa == "GRU" else dense_results
            
            # Estadísticas reales de la distribución
            error_max = model_data["metrics"]["max_error"]
            error_medio = model_data["metrics"]["mae"]
            error_std = model_data["metrics"]["std_error"]
            
            # Mostrar estadísticas reales
            st.markdown("#### Estadísticas de error")
            st.metric("Error máximo", f"{error_max:.4f}")
            st.metric("Error medio", f"{error_medio:.4f}")
            st.metric("Desviación", f"{error_std:.4f}")
            
            # Mostrar errores por segmento
            st.markdown("#### Error por segmento")
            for segmento, valor in model_data["error_by_segment"].items():
                st.metric(segmento.capitalize(), f"{valor:.4f}")
        
        with col2:
            # Crear un mapa de calor 2D basado en los puntos reales y sus errores
            # Usamos las coordenadas reales y el error punto a punto
            
            # Obtener datos reales y predichos
            real_x = model_data["real_x"]
            real_y = model_data["real_y"]
            pred_x = model_data["pred_x"]
            pred_y = model_data["pred_y"]
            error_by_point = model_data["error_by_point"]
            
            # Crear gráfico de mapa de calor usando scatter con colorscale
            fig = go.Figure()
            
            # Mostrar trayectoria real como línea base
            fig.add_trace(go.Scatter(
                x=real_x, 
                y=real_y,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Trayectoria real'
            ))
            
            # Mostrar puntos de predicción coloreados por error
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='markers',
                marker=dict(
                    size=10,
                    color=error_by_point,
                    colorscale='Jet',
                    showscale=True,
                    colorbar=dict(title="Error")
                ),
                name='Predicción'
            ))
            
            # Configuración del layout
            fig.update_layout(
                title=f"Distribución del error en predicción ({modelo_mapa})",
                xaxis_title="Posición X",
                yaxis_title="Posición Y",
                height=500,
                yaxis=dict(scaleanchor="x", scaleratio=1),
                legend=dict(
                    orientation="v",        # vertical
                    yanchor="top",          # anclada arriba
                    y=0.99,                 # posición vertical
                    xanchor="left",         # anclada a la izquierda
                    x=0.01,                 # posición horizontal
                    bgcolor="rgba(255,255,255,0.8)"  # fondo semi-transparente
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Análisis por región de la trayectoria
    with tab3:
        st.markdown("#### Rendimiento por segmentos de trayectoria")
        st.markdown("""
        Análisis de cómo se comportan los diferentes modelos en distintas partes de la trayectoria.
        Se divide la trayectoria en tres segmentos (inicial, medio y final) para un análisis más detallado.
        """)
        
        # Extraer datos reales de segmentos
        segmentos = ["Inicial", "Media", "Final"]
        
        # Obtener errores por segmento de los datos reales
        lstm_error = [
            lstm_results["error_by_segment"]["inicial"],
            lstm_results["error_by_segment"]["media"],
            lstm_results["error_by_segment"]["final"]
        ]
        
        gru_error = [
            gru_results["error_by_segment"]["inicial"],
            gru_results["error_by_segment"]["media"],
            gru_results["error_by_segment"]["final"]
        ]
        
        dense_error = [
            dense_results["error_by_segment"]["inicial"],
            dense_results["error_by_segment"]["media"],
            dense_results["error_by_segment"]["final"]
        ]
        
        # Crear gráfico de barras agrupadas con datos reales
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=segmentos,
            y=lstm_error,
            name='LSTM',
            marker_color='#4b6cb7'
        ))
        
        fig.add_trace(go.Bar(
            x=segmentos,
            y=gru_error,
            name='GRU',
            marker_color='#ff7043'
        ))
        
        fig.add_trace(go.Bar(
            x=segmentos,
            y=dense_error,
            name='Dense',
            marker_color='#4caf50'
        ))
        
        fig.update_layout(
            title="Error por segmento de trayectoria",
            xaxis_title="Segmento de trayectoria",
            yaxis_title="Error medio",
            barmode='group',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada por segmentos con datos reales
        st.markdown("#### Detalle por segmento")
        
        # Calcular qué modelo es mejor en cada segmento
        mejor_inicial = ["LSTM", "GRU", "Dense"][np.argmin([lstm_error[0], gru_error[0], dense_error[0]])]
        mejor_media = ["LSTM", "GRU", "Dense"][np.argmin([lstm_error[1], gru_error[1], dense_error[1]])]
        mejor_final = ["LSTM", "GRU", "Dense"][np.argmin([lstm_error[2], gru_error[2], dense_error[2]])]
        mejor_total = ["LSTM", "GRU", "Dense"][np.argmin([
            lstm_results["metrics"]["mse"], 
            gru_results["metrics"]["mse"], 
            dense_results["metrics"]["mse"]
        ])]
        
        detail_df = pd.DataFrame({
            "Segmento": ["Inicial", "Media", "Final", "Total"],
            "LSTM Error": [f"{lstm_error[0]:.4f}", f"{lstm_error[1]:.4f}", f"{lstm_error[2]:.4f}",
                          f"{lstm_results['metrics']['mse']:.4f}"],
            "GRU Error": [f"{gru_error[0]:.4f}", f"{gru_error[1]:.4f}", f"{gru_error[2]:.4f}",
                         f"{gru_results['metrics']['mse']:.4f}"],
            "Dense Error": [f"{dense_error[0]:.4f}", f"{dense_error[1]:.4f}", f"{dense_error[2]:.4f}",
                           f"{dense_results['metrics']['mse']:.4f}"],
            "Mejor modelo": [mejor_inicial, mejor_media, mejor_final, mejor_total]
        })
        
        st.dataframe(detail_df, use_container_width=True)
        
        # Información adicional basada en datos reales
        mejor_modelo_global = mejor_total
        peor_segmento = "final" if max(lstm_error + gru_error + dense_error) in [lstm_error[2], gru_error[2], dense_error[2]] else "inicial"
        
        st.info(f"""
        **Observación**: Los modelos muestran mayor error en los segmentos {peor_segmento}s de la trayectoria.
        El modelo {mejor_modelo_global} obtiene el mejor rendimiento global, con MSE de {min([lstm_results['metrics']['mse'], gru_results['metrics']['mse'], dense_results['metrics']['mse']]):.4f}.
        """)

