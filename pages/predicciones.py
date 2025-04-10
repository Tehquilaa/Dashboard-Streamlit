import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from streamlit_lottie import st_lottie
from components.headers import get_section_header
from components.utils import load_lottiefile, apply_default_css

# Versión robusta para diferentes versiones de TensorFlow
try:
    # Para TF 2.6+
    @tf.keras.saving.register_keras_serializable()
    def mse(y_true, y_pred):
        return tf.keras.metrics.mean_squared_error(y_true, y_pred)
except AttributeError:
    try:
        # Para TF 2.4-2.5
        @tf.keras.utils.register_keras_serializable()
        def mse(y_true, y_pred):
            return tf.keras.metrics.mean_squared_error(y_true, y_pred)
    except AttributeError:
        # Para versiones más antiguas
        def mse(y_true, y_pred):
            return tf.keras.metrics.mean_squared_error(y_true, y_pred)

# Configuración de la página
st.set_page_config(
    page_title="Predicciones Personalizadas - Dashboard Balín",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_default_css()

# Cargar animación para la página
lottie_predict = load_lottiefile("animations/prediction.json")
if not lottie_predict:
    lottie_predict = "https://assets6.lottiefiles.com/packages/lf20_ysrn2iwp.json"

# Título y descripción
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h1>🔮 Predicciones Personalizadas</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="justified-text highlight">
    Esta sección te permite cargar modelos previamente entrenados y generar predicciones
    con datos de prueba personalizados. Carga un archivo CSV con coordenadas de trayectoria
    y visualiza cómo el modelo predice su comportamiento.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st_lottie(lottie_predict, height=250, key="predict_animation")

st.markdown("---")

# Sección 1: Cargar Modelo
st.markdown(get_section_header("1", "📂", "Cargar Modelo"), unsafe_allow_html=True)

model_col1, model_col2 = st.columns(2)

with model_col1:
    # Opciones para cargar modelo
    load_option = st.radio(
        "¿Cómo quieres cargar el modelo?",
        options=["Usar modelo predeterminado", "Cargar archivo de modelo (.h5)"]
    )
    
    if load_option == "Cargar archivo de modelo (.h5)":
        model_file = st.file_uploader("Selecciona el archivo del modelo (.h5)", type=["h5"])
        
        if model_file:
            with st.spinner("Cargando modelo..."):
                # Guardar el archivo temporalmente
                with open("temp_model.h5", "wb") as f:
                    f.write(model_file.getbuffer())
                
                try:
                    # Cargar el modelo
                    model = tf.keras.models.load_model("temp_model.h5", custom_objects={'mse': mse})
                    st.session_state["model"] = model
                    st.success("¡Modelo cargado correctamente!")
                    
                    # Mostrar resumen del modelo
                    st.subheader("Resumen del modelo:")
                    # No podemos usar model.summary() directamente con st
                    # Mostramos algunas propiedades básicas
                    st.text(f"Número de capas: {len(model.layers)}")
                    st.text(f"Capa de entrada: {model.input_shape}")
                    st.text(f"Capa de salida: {model.output_shape}")
                    
                except Exception as e:
                    # Si falla incluso con el objeto personalizado, intentar sin compilar
                    try:
                        model = tf.keras.models.load_model("temp_model.h5", compile=False)
                        st.session_state["model"] = model
                        st.success("¡Modelo cargado correctamente (sin compilar)!")
                    except Exception as e2:
                        st.error(f"Error al cargar el modelo: {e2}")
    else:
        model_type = st.selectbox(
            "Selecciona el tipo de modelo predeterminado:",
            options=["LSTM", "GRU", "Red Densa"]
        )
        
        # Mapear selección a nombre de archivo
        model_file_map = {
            "LSTM": "lstm_model.h5",
            "GRU": "gru_model.h5",
            "Red Densa": "dense_model.h5"
        }
        
        model_path = os.path.join("models", model_file_map[model_type])
        
        if os.path.exists(model_path):
            if st.button("Cargar modelo predeterminado"):
                with st.spinner(f"Cargando modelo {model_type}..."):
                    try:
                        model = tf.keras.models.load_model(model_path, custom_objects={'mse': mse})
                        st.session_state["model"] = model
                        st.success(f"¡Modelo {model_type} cargado correctamente!")
                    except Exception as e:
                        try:
                            model = tf.keras.models.load_model(model_path, compile=False)
                            st.session_state["model"] = model
                            st.success(f"¡Modelo {model_type} cargado correctamente (sin compilar)!")
                        except Exception as e2:
                            st.error(f"Error al cargar el modelo: {e2}")
        else:
            st.warning(f"No se encontró el modelo predeterminado en {model_path}")
            st.info("Asegúrate de que los modelos estén guardados en la carpeta 'models/'")

with model_col2:
    # Información sobre el modelo y su uso
    st.markdown("""
    ### Información sobre los modelos
    
    Los modelos deben haber sido entrenados con la misma estructura de datos y preprocesamiento
    utilizado en este dashboard. Los modelos predeterminados son:
    
    - **LSTM**: Mejor rendimiento general, especialmente para secuencias largas.
    - **GRU**: Equilibrio entre rendimiento y eficiencia computacional.
    - **Red Densa**: Más simple, puede funcionar bien en ciertas condiciones.
    
    Para usar tus propios modelos, guárdalos en formato .h5 usando `model.save('mi_modelo.h5')` en tu código
    de entrenamiento.
    """)
    
    # Mostrar parámetros del modelo si está cargado
    if "model" in st.session_state:
        st.markdown("### Parámetros del modelo cargado:")
        model = st.session_state["model"]
        total_params = model.count_params()
        st.metric("Parámetros totales", f"{total_params:,}")

# Sección 2: Cargar Datos de Prueba
st.markdown(get_section_header("2", "📊", "Cargar Datos de Prueba"), unsafe_allow_html=True)

data_col1, data_col2 = st.columns(2)

with data_col1:
    # Opción para cargar datos
    test_data_option = st.radio(
        "¿Cómo quieres cargar los datos de prueba?",
        options=["Usar dataset de ejemplo (dataset_alto_test.csv)", "Cargar archivo CSV"]
    )
    
    if test_data_option == "Cargar archivo CSV":
        test_file = st.file_uploader("Selecciona un archivo CSV con datos de trayectoria", type=["csv"])
        
        if test_file:
            try:
                df_test = pd.read_csv(test_file)
                
                # Verificar el formato y convertir si es necesario
                # Detectar si tiene formato de múltiples columnas (xm1, ym1, etc.)
                if any("xm" in col.lower() for col in df_test.columns):
                    st.info("Detectado formato de múltiples columnas. Convirtiendo datos...")
                    
                    # Permitir seleccionar la trayectoria
                    traj_options = [f"Trayectoria {i+1}" for i in range(len(df_test))]
                    selected_traj = st.radio("Selecciona la trayectoria a utilizar:", traj_options)
                    traj_idx = traj_options.index(selected_traj)
                    
                    # Extraer las coordenadas de la trayectoria seleccionada
                    coords = []
                    for i in range(1, 511):  # Asumiendo coordenadas desde 1 hasta 510
                        col_x = f"xm{i}"
                        col_y = f"ym{i}"
                        if col_x in df_test.columns and col_y in df_test.columns:
                            x = df_test[col_x].iloc[traj_idx]  # Usamos el índice seleccionado
                            y = df_test[col_y].iloc[traj_idx]
                            coords.append([x, y])
                    
                    # Crear nuevo dataframe con formato XM, YM
                    new_df = pd.DataFrame(coords, columns=["XM", "YM"])
                    df_test = new_df
                
                # Verificar formato final
                if "XM" in df_test.columns and "YM" in df_test.columns:
                    st.session_state["test_data"] = df_test
                    st.success(f"¡Datos cargados correctamente! ({len(df_test)} puntos)")
                else:
                    st.error("El archivo debe contener columnas 'XM' y 'YM' o formato de múltiples columnas (xm1,ym1,...)")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")
    else:
        # Usar dataset de ejemplo
        example_path = "data/trajectorie/dataset_alta_test.csv"
        
        if os.path.exists(example_path):
            if st.button("Cargar datos de ejemplo"):
                with st.spinner("Cargando datos de ejemplo..."):
                    df_test = pd.read_csv(example_path)
                    
                    # Verificar el formato y convertir si es necesario
                    # Detectar si tiene formato de múltiples columnas (xm1, ym1, etc.)
                    if any("xm" in col.lower() for col in df_test.columns):
                        st.info("Detectado formato de múltiples columnas. Detectadas {} trayectorias.".format(len(df_test)))
                        
                        # Permitir seleccionar la trayectoria
                        traj_options = [f"Trayectoria {i+1}" for i in range(len(df_test))]
                        selected_traj = st.radio("Selecciona la trayectoria a utilizar:", traj_options)
                        traj_idx = traj_options.index(selected_traj)
                        
                        # Extraer las coordenadas de la trayectoria seleccionada
                        coords = []
                        for i in range(1, 1021):  # Asumiendo coordenadas desde 1 hasta 510
                            col_x = f"xm{i}"
                            col_y = f"ym{i}"
                            if col_x in df_test.columns and col_y in df_test.columns:
                                x = df_test[col_x].iloc[traj_idx]  # Usamos el índice seleccionado
                                y = df_test[col_y].iloc[traj_idx]
                                coords.append([x, y])
                        
                        # Crear nuevo dataframe con formato XM, YM
                        new_df = pd.DataFrame(coords, columns=["XM", "YM"])
                        df_test = new_df
                    
                    # Verificar formato final
                    if "XM" in df_test.columns and "YM" in df_test.columns:
                        st.session_state["test_data"] = df_test
                        st.success(f"¡Datos cargados correctamente! ({len(df_test)} puntos)")
                    else:
                        st.error("El formato del archivo no es compatible. Debe tener columnas 'XM' y 'YM' o formato de múltiples columnas (xm1,ym1,...)")
        else:
            st.warning(f"No se encontró el archivo de ejemplo en {example_path}")
            # Intentar buscar en la ubicación alternativa
            alternative_path = "12G-1Hz.csv"
            if os.path.exists(alternative_path):
                if st.button("Cargar datos de ejemplo (ruta alternativa)"):
                    with st.spinner("Cargando datos de ejemplo..."):
                        df_test = pd.read_csv(alternative_path)
                        st.session_state["test_data"] = df_test
                        st.success(f"¡Datos de ejemplo cargados! ({len(df_test)} puntos)")
            else:
                st.error("No se encontró el archivo de ejemplo. Carga un archivo CSV manualmente.")

with data_col2:
    # Mostrar visualización de los datos cargados
    if "test_data" in st.session_state:
        df_test = st.session_state["test_data"]
        
        st.markdown("### Vista previa de los datos:")
        st.dataframe(df_test.head(), use_container_width=True)
        
        # Visualizar trayectoria
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_test["XM"], df_test["YM"], "b-", alpha=0.7)
        ax.scatter(df_test["XM"].iloc[0], df_test["YM"].iloc[0], c="g", s=100, marker="o", label="Inicio")
        ax.scatter(df_test["XM"].iloc[-1], df_test["YM"].iloc[-1], c="r", s=100, marker="o", label="Fin")
        ax.set_title("Trayectoria de prueba")
        ax.set_xlabel("Posición X")
        ax.set_ylabel("Posición Y")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# Sección 3: Configurar y Ejecutar Predicción
st.markdown(get_section_header("3", "⚙️", "Configurar y Ejecutar Predicción"), unsafe_allow_html=True)

if "model" in st.session_state and "test_data" in st.session_state:
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        # Parámetros para la predicción
        st.subheader("Parámetros de predicción")
        
        window_size = st.slider(
            "Tamaño de ventana de entrada",
            min_value=100,
            max_value=1020,  # Actualizado de 510 a 1020
            value=200,
            help="Número de puntos consecutivos para predecir el siguiente"
        )
        
        forecast_horizon = st.slider(
            "Horizonte de predicción",
            min_value=100,
            max_value=1020,  # Actualizado de 510 a 1020
            value=300,
            help="Número de puntos a predecir"
        )
        
        # Punto de inicio
        max_start = max(10, len(st.session_state["test_data"]) - window_size - forecast_horizon - 1)
        start_idx = st.slider(
            "Punto de inicio",
            min_value=0,
            max_value=max_start,
            value=min(100, max_start),
            help="Posición en la trayectoria para comenzar la predicción"
        )
        
        # Botón para ejecutar predicción
        predict_btn = st.button("Generar predicción", type="primary")
    
    with config_col2:
        st.subheader("Preprocesamiento")
        
        # Opciones de normalización
        normalize = st.checkbox("Normalizar datos", value=False, 
                              help="Aplicar normalización Min-Max a los datos (recomendado)")
        
        if normalize:
            st.info("""
            La normalización es importante para que el modelo funcione correctamente.
            Se aplicará normalización Min-Max usando los valores mínimos y máximos
            de los datos de prueba.
            """)
        
        # Información sobre la secuencia
        if "test_data" in st.session_state:
            df = st.session_state["test_data"]
            st.markdown(f"""
            **Detalles de la secuencia seleccionada:**
            - Puntos disponibles: {len(df)}
            - Ventana de entrada: {window_size} puntos
            - Puntos a predecir: {forecast_horizon}
            - Punto de inicio: {start_idx}
            """)

    # Función de preprocesamiento y predicción
    if predict_btn:
        if "model" not in st.session_state:
            st.error("¡Debes cargar un modelo primero!")
        elif "test_data" not in st.session_state:
            st.error("¡Debes cargar datos de prueba primero!")
        else:
            with st.spinner("Generando predicción..."):
                try:
                    # Obtener datos
                    df = st.session_state["test_data"]
                    model = st.session_state["model"]
                    
                    # Extraer coordenadas
                    X_raw = df["XM"].values
                    Y_raw = df["YM"].values
                    
                    # Normalización si está seleccionada
                    if normalize:
                        X_min, X_max = np.min(X_raw), np.max(X_raw)
                        Y_min, Y_max = np.min(Y_raw), np.max(Y_raw)
                        
                        X_norm = (X_raw - X_min) / (X_max - X_min)
                        Y_norm = (Y_raw - Y_min) / (Y_max - Y_min)
                    else:
                        X_norm, Y_norm = X_raw, Y_raw
                        X_min, X_max = 0, 1  # Valores por defecto si no se normaliza
                        Y_min, Y_max = 0, 1  # Valores por defecto si no se normaliza
                    
                    # Crear secuencia de entrada
                    input_sequence = np.column_stack((X_norm, Y_norm))[start_idx:start_idx+window_size]
                    input_sequence = np.expand_dims(input_sequence, axis=0)  # Añadir dimensión de lote
                    
                    # Obtener secuencia real para comparar
                    real_sequence = np.column_stack((X_raw, Y_raw))[start_idx+window_size:start_idx+window_size+forecast_horizon]
                    
                    # Realizar predicción
                    predictions = []
                    current_input = input_sequence.copy()
                    
                    # Verificar qué tipo de predicción necesita el modelo
                    if len(model.input_shape) == 3:  # Modelo secuencial (LSTM/GRU)
                        # Para cada paso del horizonte de predicción
                        for _ in range(forecast_horizon):
                            # Predecir el siguiente punto
                            next_point = model.predict(current_input, verbose=0)
                            
                            # Si la salida es un vector plano, reformatearlo
                            if len(next_point.shape) == 2 and next_point.shape[1] > 2:
                                next_point = next_point.reshape(-1, 2)
                            
                            # Guardar la predicción
                            predictions.append(next_point[0])
                            
                            # Actualizar la entrada para la siguiente predicción
                            current_input = np.roll(current_input, -1, axis=1)
                            current_input[0, -1, :] = next_point[0]
                    else:  # Modelo dense (recibe entrada plana)
                        # Aplanar la entrada
                        flat_input = current_input.reshape(1, -1)
                        
                        # Predecir todos los puntos de una vez
                        pred_flat = model.predict(flat_input, verbose=0)
                        
                        # Reformatear las predicciones
                        for i in range(forecast_horizon):
                            if i*2 + 1 < pred_flat.shape[1]:
                                predictions.append([pred_flat[0, i*2], pred_flat[0, i*2 + 1]])
                    
                    # Convertir predicciones a array
                    predictions = np.array(predictions)
                    
                    # Des-normalizar si fue normalizado
                    if normalize:
                        X_pred = predictions[:, 0] * (X_max - X_min) + X_min
                        Y_pred = predictions[:, 1] * (Y_max - Y_min) + Y_min
                    else:
                        X_pred, Y_pred = predictions[:, 0], predictions[:, 1]
                    
                    # Guardar resultados en session_state
                    st.session_state["prediction_results"] = {
                        "input_x": X_raw[start_idx:start_idx+window_size],
                        "input_y": Y_raw[start_idx:start_idx+window_size],
                        "real_x": real_sequence[:, 0],
                        "real_y": real_sequence[:, 1],
                        "pred_x": X_pred,
                        "pred_y": Y_pred
                    }
                    
                    st.success("¡Predicción generada correctamente!")
                
                except Exception as e:
                    st.error(f"Error al generar la predicción: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# Sección 4: Resultados y Visualización
st.markdown(get_section_header("4", "📈", "Resultados y Visualización"), unsafe_allow_html=True)

if "prediction_results" in st.session_state:
    results = st.session_state["prediction_results"]
    
    # Crear columnas para gráfica y métricas
    viz_col, metrics_col = st.columns([2, 1])
    
    with viz_col:
        # Visualizar la predicción vs. realidad
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Dibujar la secuencia de entrada
        ax.plot(
            results["input_x"],
            results["input_y"],
            "k-",
            alpha=0.7,
            linewidth=2,
            label="Secuencia de entrada"
        )
        
        # Dibujar la secuencia real
        ax.plot(
            results["real_x"],
            results["real_y"],
            "b-",
            linewidth=2,
            label="Secuencia real"
        )
        
        # Dibujar la secuencia predicha
        ax.plot(
            results["pred_x"],
            results["pred_y"],
            "r--",
            linewidth=2,
            label="Secuencia predicha"
        )
        
        # Marcar el último punto de entrada
        ax.scatter(
            results["input_x"][-1],
            results["input_y"][-1],
            c="k",
            s=100,
            marker="o",
            label="Último punto de entrada"
        )
        
        # Conectar último punto de entrada con primer punto predicho
        ax.plot(
            [results["input_x"][-1], results["pred_x"][0]],
            [results["input_y"][-1], results["pred_y"][0]],
            "r:",
            alpha=0.7
        )
        
        ax.set_title("Predicción de Trayectoria")
        ax.set_xlabel("Posición X")
        ax.set_ylabel("Posición Y")
        ax.legend()
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    with metrics_col:
        # Calcular métricas de error
        st.subheader("Métricas de Error")
        
        # MSE
        mse_x = np.mean((results["real_x"] - results["pred_x"])**2)
        mse_y = np.mean((results["real_y"] - results["pred_y"])**2)
        mse_total = (mse_x + mse_y) / 2
        
        # MAE
        mae_x = np.mean(np.abs(results["real_x"] - results["pred_x"]))
        mae_y = np.mean(np.abs(results["real_y"] - results["pred_y"]))
        mae_total = (mae_x + mae_y) / 2
        
        # RMSE
        rmse_total = np.sqrt(mse_total)
        
        # Mostrar métricas
        st.metric("MSE (Error Cuadrático Medio)", f"{mse_total:.5f}")
        st.metric("RMSE (Raíz del Error Cuadrático Medio)", f"{rmse_total:.5f}")
        st.metric("MAE (Error Absoluto Medio)", f"{mae_total:.5f}")
        
        # Error por coordenada
        st.markdown("### Error por coordenada")
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.metric("MSE (X)", f"{mse_x:.5f}")
            st.metric("MAE (X)", f"{mae_x:.5f}")
        
        with col_y:
            st.metric("MSE (Y)", f"{mse_y:.5f}")
            st.metric("MAE (Y)", f"{mae_y:.5f}")
        
        # Información de la predicción
        st.markdown("### Detalles")
        st.markdown(f"""
        - **Número de puntos predichos:** {len(results["pred_x"])}
        - **Ventana de entrada:** {len(results["input_x"])} puntos
        - **Error promedio:** {mae_total:.5f} unidades
        """)
else:
    st.info("Ejecuta una predicción para ver los resultados.")

# Pie de página
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 | Dashboard de Predicción de Dinámica Caótica</p>", unsafe_allow_html=True)