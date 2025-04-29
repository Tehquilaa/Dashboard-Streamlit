def get_main_title():
    return """
    <h1> 🧠 Redes Neuronales para Predecir la Dinámica de un Balín</h1>
    """

def get_intro_highlight():
    return """
    <div class="justified-text highlight">
    Este dashboard presenta el diseño, implementación y evaluación de modelos de deep learning 
    para predecir la trayectoria de un balín bajo un campo magnético armónico.  
    La estructura y metodologías se basan en el documento del examen de desarrollo de proyectos.
    </div>
    """

def get_introduccion():
    return """
    <div class="justified-text margin-top-intro">
    Los sistemas caóticos, como el movimiento de un balín bajo la influencia de un campo magnético,
    representan un reto en la predicción debido a su alta sensibilidad a condiciones iniciales.
    Este proyecto utiliza redes neuronales —incluyendo arquitecturas LSTM, GRU y modelos densos—
    para modelar la dinámica caótica a partir de datos experimentales.
    </div>
    """

def get_antecedentes():
    return """
    <div class="justified-text margin-top-antecedentes">
    La predicción de trayectorias en sistemas caóticos es complicada por la naturaleza no lineal y la sensibilidad a las condiciones iniciales. Los métodos tradicionales basados en ecuaciones diferenciales tienen limitaciones, lo que ha impulsado el uso de técnicas de machine learning para capturar patrones complejos en datos experimentales.
    El problema se centra en predecir la trayectoria de un balín, cuyos datos experimentales comprenden 1020 puntos por muestra, con 195 muestras distribuidas en 5 carpetas (12G, 20G, 30G, 50G y 70G) y frecuencias entre 1Hz y 35Hz.
    <br><br>
    En este proyecto, adoptamos un enfoque basado en deep learning, comparando diferentes arquitecturas de redes neuronales.
    </div>
    """

def get_datos_experimentales():
    return """
    <div class="justified-text margin-top-intro">
    Los datos provienen de un experimento físico controlado y se estructuran de la siguiente forma:
    <ul>
      <li><strong style="color:#4b6cb7;">Total de muestras:</strong> 195 (35 por cada campo magnético).</li>
      <li><strong style="color:#4b6cb7;">Puntos por muestra:</strong> 1020 registros de coordenadas (XM, YM).</li>
      <li><strong style="color:#4b6cb7;">Variables experimentales:</strong> Campo Magnético (12G a 70G) y Frecuencia (1Hz a 35Hz).</li>
    </ul>
    <br>
    El preprocesamiento incluye:
    <ul>
      <li>Lectura y consolidación de archivos txt (cada uno con 2 columnas: XM y YM).</li>
      <li>Extracción de metadatos (campo magnético y frecuencia, a partir del nombre de carpeta y archivo).</li>
      <li>Normalización de los datos con la técnica Min-Max, preservando la forma de la distribución.</li>
      <li>División del dataset en subconjuntos: entrenamiento (70%), validación (20%) y prueba (10%).</li>
    </ul>
    </div>
    """

def get_section_header(num, icon, title):
    return f"""
    <div class="section-header"><h2>{icon} {num}. {title}</h2></div>
    """

def get_section_header_modelo(icon, title):
    return f"""
    <div class="section-header"><h2>{icon} {title}</h2></div>
    """

def get_arquitectura_modelo():
    return """
    <div class="justified-text margin-top-intro">
    Se evaluaron tres tipos de modelos:
    <ul>
      <li><strong style="color:#4b6cb7;">Redes LSTM:</strong> Ideales para capturar dependencias a largo plazo en series temporales.</li>
      <li><strong style="color:#4b6cb7;">Redes GRU:</strong> Variante de las LSTM con menor costo computacional.</li>
      <li><strong style="color:#4b6cb7;">Modelo Denso (Feedforward):</strong> Usado como referencia para comparar el desempeño.</li>
    </ul>
    <br>
    Para la red LSTM se configuraron dos capas con 80 y 29 neuronas, respectivamente, y para la GRU se utilizaron 62 y 47 neuronas.
    </div>
    """

def intro_highlight_modelo():
    return """
    <div class="justified-text highlight">
    Esta sección presenta el diseño, implementación y evaluación de tres tipos de arquitecturas 
    de redes neuronales utilizadas para modelar y predecir la dinámica caótica del balín.
    El análisis incluye los hiperparámetros óptimos, la metodología de entrenamiento y los resultados comparativos.
    </div>
    """

def get_arquitecturas():

    return """
    <div class="justified-text">
    Se evaluaron tres tipos diferentes de arquitecturas de redes neuronales, cada una con características 
    y ventajas específicas para el modelado de sistemas dinámicos como el movimiento del balín:
    
    - **Redes LSTM (Long Short-Term Memory):** Diseñadas específicamente para capturar dependencias 
    temporales a largo plazo en series de tiempo, incorporando mecanismos de puertas que regulan el flujo 
    de información y permiten "recordar" patrones relevantes.
    
    - **Redes GRU (Gated Recurrent Unit):** Una variante optimizada de las LSTM que mantiene su 
    capacidad de modelar dependencias temporales pero con menor complejidad computacional, combinando 
    las puertas de olvido y entrada en una única estructura.
    
    - **Red Densa (Feedforward):** Utilizada como modelo base de comparación, consiste en capas 
    totalmente conectadas que procesan la información de manera directa sin mecanismos de memoria explícitos.
    </div>
    """

def get_hiperparametros():
    return """
    <div class="justified-text">
    Los hiperparámetros fueron optimizados mediante búsqueda aleatoria (Random Search), evaluando múltiples 
    combinaciones para encontrar la configuración óptima. Se implementaron estrategias como early stopping 
    para prevenir el sobreajuste y se ajustaron las tasas de aprendizaje para cada modelo.
    </div>
    """

def get_tarjeta_lstm():
    return """
    <div class="model-card">
        <div class="model-title">Modelo LSTM</div>
        <hr>
        <p><strong>Dropout:</strong> 0.226</p>
        <p><strong>Learning Rate:</strong> 0.006</p>
        <p><strong>Capas:</strong> [80, 29]</p>
        <p><strong>Optimizer:</strong> Adamax</p>
        <p><strong>Batch Size:</strong> 64</p>
        <p><strong>Epochs:</strong> 100 (con early stopping)</p>
    </div>
    """

def get_tarjeta_gru():
    return """
    <div class="model-card">
        <div class="model-title">Modelo GRU</div>
        <hr>
        <p><strong>Dropout:</strong> 0.41</p>
        <p><strong>Learning Rate:</strong> 0.0004</p>
        <p><strong>Capas:</strong> [62, 47]</p>
        <p><strong>Optimizer:</strong> Adamax</p>
        <p><strong>Batch Size:</strong> 32</p>
        <p><strong>Epochs:</strong> 100 (con early stopping)</p>
    </div>
    """

def get_tarjeta_densa():
    return """
    <div class="model-card">
        <div class="model-title">Modelo Denso</div>
        <hr>
        <p><strong>Dropout:</strong> 0.57</p>
        <p><strong>Learning Rate:</strong> 0.007</p>
        <p><strong>Capas:</strong> [56, 16, 11, 11]</p>
        <p><strong>Optimizer:</strong> Adamax</p>
        <p><strong>Batch Size:</strong> 128</p>
        <p><strong>Epochs:</strong> 80 (con early stopping)</p>
    </div>
    """

def get_proceso_train():
    return """
    <div class="justified-text">
    El proceso de entrenamiento siguió un enfoque riguroso para garantizar la robustez de los modelos:
    
    1. **Preparación de datos**: Los datos se normalizaron para facilitar el entrenamiento y se dividieron en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%).
    
    2. **Flujo de datos**: Sedividieron los datos categoricamente por frecuencia y se escogio la media (media 10-15hz). Posteriormente se partio a la mitad las coordenadas, para implementar la primera mitad como entrada y la siguiente como salida
    
    3. **Regularización**: Se aplicaron técnicas como dropout para prevenir el sobreajuste y se monitoreó la pérdida en el conjunto de validación.
    
    4. **Early stopping**: Se configuró un mecanismo de parada temprana con paciencia de 15 épocas para detener el entrenamiento cuando no hubiera mejora en la métrica de validación.
    
    5. **Validación cruzada**: Para garantizar la robustez de los resultados, los modelos se evaluaron utilizando validación cruzada con 5 pliegues.
    </div>
    """

def get_inter_graficas():
    return """
        - **MSE (Error Cuadrático Medio)**: Mide la diferencia cuadrática promedio entre los valores predichos y reales. Menor es mejor.
        - **MAE (Error Absoluto Medio)**: Mide la diferencia absoluta promedio. Menor es mejor.
       
        **Conclusión**: La arquitectura LSTM tiene el "mejor" rendimiento con el menor error.
        """

def get_evaluacion_modelos():
    return """
    <div class="justified-text highlight">
    Esta sección presenta los resultados detallados del entrenamiento y evaluación de los modelos de deep learning.
    Se incluyen visualizaciones interactivas del proceso de entrenamiento, predicciones en tiempo real y
    métricas comparativas entre las distintas arquitecturas implementadas.
    </div>
    """