import numpy as np

def generate_training_history(epochs=100, model_type='lstm'):
    """
    Genera datos simulados del historial de entrenamiento para visualización
    
    Args:
        epochs: Número de épocas a simular
        model_type: Tipo de modelo ('lstm', 'gru', o 'dense')
    
    Returns:
        Diccionario con datos de entrenamiento simulados
    """
    # Factor de ruido y convergencia según el modelo
    if model_type == 'lstm':
        noise_factor = 0.018
        decay_rate = 35
        min_loss = 0.015
    elif model_type == 'gru':
        noise_factor = 0.022
        decay_rate = 33
        min_loss = 0.018
    else:  # dense
        noise_factor = 0.025
        decay_rate = 40
        min_loss = 0.028
    
    # Generar curvas simuladas
    epochs_range = np.arange(1, epochs + 1)
    train_loss = min_loss + (0.5 - min_loss) * np.exp(-epochs_range/decay_rate) + noise_factor * np.random.randn(epochs)
    val_loss = min_loss + (0.5 - min_loss) * np.exp(-epochs_range/(decay_rate*0.9)) + noise_factor * 1.2 * np.random.randn(epochs)
    
    # Asegurar que los valores sean positivos y tengan sentido
    train_loss = np.maximum(0.01, train_loss)
    val_loss = np.maximum(0.01, val_loss)
    
    # Calcular métricas adicionales
    train_mae = train_loss * 1.2 + 0.01 * np.random.randn(epochs)
    val_mae = val_loss * 1.2 + 0.01 * np.random.randn(epochs)
    
    return {
        'epochs': epochs_range,
        'loss': train_loss,
        'val_loss': val_loss,
        'mae': train_mae,
        'val_mae': val_mae
    }

def generate_trajectory_data():
    """
    Genera datos simulados de trayectorias para visualización
    
    Returns:
        Diccionario con trayectorias reales y predichas por diferentes modelos
    """
    # Generar datos de trayectoria real para demostración
    t = np.linspace(0, 2*np.pi, 100)
    x_real = 0.5 * np.sin(2*t) + 0.2 * np.sin(5*t)
    y_real = 0.5 * np.cos(3*t) + 0.1 * np.sin(4*t)
    
    # Simulación de predicciones para diferentes modelos con distintos niveles de error
    noise_lstm = 0.03
    noise_gru = 0.05
    noise_dense = 0.08
    
    # Predicciones LSTM (más cercanas a lo real)
    x_lstm = x_real + noise_lstm * np.random.randn(len(t))
    y_lstm = y_real + noise_lstm * np.random.randn(len(t))
    
    # Predicciones GRU (error intermedio)
    x_gru = x_real + noise_gru * np.random.randn(len(t))
    y_gru = y_real + noise_gru * np.random.randn(len(t))
    
    # Predicciones Red Densa (más error)
    x_dense = x_real + noise_dense * np.random.randn(len(t))
    y_dense = y_real + noise_dense * np.random.randn(len(t))
    
    return {
        'real': {'x': x_real, 'y': y_real},
        'lstm': {'x': x_lstm, 'y': y_lstm},
        'gru': {'x': x_gru, 'y': y_gru},
        'dense': {'x': x_dense, 'y': y_dense}
    }

def generate_realtime_prediction(coords, steps, model_type):
    """
    Genera predicciones simuladas en tiempo real basadas en coordenadas iniciales
    
    Args:
        coords: Lista de tuplas (x,y) con coordenadas iniciales
        steps: Número de pasos a predecir
        model_type: Tipo de modelo ('LSTM', 'GRU', o 'Red Densa')
    
    Returns:
        Diccionario con datos de predicción
    """
    # Coordenadas iniciales
    init_x = [coord[0] for coord in coords]
    init_y = [coord[1] for coord in coords]
    
    # Iniciar con los puntos existentes
    x_pred = list(init_x)
    y_pred = list(init_y)
    
    # Agregar ruido según el modelo seleccionado
    noise_level = 0.02 if model_type == "LSTM" else 0.03 if model_type == "GRU" else 0.05
    
    # Generar siguiente punto basado en los anteriores + un componente de ruido
    for i in range(steps):
        next_x = 0.8*x_pred[-1] - 0.2*x_pred[-2] + 0.2*np.sin(i*0.2) + noise_level * np.random.randn()
        next_y = 0.8*y_pred[-1] - 0.2*y_pred[-2] + 0.2*np.cos(i*0.2) + noise_level * np.random.randn()
        x_pred.append(next_x)
        y_pred.append(next_y)
    
    return {
        'initial': {'x': init_x, 'y': init_y},
        'predicted': {'x': x_pred[len(init_x):], 'y': y_pred[len(init_y):]}
    }