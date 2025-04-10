
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adamax, Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# 1. Cargar datos
data_train = pd.read_csv('dataset_media_train.csv')
data_val = pd.read_csv('dataset_media_val.csv')
data_test = pd.read_csv('dataset_media_test.csv')

# Calcular punto de división para coordenadas
total_coords = (len(data_train.columns) - 1) // 2
n_coords_input = total_coords // 2  # Número de pares de coordenadas para entrada

# 3. Preparar datos de entrenamiento
# Datos de entrada (X) y salida (Y)
X = data_train.iloc[:, 1:1+n_coords_input*2].values
Y = data_train.iloc[:, 1+n_coords_input*2:].values

# Normalizar datos
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Reestructurar para LSTM: cada muestra tendrá n_coords_input pasos de tiempo y 2 características
X_reshaped = X_scaled.reshape(X_scaled.shape[0], n_coords_input, 2)

# 4. Preparar datos de validación
X_val = data_val.iloc[:, 1:1+n_coords_input*2].values
Y_val = data_val.iloc[:, 1+n_coords_input*2:].values

X_val_scaled = scaler_X.transform(X_val)
Y_val_scaled = scaler_Y.transform(Y_val)
X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], n_coords_input, 2)

# 5. Crear y compilar modelo
model = Sequential([
    GRU(62, input_shape=(n_coords_input, 2), return_sequences=True),
    GRU(47, return_sequences=False),
    Dense(Y_scaled.shape[1])
])

model.summary()
optimizer = Adamax(learning_rate=0.0004)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Configurar callbacks
callback = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose=1
)


# 6. Entrenar modelo
history = model.fit(
    X_reshaped,
    Y_scaled,
    epochs=200,
    batch_size=8,
    validation_data=(X_val_reshaped, Y_val_scaled),
    verbose=1,
    callbacks=[callback]
)

model_save = model.save('lstm_model.h5')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv('gru_history.csv', index=False)


# 7. Preparar datos de prueba
X_test = data_test.iloc[:, 1:1+n_coords_input*2].values
Y_test = data_test.iloc[:, 1+n_coords_input*2:].values

X_test_scaled = scaler_X.transform(X_test)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], n_coords_input, 2)

# 8. Predicción
Y_pred_scaled = model.predict(X_test_reshaped)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# Evaluación y visualización
mse= model.evaluate(X_test_reshaped, scaler_Y.transform(Y_test))
print(f'Error en el conjunto de prueba (MSE): {mse}')
from sklearn.metrics import mean_absolute_error
Y_test_scaled = scaler_Y.transform(Y_test)
mae_scaled = mean_absolute_error(Y_test_scaled, Y_pred_scaled)
print(f'Error absoluto medio (MAE): {mae_scaled}')





plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida (entrenamiento)')
plt.plot(history.history['val_loss'], label='Pérdida (validación)')
plt.title('Historia del Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Error (MSE)')
plt.legend()
plt.show()

def visualizar_predicciones(Y_test, Y_pred, num_ejemplos=3):
    """
    Visualiza la comparación entre coordenadas reales y predichas.
    Una trayectoria por plot.
    """
    indices = np.random.choice(len(Y_test), num_ejemplos, replace=False)

    for idx in indices:
        coord_reales = Y_test[idx].reshape(-1, 2)
        coord_pred = Y_pred[idx].reshape(-1, 2)

        plt.figure(figsize=(8, 6))
        plt.plot(coord_reales[:, 0], coord_reales[:, 1],
                 'b-', label='Real',
                 marker='o', markersize=5, linewidth=2, alpha=0.7)
        plt.plot(coord_pred[:, 0], coord_pred[:, 1],
                 'r-', label='Predicción',
                 marker='o', markersize=5, linewidth=2, alpha=0.7)

        plt.title(f'Trayectoria {idx}', fontsize=14)
        plt.xlabel('Coordenada X', fontsize=12)
        plt.ylabel('Coordenada Y', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='-', alpha=0.7)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

visualizar_predicciones(Y_test, Y_pred)
