import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models

# --- 1. MÓDULO DE PROCESAMIENTO DE DATOS (Backend) ---
def preparar_datos(df):
    """
    Normalización y segmentación de datos para el Gemelo Digital.
    Limpia y prepara las señales biométricas para la red neuronal.
    """
    scaler = StandardScaler()
    # Escalamos los datos para que tengan media 0 y varianza 1 (estandarización)
    data_scaled = scaler.fit_transform(df)
    
    # Dividimos el dataset: 80% para entrenamiento y 20% para pruebas
    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)
    
    # Redimensionamos los datos para la capa LSTM: (muestras, pasos_de_tiempo, características)
    # En este prototipo usamos 1 paso de tiempo por cada lectura
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    return X_train, X_test, scaler

# --- 2. NÚCLEO INTELIGENTE (Motor de IA) ---
def crear_modelo_gemelo_digital(input_shape):
    """
    Arquitectura de red neuronal recurrente (LSTM).
    Diseñada para analizar secuencias temporales según Nature Medicine.
    """
    model = Sequential([
        # Primera capa LSTM: extrae patrones secuenciales de la biometría
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2), # Técnica de regularización para evitar el sobreajuste (overfitting)
        
        # Segunda capa LSTM para mayor profundidad en el análisis de patrones
        LSTM(32, activation='relu'),
        
        # Capas densas (Fully Connected) para la toma de decisiones final
        Dense(16, activation='relu'),
        
        # Capa de salida con activación Sigmoide (ideal para clasificación binaria)
        # Devuelve una probabilidad entre 0 (Normal) y 1 (Riesgo/Anomalía)
        Dense(1, activation='sigmoid')
    ])
    
    # Compilación con optimizador Adam y pérdida de entropía cruzada binaria
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. INTERFAZ DE ALERTAS (Lógica de Decisión) ---
def detectar_anomalia(probabilidad):
    """
    Simulación de la interfaz de usuario que interpreta el juicio de la IA.
    """
    umbral = 0.80  # Umbral de confianza del 80%
    if probabilidad > umbral:
        print(f"⚠️ [ALERTA CRÍTICA]: Patrón de riesgo detectado. Probabilidad: {probabilidad:.2%}")
    else:
        print(f"✅ [ESTADO ESTABLE]: El Gemelo Digital no detecta anomalías. Probabilidad: {probabilidad:.2%}")

# --- BLOQUE PRINCIPAL DE EJECUCIÓN (PROTOTIPO) ---
if __name__ == "__main__":
    print("--- Inicializando Prototipo de Gemelo Digital Sanitario ---")

    # SIMULACIÓN: Creamos datos sintéticos (Pulso, Presión, Oxígeno, Temperatura)
    columnas = ['frec_cardiaca', 'presion_art', 'spo2', 'temp_corp']
    datos_ficticios = np.random.rand(200, 4) 
    df_biometrico = pd.DataFrame(datos_ficticios, columns=columnas)

    # Paso 1: Procesamiento
    X_train, X_test, scaler = preparar_datos(df_biometrico)

    # Paso 2: Construcción del modelo basado en la forma de los datos
    input_dim = (X_train.shape[1], X_train.shape[2])
    modelo = crear_modelo_gemelo_digital(input_dim)
    
    # Entrenamiento rápido de prueba (1 época) para validar la estructura
    etiquetas_prueba = np.random.randint(0, 2, size=X_train.shape[0])
    modelo.fit(X_train, etiquetas_prueba, epochs=1, verbose=0)

    # Paso 3: Simulación de monitoreo en tiempo real de un nuevo paciente
    print("\nAnalizando flujo de datos del paciente en tiempo real...")
    paciente_nuevo = np.random.rand(1, 4)  # Nuevos datos entrantes
    paciente_escalado = scaler.transform(paciente_nuevo)
    paciente_final = np.expand_dims(paciente_escalado, axis=1)

    # Ejecución de la inferencia de IA
    prediccion = modelo.predict(paciente_final, verbose=0)[0][0]
    
    # Resultado en la interfaz
    detectar_anomalia(prediccion)