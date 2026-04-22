import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import sqlite3
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce logs de TensorFlow

# --- 1. CAPA DE PERSISTENCIA (Base de Datos) ---
class PersistenciaDB:
    def __init__(self, db_path='gemelo_digital.db'):
        self.conn = sqlite3.connect(db_path)
        self._crear_tabla()
    
    def _crear_tabla(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pacientes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                frec_cardiaca REAL,
                presion_art REAL,
                spo2 REAL,
                temp_corp REAL,
                prediccion_riesgo REAL,
                alerta_generada INTEGER
            )
        ''')
        self.conn.commit()
    
    def guardar_lectura(self, datos, prediccion, alerta):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO pacientes (frec_cardiaca, presion_art, spo2, temp_corp, prediccion_riesgo, alerta_generada)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (*datos, float(prediccion), int(alerta)))
        self.conn.commit()
    
    def obtener_historico(self, limit=100):
        return pd.read_sql_query(f"SELECT * FROM pacientes ORDER BY timestamp DESC LIMIT {limit}", self.conn)
    
    def cerrar(self):
        self.conn.close()

# --- 2. MÓDULO DE PROCESAMIENTO DE SEÑALES ---
class ProcesadorSenales:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def filtrar_ruido(self, df, metodo='media_movil', ventana=3):
        """Filtrado digital básico para eliminar ruido en señales"""
        if metodo == 'media_movil':
            return df.rolling(window=ventana, min_periods=1).mean()
        return df
    
    def detectar_outliers(self, df, umbral=3):
        """Detección de outliers usando Z-score"""
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = (z_scores > umbral).sum().sum()
        return outliers
    
    def preparar_datos(self, df, test_size=0.2, time_steps=10):
        """Pipeline completo de procesamiento con ventanas temporales"""
        # Limpieza básica
        df = df.dropna().reset_index(drop=True)
        
        # Filtrado de ruido
        df_filtrado = self.filtrar_ruido(df)
        
        # Normalización (solo fit en entrenamiento)
        if not self.fitted:
            datos_escalados = self.scaler.fit_transform(df_filtrado)
            self.fitted = True
        else:
            datos_escalados = self.scaler.transform(df_filtrado)
        
        # Crear secuencias para LSTM
        X, y_aux = [], []
        for i in range(len(datos_escalados) - time_steps):
            X.append(datos_escalados[i:i + time_steps])
            y_aux.append(i + time_steps)  # Índice para la etiqueta
        
        X = np.array(X)
        
        # División temporal
        split_idx = int(len(X) * (1 - test_size))
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        
        return X_train, X_test, self.scaler
    
    def transformar_nuevos_datos(self, df, time_steps=10):
        """Transformar nuevos datos (una sola muestra)"""
        if not self.fitted:
            raise ValueError("El scaler no ha sido entrenado")
        
        df_filtrado = self.filtrar_ruido(df)
        datos_escalados = self.scaler.transform(df_filtrado)
        
        # Para predicción individual, necesitamos la ventana completa
        if len(datos_escalados) < time_steps:
            # Padding con ceros si no hay suficientes datos
            padding = np.zeros((time_steps - len(datos_escalados), datos_escalados.shape[1]))
            datos_escalados = np.vstack([padding, datos_escalados])
        
        X = datos_escalados[-time_steps:].reshape(1, time_steps, -1)
        return X

# --- 3. NÚCLEO INTELIGENTE (Motor de IA) ---
class MotorIA:
    def __init__(self, input_shape):
        self.model = self._crear_modelo(input_shape)
        self.history = None
    
    def _crear_modelo(self, input_shape):
        model = Sequential([
            # BatchNormalization para estabilizar el entrenamiento
            Bidirectional(LSTM(128, activation='tanh', return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            
            Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
            Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            
            Bidirectional(LSTM(32, activation='tanh')),
            Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Learning rate más pequeño
        initial_learning_rate = 0.0005
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # CORREGIDO: Usar métricas estándar sin F1Score que causa problemas
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model
    
    def entrenar(self, X_train, y_train, X_val, y_val, epochs=50):
        """Entrenamiento con pesos de clase"""
        
        # Calcular pesos automáticos para clases desbalanceadas
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
        
        print(f"📊 Distribución clases entrenamiento:")
        print(f"   - Clase 0 (Normal): {sum(y_train==0)} muestras")
        print(f"   - Clase 1 (Riesgo): {sum(y_train==1)} muestras")
        print(f"📊 Pesos de clase aplicados: {class_weight}")
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )
        
        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # CORREGIDO: Checkpoint sin f1_score problemático
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'mejor_modelo.h5',
            monitor='val_loss',  # Cambiado de val_f1_score a val_loss
            mode='min',
            save_best_only=True,
            verbose=1
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop, reduce_lr, checkpoint],
            class_weight=class_weight,
            verbose=1
        )
        return self.history
    
    def predecir(self, X):
        """Predicción simple"""
        return self.model.predict(X, verbose=0)
    
    def predecir_con_umbral(self, X, umbral=0.5):
        """Predicción con umbral ajustable"""
        probabilidad = self.model.predict(X, verbose=0)
        prediccion = (probabilidad > umbral).astype(int)
        return probabilidad, prediccion
    
    def evaluar(self, X_test, y_test, umbral=0.5):
        """Evaluación completa del modelo"""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report)
        
        y_pred_prob = self.predecir(X_test)
        y_pred = (y_pred_prob > umbral).astype(int)
        
        print("\n" + "="*60)
        print("📊 REPORTE DE EVALUACIÓN DEL MODELO")
        print("="*60)
        
        # Métricas principales
        print(f"🎯 Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"🎯 Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"🎯 Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"🎯 F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"🎯 AUC-ROC:   {roc_auc_score(y_test, y_pred_prob):.4f}")
        
        # Buscar mejor umbral
        self._encontrar_mejor_umbral(y_test, y_pred_prob)
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Normal (0)', 'Riesgo (1)'],
                                   zero_division=0))
        
        # Matriz de confusión mejorada
        cm = confusion_matrix(y_test, y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
        axes[0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Valor Real', fontsize=12)
        axes[0].set_xlabel('Predicción', fontsize=12)
        
        # Curva ROC
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.3f}')
        axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Clasificador Aleatorio')
        axes[1].set_xlabel('Tasa de Falsos Positivos', fontsize=12)
        axes[1].set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
        axes[1].set_title('Curva ROC', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return y_pred, y_pred_prob
    
    def _encontrar_mejor_umbral(self, y_true, y_prob):
        """Encuentra el mejor umbral para balancear precisión y recall"""
        from sklearn.metrics import f1_score, precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        if len(thresholds) > 0:
            mejor_idx = np.argmax(f1_scores[:-1])
            mejor_umbral = thresholds[mejor_idx]
            mejor_f1 = f1_scores[mejor_idx]
            print(f"\n🔍 Mejor F1-Score: {mejor_f1:.4f} con umbral = {mejor_umbral:.3f}")
        else:
            print("\n🔍 No hay suficientes thresholds para calcular mejor umbral")
        
        # Probar diferentes umbrales
        print("\n📊 Prueba de diferentes umbrales:")
        for umbral in [0.3, 0.4, 0.5, 0.6, 0.7]:
            y_pred_umbral = (y_prob > umbral).astype(int)
            f1 = f1_score(y_true, y_pred_umbral, zero_division=0)
            print(f"   Umbral {umbral:.1f} → F1-Score: {f1:.4f}")
        
        return mejor_umbral if len(thresholds) > 0 else 0.5
    
    def guardar_modelo(self, path='modelo_gemelo_digital.h5'):
        """Guarda modelo"""
        self.model.save(path)
        print(f"✅ Modelo guardado en {path}")
    
    def cargar_modelo(self, path='modelo_gemelo_digital.h5'):
        """Carga modelo"""
        self.model = load_model(path)
        print(f"✅ Modelo cargado desde {path}")
    
    def plot_historial_entrenamiento(self):
        """Visualiza el historial de entrenamiento"""
        if not self.history:
            print("❌ No hay historial de entrenamiento")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Pérdida
        axes[0,0].plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
        axes[0,0].plot(self.history.history['val_loss'], label='Validación', linewidth=2)
        axes[0,0].set_title('Pérdida del Modelo', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Pérdida')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0,1].plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0,1].plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0,1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1,0].plot(self.history.history['precision'], label='Entrenamiento', linewidth=2)
            axes[1,0].plot(self.history.history['val_precision'], label='Validación', linewidth=2)
            axes[1,0].set_title('Precisión', fontsize=12, fontweight='bold')
            axes[1,0].set_xlabel('Época')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1,1].plot(self.history.history['recall'], label='Entrenamiento', linewidth=2)
            axes[1,1].plot(self.history.history['val_recall'], label='Validación', linewidth=2)
            axes[1,1].set_title('Recall', fontsize=12, fontweight='bold')
            axes[1,1].set_xlabel('Época')
            axes[1,1].set_ylabel('Recall')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
# --- 4. ANÁLISIS EXPLORATORIO Y CLUSTERING ---
class AnalisisExploratorio:
    @staticmethod
    def analizar_distribucion(df):
        """Visualización de distribuciones de variables biométricas"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, col in enumerate(df.columns):
            axes[idx].hist(df[col], bins=30, alpha=0.7, edgecolor='black', color='steelblue')
            axes[idx].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frecuencia', fontsize=10)
            axes[idx].axvline(df[col].mean(), color='red', linestyle='--', label=f'Media: {df[col].mean():.2f}')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def clustering_no_supervisado(df, n_clusters=3):
        """K-Means clustering para identificar patrones de pacientes"""
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_scaled)
        
        # Visualización
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        scatter = axes[0].scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, label='Centroides', edgecolors='black', linewidths=2)
        axes[0].set_xlabel(df.columns[0], fontsize=11)
        axes[0].set_ylabel(df.columns[1], fontsize=11)
        axes[0].set_title(f'Clustering de Pacientes (K={n_clusters})', fontsize=13, fontweight='bold')
        axes[0].legend()
        plt.colorbar(scatter, ax=axes[0])
        
        # Distribución de clusters
        axes[1].bar(range(n_clusters), np.bincount(clusters), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1].set_xlabel('Cluster', fontsize=11)
        axes[1].set_ylabel('Número de Pacientes', fontsize=11)
        axes[1].set_title('Distribución de Clusters', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return clusters, kmeans

# --- 5. INTERFAZ DE MONITORIZACIÓN Y ALERTAS ---
class MonitorSalud:
    def __init__(self, modelo_ia, procesador, db):
        self.modelo = modelo_ia
        self.procesador = procesador
        self.db = db
        self.umbral_riesgo = 0.65
        self.historial_predicciones = []
        # AÑADIR SISTEMA DE RECOMENDACIONES
        self.recomendador = SistemaRecomendaciones()
        self.historial_informes = []
        self.historial_patologias = []
    
    def procesar_paciente_tiempo_real(self, nuevos_datos, time_steps=10):
        """Procesa flujo continuo de datos del paciente con recomendaciones"""
        try:
            # Transformar nuevos datos
            datos_lstm = self.procesador.transformar_nuevos_datos(nuevos_datos, time_steps)
            
            # Predicción
            probabilidad_riesgo = self.modelo.predecir(datos_lstm)[0][0]
            self.historial_predicciones.append(probabilidad_riesgo)
            
            # Evaluar alerta
            alerta = probabilidad_riesgo > self.umbral_riesgo
            
            # --- NUEVO: Generar recomendaciones diagnósticas ---
            ultima_muestra = nuevos_datos.iloc[-1].to_dict()
            
            # Informe completo
            informe_completo = self.recomendador.generar_informe_completo(
                ultima_muestra, 
                probabilidad_riesgo
            )
            
            # Recomendaciones cortas
            recomendaciones_cortas = self.recomendador.generar_recomendaciones_cortas(
                ultima_muestra,
                probabilidad_riesgo
            )
            
            # Recomendaciones específicas por patología
            recomendaciones_patologia = self.recomendador.generar_recomendaciones_especificas_por_patologia(
                ultima_muestra
            )
            
            # Guardar en historial
            self.historial_informes.append({
                'timestamp': pd.Timestamp.now(),
                'riesgo': probabilidad_riesgo,
                'alerta': alerta,
                'muestra': ultima_muestra,
                'informe': informe_completo,
                'recomendaciones_cortas': recomendaciones_cortas,
                'patologias_sospechosas': recomendaciones_patologia
            })
            
            # Guardar en BD
            self.db.guardar_lectura(nuevos_datos.iloc[-1].values, probabilidad_riesgo, alerta)
            
            # Mostrar resultado MEJORADO
            self._mostrar_alerta_con_recomendaciones(
                probabilidad_riesgo, 
                alerta, 
                recomendaciones_cortas,
                recomendaciones_patologia,
                informe_completo
            )
            
            return probabilidad_riesgo, alerta, informe_completo
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            return 0.0, False, None
    
    def _mostrar_alerta_con_recomendaciones(self, probabilidad, alerta, 
                                             rec_cortas, rec_patologia, informe_completo):
        """Muestra alerta mejorada con todas las recomendaciones"""
        
        print("\n" + "="*60)
        if alerta:
            print(f"🚨 [ALERTA CRÍTica] ¡Riesgo detectado! Probabilidad: {probabilidad:.2%}")
        else:
            print(f"✅ [ESTADO ESTABLE] Parámetros normales: {probabilidad:.2%}")
        
        # Barra de riesgo
        barras = int(probabilidad * 40)
        barra_color = '🟥' if alerta else '🟩'
        print(f"   Nivel riesgo: [{barra_color * barras}{'⬜'*(40-barras)}] {probabilidad:.1%}")
        
        # Recomendaciones inmediatas
        print("\n📋 RECOMENDACIONES INMEDIATAS:")
        for rec in rec_cortas[:3]:
            print(f"   {rec}")
        
        # Posibles patologías (si las hay)
        if rec_patologia:
            print("\n🩺 POSIBLES PATOLOGÍAS SUGERIDAS:")
            for pat in rec_patologia:
                print(f"   🔹 {pat['nombre']}")
                for rec in pat['recomendaciones'][:2]:
                    print(f"      {rec}")
        
        print("\n💡 Para informe completo, escriba 'informe'")
        print("="*60)
    
    def mostrar_informe_completo(self, idx=-1):
        """Muestra informe diagnóstico completo guardado"""
        if self.historial_informes:
            if idx == -1:
                idx = len(self.historial_informes) - 1
            if 0 <= idx < len(self.historial_informes):
                print(self.historial_informes[idx]['informe'])
                
                # Mostrar también patologías sospechosas si existen
                if self.historial_informes[idx].get('patologias_sospechosas'):
                    print("\n" + "="*70)
                    print("🩺 ANÁLISIS DETALLADO POR PATOLOGÍA:")
                    print("-"*70)
                    for pat in self.historial_informes[idx]['patologias_sospechosas']:
                        print(f"\n📌 {pat['nombre']}:")
                        for rec in pat['recomendaciones']:
                            print(f"   {rec}")
            else:
                print("❌ Índice fuera de rango")
        else:
            print("❌ No hay informes guardados")
    
    def exportar_informe_paciente(self, filename="informe_paciente.txt"):
        """Exporta el último informe a archivo de texto"""
        if self.historial_informes:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.historial_informes[-1]['informe'])
                if self.historial_informes[-1].get('patologias_sospechosas'):
                    f.write("\n\n" + "="*70)
                    f.write("\n🩺 ANÁLISIS DETALLADO POR PATOLOGÍA\n")
                    f.write("="*70 + "\n")
                    for pat in self.historial_informes[-1]['patologias_sospechosas']:
                        f.write(f"\n📌 {pat['nombre']}:\n")
                        for rec in pat['recomendaciones']:
                            f.write(f"   {rec}\n")
            print(f"✅ Informe guardado en: {filename}")
        else:
            print("❌ No hay informe para exportar")
    
    def dashboard_con_diagnosticos(self):
        """Dashboard mejorado con resumen de diagnósticos"""
        historico = self.db.obtener_historico()
        
        if len(historico) > 0 and len(self.historial_predicciones) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Evolución del riesgo
            axes[0,0].plot(range(len(self.historial_predicciones)), self.historial_predicciones, 
                        marker='o', linestyle='-', color='red', linewidth=2, markersize=4)
            axes[0,0].axhline(y=self.umbral_riesgo, color='orange', linestyle='--', 
                           linewidth=2, label=f'Umbral ({self.umbral_riesgo})')
            axes[0,0].set_title('Evolución del Riesgo Detectado', fontsize=12, fontweight='bold')
            axes[0,0].set_ylabel('Probabilidad de Riesgo', fontsize=10)
            axes[0,0].set_xlabel('Muestras', fontsize=10)
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Alertas generadas
            alertas = historico['alerta_generada'].values[:len(self.historial_predicciones)]
            colores = ['red' if a else 'green' for a in alertas]
            axes[0,1].bar(range(len(alertas)), alertas, color=colores, alpha=0.7)
            axes[0,1].set_title('Historial de Alertas Generadas', fontsize=12, fontweight='bold')
            axes[0,1].set_ylabel('Alerta (1=Activada)', fontsize=10)
            axes[0,1].set_xlabel('Muestras', fontsize=10)
            axes[0,1].set_ylim(-0.1, 1.1)
            
            # Distribución de predicciones
            axes[1,0].hist(self.historial_predicciones, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            axes[1,0].axvline(x=self.umbral_riesgo, color='red', linestyle='--', linewidth=2)
            axes[1,0].set_title('Distribución de Predicciones', fontsize=12, fontweight='bold')
            axes[1,0].set_xlabel('Probabilidad de Riesgo', fontsize=10)
            axes[1,0].set_ylabel('Frecuencia', fontsize=10)
            
            # Estadísticas y diagnósticos
            stats_text = f"📊 ESTADÍSTICAS:\n"
            stats_text += f"Media riesgo: {np.mean(self.historial_predicciones):.3f}\n"
            stats_text += f"Mediana: {np.median(self.historial_predicciones):.3f}\n"
            stats_text += f"Máximo: {max(self.historial_predicciones):.3f}\n"
            stats_text += f"Alertas: {sum(alertas)}/{len(alertas)} ({sum(alertas)/len(alertas)*100:.1f}%)\n\n"
            stats_text += f"🩺 ÚLTIMO DIAGNÓSTICO:\n"
            if self.historial_informes:
                ultimo_informe = self.historial_informes[-1]
                if ultimo_informe.get('patologias_sospechosas'):
                    stats_text += f"Patologías sugeridas:\n"
                    for pat in ultimo_informe['patologias_sospechosas'][:2]:
                        stats_text += f"  • {pat['nombre']}\n"
            
            axes[1,1].text(0.05, 0.5, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=10, verticalalignment='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1,1].set_title('Estadísticas y Diagnósticos', fontsize=12, fontweight='bold')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Datos insuficientes para mostrar dashboard")

# --- EJECUCIÓN PRINCIPAL ---
def main():
    print("🚀 INICIANDO GEMELO DIGITAL SANITARIO CON IA")
    print("="*60)
    
    # 1. Generar datos más realistas con anomalías claras
    np.random.seed(42)
    n_muestras = 2000
    time_steps = 10
    
    print("📊 Generando dataset de pacientes...")
    
    # Datos normales (80% de los datos)
    n_normales = int(n_muestras * 0.8)
    n_anomalias = n_muestras - n_normales
    
    # Parámetros para pacientes sanos
    frec_cardiaca_normal = np.random.normal(75, 8, n_normales)
    presion_art_normal = np.random.normal(118, 10, n_normales)
    spo2_normal = np.random.normal(97, 1.5, n_normales)
    temp_corp_normal = np.random.normal(36.6, 0.3, n_normales)
    
    # Parámetros para pacientes con anomalías (patologías)
    frec_cardiaca_anomalia = np.random.normal(115, 15, n_anomalias)  # Taquicardia
    presion_art_anomalia = np.random.normal(155, 20, n_anomalias)    # Hipertensión
    spo2_anomalia = np.random.normal(88, 5, n_anomalias)             # Hipoxia
    temp_corp_anomalia = np.random.normal(38.2, 0.8, n_anomalias)    # Fiebre
    
    # Combinar
    frec_cardiaca = np.concatenate([frec_cardiaca_normal, frec_cardiaca_anomalia])
    presion_art = np.concatenate([presion_art_normal, presion_art_anomalia])
    spo2 = np.concatenate([spo2_normal, spo2_anomalia])
    temp_corp = np.concatenate([temp_corp_normal, temp_corp_anomalia])
    
    # Mezclar los datos
    indices = np.random.permutation(n_muestras)
    df = pd.DataFrame({
        'frec_cardiaca': frec_cardiaca[indices],
        'presion_art': presion_art[indices],
        'spo2': spo2[indices],
        'temp_corp': temp_corp[indices]
    })
    
    # Etiquetas (1=anomalía, 0=normal)
    y = np.zeros(n_muestras)
    y[n_normales:] = 1
    y = y[indices]
    
    print(f"✅ Dataset creado: {n_muestras} muestras")
    print(f"   - Pacientes normales: {sum(y==0)} ({sum(y==0)/n_muestras*100:.1f}%)")
    print(f"   - Pacientes con riesgo: {sum(y==1)} ({sum(y==1)/n_muestras*100:.1f}%)")
    
    # 2. Análisis exploratorio
    print("\n📈 Realizando análisis exploratorio...")
    analizador = AnalisisExploratorio()
    analizador.analizar_distribucion(df)
    clusters, _ = analizador.clustering_no_supervisado(df, n_clusters=2)
    
    # 3. Procesamiento de datos
    print("\n🔄 Procesando señales biomédicas...")
    procesador = ProcesadorSenales()
    outliers = procesador.detectar_outliers(df)
    print(f"📊 Outliers detectados: {outliers}")
    
    X_train, X_test, scaler = procesador.preparar_datos(df, test_size=0.2, time_steps=time_steps)
    
    # Ajustar etiquetas para las secuencias
    y_train = y[time_steps:len(X_train)+time_steps]
    y_test = y[len(X_train)+time_steps:len(X_train)+time_steps+len(X_test)]
    
    print(f"   - Secuencias de entrenamiento: {X_train.shape}")
    print(f"   - Secuencias de prueba: {X_test.shape}")
    
    # 4. Crear y entrenar modelo
    print("\n🧠 Construyendo modelo de Deep Learning...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    motor_ia = MotorIA(input_shape)
    
    print("📚 Entrenando modelo (esto puede tomar unos minutos)...")
    motor_ia.entrenar(X_train, y_train, X_test, y_test, epochs=50)
    
    # 5. Evaluar modelo
    print("\n🎯 Evaluando rendimiento del modelo...")
    y_pred, y_pred_prob = motor_ia.evaluar(X_test, y_test)
    
    # 6. Persistencia y monitoreo con SISTEMA DE RECOMENDACIONES
    print("\n💾 Inicializando base de datos y sistema de monitoreo...")
    db = PersistenciaDB()
    monitor = MonitorSalud(motor_ia, procesador, db)
    
    # 7. Simular monitoreo en tiempo real CON RECOMENDACIONES DIAGNÓSTICAS
    print("\n🔄 INICIANDO MONITOREO EN TIEMPO REAL CON DIAGNÓSTICOS")
    print("Simulando llegada de datos de pacientes...")
    print("="*60)
    
    # Usar los datos de prueba para simular monitoreo
    df_test = df.iloc[-100:].copy()
    
    # Variables para seguimiento
    casos_criticos = 0
    casos_riesgo = 0
    
    for i in range(min(20, len(df_test))):
        print(f"\n📌 Lectura #{i+1} de 20")
        print("-" * 40)
        muestra = df_test.iloc[i:i+time_steps].copy()
        
        if len(muestra) == time_steps:
            riesgo, alerta, informe = monitor.procesar_paciente_tiempo_real(muestra, time_steps)
            
            # Contadores para estadísticas
            if riesgo > 0.8:
                casos_criticos += 1
            elif riesgo > 0.65:
                casos_riesgo += 1
            
            # Para casos críticos, mostrar informe completo automáticamente
            if riesgo > 0.8:
                print("\n" + "🔴" * 35)
                print("📋 INFORME DIAGNÓSTICO COMPLETO - CASO CRÍTICO:")
                print("🔴" * 35)
                monitor.mostrar_informe_completo()
                
                # Preguntar si quiere exportar
                if i == 0 or riesgo > 0.9:
                    resp = input("\n💾 ¿Desea exportar este informe? (s/n): ").lower()
                    if resp == 's':
                        monitor.exportar_informe_paciente(f"informe_critico_{i+1}.txt")
            
            # Pequeña pausa para simular tiempo real
            time.sleep(0.5)
    
    # 8. Mostrar resumen del monitoreo
    print("\n" + "="*60)
    print("📊 RESUMEN DEL MONITOREO REALIZADO:")
    print("="*60)
    print(f"   ✅ Pacientes normales: {20 - casos_criticos - casos_riesgo}")
    print(f"   ⚠️  Pacientes con riesgo moderado: {casos_riesgo}")
    print(f"   🔴 Pacientes con riesgo crítico: {casos_criticos}")
    
    # 9. Dashboard mejorado con diagnósticos
    print("\n📊 Generando dashboard con análisis de diagnósticos...")
    monitor.dashboard_con_diagnosticos()
    
    # 10. Exportar resumen completo del paciente
    print("\n💾 Exportando documentación completa...")
    monitor.exportar_informe_paciente("diagnostico_final_paciente.txt")
    
    # 11. Mostrar historial de patologías sugeridas
    if monitor.historial_informes:
        print("\n🩺 HISTORIAL DE PATOLOGÍAS SUGERIDAS:")
        print("-" * 50)
        patologias_detectadas = {}
        
        for idx, informe in enumerate(monitor.historial_informes):
            if informe.get('patologias_sospechosas'):
                for pat in informe['patologias_sospechosas']:
                    nombre = pat['nombre']
                    patologias_detectadas[nombre] = patologias_detectadas.get(nombre, 0) + 1
        
        if patologias_detectadas:
            for patologia, frecuencia in sorted(patologias_detectadas.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {patologia}: detectada {frecuencia} vez/veces")
        else:
            print("   No se detectaron patologías específicas durante el monitoreo")
    
    # 12. Guardar modelo y scaler
    print("\n💾 Guardando modelo y artefactos...")
    motor_ia.guardar_modelo()
    joblib.dump(scaler, 'scaler_gemelo_digital.pkl')
    
    # 13. Guardar historial de diagnósticos
    if monitor.historial_informes:
        historial_diagnosticos = []
        for informe in monitor.historial_informes:
            historial_diagnosticos.append({
                'timestamp': informe['timestamp'],
                'riesgo': informe['riesgo'],
                'alerta': informe['alerta'],
                'patologias': [p['nombre'] for p in informe.get('patologias_sospechosas', [])]
            })
        
        df_diagnosticos = pd.DataFrame(historial_diagnosticos)
        df_diagnosticos.to_csv('historial_diagnosticos.csv', index=False)
        print("📁 Historial de diagnósticos guardado en 'historial_diagnosticos.csv'")
    
    # 14. Reporte final
    print("\n" + "="*60)
    print("✅ SISTEMA COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("📁 ARCHIVOS GENERADOS:")
    print("   🤖 modelo_gemelo_digital.h5 (modelo entrenado)")
    print("   📊 scaler_gemelo_digital.pkl (normalizador)")
    print("   💾 gemelo_digital.db (base de datos de pacientes)")
    print("   📄 diagnostico_final_paciente.txt (informe completo)")
    print("   📈 historial_diagnosticos.csv (historial de diagnósticos)")
    
    # Mostrar informes adicionales si se generaron
    import glob
    informes_criticos = glob.glob("informe_critico_*.txt")
    if informes_criticos:
        print(f"\n📋 INFORMES DE CASOS CRÍTICOS GENERADOS:")
        for informe in informes_criticos:
            print(f"   - {informe}")
    
    print("\n" + "="*60)
    print("💡 RECOMENDACIONES FINALES:")
    print("   • Revise los informes generados para cada paciente")
    print("   • Los casos críticos requieren atención médica inmediata")
    print("   • El historial de diagnósticos puede usarse para seguimiento")
    print("="*60)
    
    # Cerrar conexión a BD
    db.cerrar()
    
    # Mensaje de despedida
    print("\n👋 Sistema finalizado. ¡Gracias por usar Gemelo Digital Sanitario!")
# --- 6. SISTEMA DE RECOMENDACIONES DIAGNÓSTICAS ---
class SistemaRecomendaciones:
    """Genera recomendaciones personalizadas basadas en signos vitales y riesgo"""
    
    def __init__(self):
        # Umbrales para valores críticos
        self.umbrales = {
            'frec_cardiaca': {'min': 60, 'max': 100, 'critico_min': 50, 'critico_max': 120},
            'presion_art': {'min': 90, 'max': 140, 'critico_min': 70, 'critico_max': 180},
            'spo2': {'min': 95, 'max': 100, 'critico_min': 90, 'critico_max': 100},
            'temp_corp': {'min': 36.1, 'max': 37.2, 'critico_min': 35.0, 'critico_max': 39.0}
        }
        
        # Diccionario de recomendaciones por categoría
        self.recomendaciones_por_categoria = {
            'cardiovascular': {
                'leve': [
                    "📌 Controlar la presión arterial 2 veces al día",
                    "📌 Reducir el consumo de sal (menos de 5 g/día)",
                    "📌 Caminar 30 minutos diarios a ritmo moderado"
                ],
                'moderado': [
                    "⚠️ Consultar con cardiólogo en 3-5 días",
                    "⚠️ Eliminar cafeína y alcohol",
                    "⚠️ Llevar un diario de presión arterial"
                ],
                'critico': [
                    "🚨 ACUDIR URGENTEMENTE al cardiólogo hoy mismo",
                    "🚨 Si hay dolor en el pecho o dificultad respiratoria - llamar ambulancia",
                    "🚨 No realizar actividad física hasta consulta médica"
                ]
            },
            'respiratorio': {
                'leve': [
                    "📌 Realizar ejercicios de respiración 2 veces al día",
                    "📌 Ventilar la habitación cada 2 horas",
                    "📌 Humidificar el ambiente"
                ],
                'moderado': [
                    "⚠️ Consulta con neumólogo en los próximos días",
                    "⚠️ Evitar lugares con polvo o humo",
                    "⚠️ Usar pulsioxímetro para controlar SpO2"
                ],
                'critico': [
                    "🚨 HOSPITALIZACIÓN INMEDIATA! Bajo nivel de oxígeno",
                    "🚨 Oxigenoterapia según indicación médica",
                    "🚨 Llamar a emergencias: SpO2 < 90%"
                ]
            },
            'temporal': {  # Temperatura/Infección
                'leve': [
                    "📌 Beber líquidos abundantes (2-2.5 L/día)",
                    "📌 Reposo relativo 1-2 días",
                    "📌 Vitamina C y zinc para apoyar el sistema inmune"
                ],
                'moderado': [
                    "⚠️ Realizar análisis de sangre y orina",
                    "⚠️ Consulta con médico de cabecera",
                    "⚠️ Antitérmicos si temperatura > 38.5°C"
                ],
                'critico': [
                    "🚨 CONSULTA URGENTE con infectólogo",
                    "🚨 Hospitalización si temperatura > 39.5°C",
                    "🚨 Descartar sepsis o infección bacteriana"
                ]
            },
            'prevencion': {
                'recomendaciones': [
                    "✅ Parámetros normales. Continúe con estilo de vida saludable",
                    "✅ Revisiones preventivas anuales",
                    "✅ Alimentación equilibrada y actividad física",
                    "✅ Control del estrés y sueño de 7-8 horas"
                ]
            },
            'riesgo_alto': {
                'recomendaciones': [
                    "🏥 CONSULTA MÉDICA URGENTE",
                    "🏥 Evaluación clínica completa (ECG, análisis sangre)",
                    "🏥 Descartar condiciones agudas: infarto, ACV, TEP",
                    "🏥 Hospitalización en unidad especializada"
                ]
            }
        }
    
    def diagnosticar_desviaciones(self, muestra):
        """Analiza las desviaciones de los valores normales"""
        desviaciones = []
        
        for param, valor in muestra.items():
            if param in self.umbrales:
                umbral = self.umbrales[param]
                
                # Determinar gravedad
                if valor < umbral['critico_min'] or valor > umbral['critico_max']:
                    gravedad = 'critico'
                elif valor < umbral['min'] or valor > umbral['max']:
                    gravedad = 'moderado'
                else:
                    gravedad = 'normal'
                
                if gravedad != 'normal':
                    tipo = 'bajo' if valor < umbral['min'] else 'alto'
                    desviaciones.append({
                        'parametro': param,
                        'valor': valor,
                        'rango_normal': f"{umbral['min']}-{umbral['max']}",
                        'gravedad': gravedad,
                        'tipo': tipo,
                        'porcentaje_desviacion': abs((valor - umbral['min']) / umbral['min'] * 100) if valor < umbral['min'] 
                                                   else abs((valor - umbral['max']) / umbral['max'] * 100)
                    })
        
        return desviaciones
    
    def mapear_categoria(self, parametro):
        """Mapea el parámetro a una categoría de enfermedad"""
        mapa = {
            'frec_cardiaca': 'cardiovascular',
            'presion_art': 'cardiovascular',
            'spo2': 'respiratorio',
            'temp_corp': 'temporal'
        }
        return mapa.get(parametro, 'prevencion')
    
    def generar_informe_completo(self, muestra, probabilidad_riesgo):
        """Genera informe diagnóstico completo con recomendaciones"""
        
        # 1. Análisis de desviaciones
        desviaciones = self.diagnosticar_desviaciones(muestra)
        
        # 2. Determinar nivel de urgencia
        if probabilidad_riesgo > 0.8:
            nivel_urgencia = "🔴 NIVEL CRÍTICO"
            color = "🔴"
        elif probabilidad_riesgo > 0.65:
            nivel_urgencia = "🟠 NIVEL ALTO"
            color = "🟠"
        elif probabilidad_riesgo > 0.4:
            nivel_urgencia = "🟡 NIVEL MEDIO"
            color = "🟡"
        else:
            nivel_urgencia = "🟢 NIVEL BAJO"
            color = "🟢"
        
        # 3. Construir informe
        informe = []
        informe.append("\n" + "="*70)
        informe.append(f"{color} INFORME DIAGNÓSTICO DEL GEMELO DIGITAL {color}")
        informe.append("="*70)
        informe.append(f"\n📊 PROBABILIDAD DE RIESGO: {probabilidad_riesgo:.1%}")
        informe.append(f"🚨 NIVEL DE URGENCIA: {nivel_urgencia}")
        
        # 4. Desviaciones encontradas
        if desviaciones:
            informe.append("\n" + "="*70)
            informe.append("🔍 DESVIACIONES EN SIGNOS VITALES:")
            informe.append("-"*70)
            
            for d in desviaciones:
                emoji = "🔴" if d['gravedad'] == 'critico' else "🟡" if d['gravedad'] == 'moderado' else "🟢"
                
                param_nombre = {
                    'frec_cardiaca': '❤️ Frecuencia Cardíaca',
                    'presion_art': '💉 Presión Arterial',
                    'spo2': '🫁 Saturación Oxígeno (SpO2)',
                    'temp_corp': '🌡️ Temperatura Corporal'
                }.get(d['parametro'], d['parametro'])
                
                informe.append(f"\n{emoji} {param_nombre}: {d['valor']:.1f} (normal: {d['rango_normal']})")
                informe.append(f"   ➤ Desviación: {d['tipo']} en {d['porcentaje_desviacion']:.1f}%")
                informe.append(f"   ➤ Severidad: {d['gravedad'].upper()}")
        
        # 5. Recomendaciones por desviación
        informe.append("\n" + "="*70)
        informe.append("💊 RECOMENDACIONES DIAGNÓSTICAS Y TRATAMIENTO:")
        informe.append("-"*70)
        
        # Si riesgo crítico - recomendaciones urgentes
        if probabilidad_riesgo > 0.8:
            for rec in self.recomendaciones_por_categoria['riesgo_alto']['recomendaciones']:
                informe.append(f"   {rec}")
        
        # Recomendaciones por cada categoría afectada
        categorias_afectadas = set()
        for d in desviaciones:
            categoria = self.mapear_categoria(d['parametro'])
            categorias_afectadas.add((categoria, d['gravedad']))
        
        for categoria, gravedad in sorted(categorias_afectadas, key=lambda x: 
                                          {'critico': 0, 'moderado': 1, 'leve': 2}.get(x[1], 3)):
            if categoria in self.recomendaciones_por_categoria:
                rec_categoria = self.recomendaciones_por_categoria[categoria]
                
                if gravedad == 'critico' and 'critico' in rec_categoria:
                    recs = rec_categoria['critico']
                elif gravedad == 'moderado' and 'moderado' in rec_categoria:
                    recs = rec_categoria['moderado']
                elif 'leve' in rec_categoria:
                    recs = rec_categoria['leve']
                else:
                    continue
                
                informe.append(f"\n   📍 {categoria.upper()}:")
                for rec in recs:
                    informe.append(f"      {rec}")
        
        # Si no hay desviaciones - prevención
        if not desviaciones and probabilidad_riesgo < 0.4:
            for rec in self.recomendaciones_por_categoria['prevencion']['recomendaciones']:
                informe.append(f"   {rec}")
        
        # 6. Plan de acción
        informe.append("\n" + "="*70)
        informe.append("📋 PLAN DE ACCIÓN:")
        informe.append("-"*70)
        
        if probabilidad_riesgo > 0.8:
            informe.append("   1️⃣ LLAMAR INMEDIATAMENTE a emergencias (112/061)")
            informe.append("   2️⃣ Proporcionar acceso a aire fresco")
            informe.append("   3️⃣ Colocar al paciente en posición semisentada")
            informe.append("   4️⃣ Preparar documentación médica")
        elif probabilidad_riesgo > 0.65:
            informe.append("   1️⃣ Solicitar cita con médico de cabecera en 24-48 horas")
            informe.append("   2️⃣ Realizar analítica sanguínea, ECG")
            informe.append("   3️⃣ Evitar actividad física intensa durante 3 días")
            informe.append("   4️⃣ Control horario de signos vitales")
        elif probabilidad_riesgo > 0.4:
            informe.append("   1️⃣ Solicitar revisión médica programada (1-2 semanas)")
            informe.append("   2️⃣ Realizar chequeo básico")
            informe.append("   3️⃣ Modificar hábitos según recomendaciones")
        else:
            informe.append("   1️⃣ Revisión preventiva anual")
            informe.append("   2️⃣ Mantener estilo de vida saludable")
            informe.append("   3️⃣ Autocontrol periódico de signos vitales")
        
        informe.append("\n" + "="*70)
        informe.append("⚠️ ESTE INFORME NO SUSTITUYE LA CONSULTA MÉDICA")
        informe.append("="*70)
        
        return "\n".join(informe)
    
    def generar_recomendaciones_cortas(self, muestra, probabilidad_riesgo):
        """Genera recomendaciones breves para acción inmediata"""
        desviaciones = self.diagnosticar_desviaciones(muestra)
        
        rec_cortas = []
        
        # Priorizar por criticidad
        criticos = [d for d in desviaciones if d['gravedad'] == 'critico']
        moderados = [d for d in desviaciones if d['gravedad'] == 'moderado']
        
        if criticos:
            rec_cortas.append("🚨 DESVIACIONES CRÍTICAS - ¡ATENCIÓN MÉDICA URGENTE!")
            for c in criticos:
                if c['parametro'] == 'spo2' and c['valor'] < 90:
                    rec_cortas.append("   → Oxigenoterapia urgente, hospitalización")
                elif c['parametro'] == 'presion_art' and c['valor'] > 180:
                    rec_cortas.append("   → Crisis hipertensiva - medicación urgente")
                elif c['parametro'] == 'temp_corp' and c['valor'] > 39.5:
                    rec_cortas.append("   → Antitérmico + llamar emergencias")
        
        elif moderados:
            rec_cortas.append("⚠️ REQUIERE CONSULTA MÉDICA PRÓXIMAMENTE")
            for m in moderados:
                if m['parametro'] == 'presion_art':
                    rec_cortas.append("   → Medir presión mañana y tarde, llevar registro")
                elif m['parametro'] == 'temp_corp':
                    rec_cortas.append("   → Controlar temperatura cada 4 horas")
                elif m['parametro'] == 'frec_cardiaca':
                    rec_cortas.append("   → Evitar cafeína, estrés, realizar ECG")
        
        elif probabilidad_riesgo > 0.65:
            rec_cortas.append("📌 RIESGO ELEVADO - RECOMENDACIONES:")
            rec_cortas.append("   → Limitar actividad física intensa")
            rec_cortas.append("   → Control de signos cada 2 horas")
            rec_cortas.append("   → Contactar con médico de cabecera")
        
        if not rec_cortas:
            rec_cortas.append("✅ Signos vitales normales. Mantenga hábitos saludables")
        
        return rec_cortas
    
    def generar_recomendaciones_especificas_por_patologia(self, muestra):
        """Genera recomendaciones específicas según posible patología"""
        desviaciones = self.diagnosticar_desviaciones(muestra)
        recomendaciones_especificas = []
        
        # Detectar patrones específicos
        patrones = {
            'hipertension': {'presion_art': ('alto', 140)},
            'taquicardia': {'frec_cardiaca': ('alto', 100)},
            'bradicardia': {'frec_cardiaca': ('bajo', 60)},
            'hipoxia': {'spo2': ('bajo', 95)},
            'fiebre': {'temp_corp': ('alto', 37.2)},
            'hipotension': {'presion_art': ('bajo', 90)}
        }
        
        for patologia, condiciones in patrones.items():
            cumple = True
            for param, (tipo, umbral) in condiciones.items():
                valor = muestra.get(param, 0)
                if tipo == 'alto' and valor <= umbral:
                    cumple = False
                elif tipo == 'bajo' and valor >= umbral:
                    cumple = False
            if cumple:
                recomendaciones_especificas.append(self._get_recomendacion_patologia(patologia))
        
        return recomendaciones_especificas
    
    def _get_recomendacion_patologia(self, patologia):
        """Recomendaciones específicas por patología"""
        recs = {
            'hipertension': {
                'nombre': 'Hipertensión Arterial',
                'recomendaciones': [
                    "🔹 Dieta baja en sodio (menos de 2g/día)",
                    "🔹 Medicación antihipertensiva según prescripción",
                    "🔹 Evitar AINEs (ibuprofeno, naproxeno)",
                    "🔹 Monitoreo ambulatorio de presión arterial (MAPA)"
                ]
            },
            'taquicardia': {
                'nombre': 'Taquicardia Sinusal',
                'recomendaciones': [
                    "🔹 Descartar anemia, hipertiroidismo, ansiedad",
                    "🔹 Evitar estimulantes (café, té, energéticas)",
                    "🔹 ECG Holter para evaluación completa"
                ]
            },
            'hipoxia': {
                'nombre': 'Insuficiencia Respiratoria',
                'recomendaciones': [
                    "🔹 Oxigenoterapia suplementaria",
                    "🔹 Fisioterapia respiratoria",
                    "🔹 Descartar EPOC, asma, neumonía",
                    "🔹 Gasometría arterial para evaluación precisa"
                ]
            },
            'fiebre': {
                'nombre': 'Síndrome Febril',
                'recomendaciones': [
                    "🔹 Descartar foco infeccioso (orina, sangre, radiografía)",
                    "🔹 Antipiréticos según indicación",
                    "🔹 Hidratación adecuada",
                    "🔹 Cultivos si fiebre > 5 días"
                ]
            }
        }
        return recs.get(patologia, {'nombre': patologia, 'recomendaciones': ["Consultar con especialista"]})


if __name__ == "__main__":
    main()
