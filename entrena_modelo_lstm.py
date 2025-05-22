import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Ruta base
DATA_PATH = os.path.join('MP_Data')
acciones = np.array(os.listdir(DATA_PATH))

secuencias, etiquetas = [], []

print("📦 Cargando datos desde MP_Data...\n")
for label in acciones:
    for secuencia in os.listdir(os.path.join(DATA_PATH, label)):
        archivo = os.path.join(DATA_PATH, label, secuencia, "datos.npy")
        if os.path.isfile(archivo):
            secuencia_datos = np.load(archivo)

            # ⚠️ VALIDAR SHAPE EXACTO (30, 63)
            if secuencia_datos.shape != (30, 63):
                print(f"❌ Descarta: {archivo} con shape {secuencia_datos.shape}")
                continue

            secuencias.append(secuencia_datos)
            etiquetas.append(label)

print(f"\n📊 Total secuencias válidas: {len(secuencias)}")
print(f"🏷️  Clases válidas: {np.unique(etiquetas).tolist()}")

# Codificar etiquetas
le = LabelEncoder()
etiquetas_codificadas = le.fit_transform(etiquetas)
etiquetas_onehot = to_categorical(etiquetas_codificadas)

# Preparar datos
X = np.array(secuencias)
y = etiquetas_onehot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=etiquetas_codificadas, random_state=42)

# Crear modelo LSTM
modelo = Sequential()
modelo.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
modelo.add(LSTM(64, return_sequences=False, activation='relu'))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dense(len(le.classes_), activation='softmax'))

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar modelo
print("\n🔁 Entrenando modelo LSTM...")
modelo.fit(X_train, y_train, epochs=30, callbacks=[EarlyStopping(patience=5)], validation_data=(X_test, y_test))

# Evaluar modelo
print("\n📈 Evaluación del modelo:")
y_pred = modelo.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# Guardar modelo y etiquetas
modelo.save('modelo.h5')
np.save('clases.npy', le.classes_)

print("\n✅ Modelo guardado como 'modelo.h5'")
print(f"📦 Clases guardadas en 'clases.npy': {le.classes_.tolist()}")
