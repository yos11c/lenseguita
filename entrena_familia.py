import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Parámetros de datos
DATA_PATH = 'MP_Data_familia'
LONGITUD_SECUENCIA = 30
# Número total de características por frame: 63 coords × 2 manos
N_FEATURES = 63 * 2

# Obtener las subcarpetas (acciones) de MP_Data
acciones = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

secuencias, etiquetas = [], []
print("📦 Cargando datos desde MP_Data...\n")

for label in acciones:
    ruta_label = os.path.join(DATA_PATH, label)
    for seq in os.listdir(ruta_label):
        archivo = os.path.join(ruta_label, seq, 'datos.npy')
        if not os.path.isfile(archivo):
            continue
        data_seq = np.load(archivo)
        # Aceptar secuencias con 1 mano (63 features) o 2 manos (126 features)
        if data_seq.shape == (LONGITUD_SECUENCIA, N_FEATURES):
            valid_seq = data_seq
        elif data_seq.shape == (LONGITUD_SECUENCIA, N_FEATURES // 2):
            # pad zeros para la segunda mano
            pad = np.zeros((LONGITUD_SECUENCIA, N_FEATURES // 2))
            valid_seq = np.concatenate([data_seq, pad], axis=1)
        else:
            print(f"❌ Descarta: {archivo} con shape {data_seq.shape}")
            continue
        secuencias.append(valid_seq)
        etiquetas.append(label)

print(f"\n📊 Total secuencias válidas: {len(secuencias)}")
print(f"🏷️  Clases detectadas: {sorted(set(etiquetas))}\n")

# Codificar etiquetas a one-hot
le = LabelEncoder()
labels_encoded = le.fit_transform(etiquetas)
y = to_categorical(labels_encoded)

# Convertir a arrays numpy
X = np.array(secuencias)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

# Definir el modelo LSTM
modelo = Sequential([
    LSTM(64, return_sequences=True, activation='relu',
         input_shape=(LONGITUD_SECUENCIA, N_FEATURES)),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
print("🔁 Entrenando modelo LSTM...")
modelo.fit(
    X_train, y_train,
    epochs=30,
    callbacks=[EarlyStopping(patience=5)],
    validation_data=(X_test, y_test)
)

# Evaluación
print("\n📈 Evaluación del modelo:")
y_pred = modelo.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(
    classification_report(
        y_true_labels,
        y_pred_labels,
        target_names=le.classes_
    )
)

# Guardar modelo y clases
modelo.save('modelo_familia.h5')
np.save('clases_familia.npy', le.classes_)

print("\n✅ Modelo guardado como 'modelo.h5'")
print(f"📦 Clases guardadas en 'clases.npy': {le.classes_.tolist()}")
