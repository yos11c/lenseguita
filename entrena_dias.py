import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ===================== CONFIGURACIÓN =====================
DATA_PATH = 'MP_Data_dias'
LONGITUD_SECUENCIA = 50       # usa la misma que usaste para capturar
N_FEATURES = 63 * 2           # 2 manos
# ========================================================

acciones = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

secuencias, etiquetas = [], []
print(" Cargando datos desde MP_Data...\n")

def ajustar_secuencia(seq, target_length=LONGITUD_SECUENCIA, feature_dim=N_FEATURES):
    current_length = seq.shape[0]

    # Agrega ceros para simular la segunda mano si solo tiene una
    if seq.shape[1] == feature_dim // 2:
        pad = np.zeros((current_length, feature_dim // 2))
        seq = np.concatenate([seq, pad], axis=1)

    # Ajusta longitud
    if current_length > target_length:
        return seq[:target_length]
    elif current_length < target_length:
        padding = np.tile(seq[-1], (target_length - current_length, 1))
        return np.concatenate([seq, padding], axis=0)
    else:
        return seq

for label in acciones:
    ruta_label = os.path.join(DATA_PATH, label)
    for seq in os.listdir(ruta_label):
        archivo = os.path.join(ruta_label, seq, 'datos.npy')
        if not os.path.isfile(archivo):
            continue
        try:
            data_seq = np.load(archivo)
            if data_seq.ndim == 2 and data_seq.shape[1] in [N_FEATURES, N_FEATURES // 2]:
                valid_seq = ajustar_secuencia(data_seq)
                secuencias.append(valid_seq)
                etiquetas.append(label)
            else:
                print(f" Formato no compatible: {archivo} (shape: {data_seq.shape})")
        except Exception as e:
            print(f" Error leyendo {archivo}: {e}")

print(f"\n Total secuencias cargadas: {len(secuencias)}")
print(f"  Clases detectadas: {sorted(set(etiquetas))}\n")

le = LabelEncoder()
labels_encoded = le.fit_transform(etiquetas)
y = to_categorical(labels_encoded)
X = np.array(secuencias)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

modelo = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(LONGITUD_SECUENCIA, N_FEATURES)),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(" Entrenando modelo con TODAS las secuencias...")
modelo.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=5)]
)

print("\n Evaluación del modelo:")
y_pred = modelo.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

modelo.save('modelo_dias.h5')
np.save('clases_dias.npy', le.classes_)

print(f"\n Modelo guardado como 'modelo_dias.h5'")
print(f" Clases guardadas en 'clases_dias.npy': {le.classes_.tolist()}")