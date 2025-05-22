import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Cargar modelo y clases
modelo = load_model('modelo.h5')
clases = np.load('clases.npy')

# MediaPipe manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# Parámetros
secuencia = []
longitud_secuencia = 30
prediccion_actual = ''

# Iniciar cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer puntos clave
            puntos = []
            for lm in hand_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])

            secuencia.append(puntos)

            if len(secuencia) > longitud_secuencia:
                secuencia.pop(0)

            # Si ya tenemos la secuencia completa
            if len(secuencia) == longitud_secuencia:
                entrada = np.expand_dims(secuencia, axis=0)
                prediccion = modelo.predict(entrada, verbose=0)[0]
                clase_idx = np.argmax(prediccion)
                prediccion_actual = clases[clase_idx]

    # Mostrar predicción en pantalla
    cv2.putText(image, f"Sena detectada: {prediccion_actual}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Reconocimiento de Senas', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
