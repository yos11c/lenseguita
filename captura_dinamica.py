import cv2
import numpy as np
import os
import time
import mediapipe as mp

print("📸 Iniciando captura dinámica...")

# Configurar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing = mp.solutions.drawing_utils

# Lista de señas dinámicas a capturar
acciones = np.array(['A'])  # usa minúsculas si así las guardaste

# Configuraciones de captura
no_secuencias = 30
longitud_secuencia = 30

# Crear carpetas si no existen y calcular desde qué número continuar
for accion in acciones:
    base_path = os.path.join('MP_Data', accion)
    os.makedirs(base_path, exist_ok=True)
    inicio = len(os.listdir(base_path))  # cuenta cuántas secuencias ya hay

    for secuencia in range(inicio, inicio + no_secuencias):
        os.makedirs(os.path.join(base_path, str(secuencia)), exist_ok=True)

# Iniciar cámara
cap = cv2.VideoCapture(0)

for accion in acciones:
    print(f'\n🔥 Prepárate para grabar la seña: {accion.upper()}')
    time.sleep(2)

    base_path = os.path.join('MP_Data', accion)
    inicio = len(os.listdir(base_path)) - no_secuencias

    for secuencia in range(inicio, inicio + no_secuencias):
        print(f'🎬 Grabando secuencia {secuencia + 1}/{inicio + no_secuencias} para: {accion}')
        data_secuencia = []

        for frame_num in range(longitud_secuencia):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    puntos = []
                    for lm in hand_landmarks.landmark:
                        puntos.extend([lm.x, lm.y, lm.z])
                    data_secuencia.append(puntos)
            else:
                data_secuencia.append([0]*63)

            cv2.putText(frame, f'Seña: {accion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Secuencia: {secuencia + 1}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Grabando seña dinámica', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        np.save(os.path.join('MP_Data', accion, str(secuencia), 'datos'), np.array(data_secuencia))
        print(f'✅ Secuencia {secuencia} guardada para {accion}')


cap.release()
cv2.destroyAllWindows()
