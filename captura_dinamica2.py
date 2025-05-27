import cv2
import numpy as np
import os
import time
import mediapipe as mp

print("📸 Iniciando captura dinámica...")

# Configurar MediaPipe para detectar hasta 2 manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# Lista de señas dinámicas a capturar

acciones = np.array(['Domingo'])       # <---------- Escribe la letra/palabra a entrenar
base_dir = 'MP_Data_dias'  # <---------- Escribe la categoria 

# Configuraciones de captura
o_secuencias = 30
longitud_secuencia = 30

# Crear las carpetas base y subdirectorios para cada acción
for accion in acciones:
    base_path = os.path.join(base_dir, accion)
    os.makedirs(base_path, exist_ok=True)
    existentes = len([d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))])
    for seq in range(existentes, existentes + o_secuencias):
        os.makedirs(os.path.join(base_path, str(seq)), exist_ok=True)

# Iniciar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

# Bucle principal de captura
for accion in acciones:
    print(f"\n🔥 Prepárate para grabar la seña: {accion.upper()}")
    time.sleep(2)
    base_path = os.path.join(base_dir, accion)
    existentes = len([d for d in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, d))])
    inicio = existentes - o_secuencias

    for secuencia in range(inicio, inicio + o_secuencias):
        print(f"🎬 Grabando secuencia {secuencia+1}/{inicio + o_secuencias} para: {accion}")
        data_secuencia = []

        for frame_num in range(longitud_secuencia):
            ret, frame = cap.read()
            if not ret:
                print("❌ Error leyendo frame de la cámara")
                break

            # Voltear y convertir para MediaPipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Dibujar esqueletos y extraer coordenadas
            if results.multi_hand_landmarks:
                puntos = []
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        drawing.DrawingSpec(color=(0,0,255), thickness=2)
                    )
                    for lm in hand_landmarks.landmark:
                        puntos.extend([lm.x, lm.y, lm.z])
                # Si sólo hay una mano, rellena para la segunda
                if len(results.multi_hand_landmarks) == 1:
                    puntos.extend([0]*63)
                data_secuencia.append(puntos)
            else:
                # Ninguna mano detectada
                data_secuencia.append([0]*126)

            # Mostrar texto e imagen
            cv2.putText(frame, f'Sena: {accion}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f'Secuencia: {secuencia+1}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Grabando sena dinámica', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Guardar datos de la secuencia
        np.save(
            os.path.join(base_path, str(secuencia), 'datos'),
            np.array(data_secuencia)
        )
        print(f"✅ Secuencia {secuencia} guardada para {accion}")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()