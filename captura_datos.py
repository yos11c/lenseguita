import cv2
import mediapipe as mp
import pandas as pd
import os
import datetime

# Configuración inicial
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Crear directorio para dataset
os.makedirs('dataset', exist_ok=True)

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Estructuras para datos
data = []
labels = []
letras = ['A', 'B', 'C']  # Señas a capturar

print("Instrucciones:")
print("1. Coloca tu mano frente a la cámara")
print("2. Presiona la tecla correspondiente (A, B, C) para capturar")
print("3. Presiona ESC para terminar")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error al leer la cámara")
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Dibujar landmarks si se detecta mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar instrucciones en pantalla
    cv2.putText(frame, "Presiona A, B o C para capturar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "ESC para terminar", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Captura de Datos - Lenguaje de Señas", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key in [ord('a'), ord('b'), ord('c')]:
        if results.multi_hand_landmarks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            letra = chr(key).upper()

            # Guardar imagen
            img_name = f"dataset/{letra}_{timestamp}.jpg"
            cv2.imwrite(img_name, frame)

            # Guardar landmarks
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            data.append(landmarks)
            labels.append(letra)
            print(f"Capturada seña: {letra} - Total muestras: {len(data)}")
        else:
            print("No se detectó mano. Intenta de nuevo.")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()

# Guardar datos en CSV
if data:
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('datos_senas.csv', index=False)
    print(f"\nDatos guardados en 'datos_senas.csv' ({len(df)} muestras)")
else:
    print("\nNo se capturaron datos")