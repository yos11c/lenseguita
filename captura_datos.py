import cv2
import mediapipe as mp
import pandas as pd
import os
import datetime
import numpy as np

# Configuración inicial
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Cambiado a 2 manos
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
letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

print("Instrucciones:")
print("1. Coloca tus manos frente a la cámara")
print("2. Presiona la tecla correspondiente (A-Z) para capturar")
print("3. Presiona ESPACIO para capturar señas que requieran dos manos")
print("4. Presiona ESC para terminar")

# Configuración de ventana
cv2.namedWindow("Captura de Datos - Lenguaje de Señas", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Captura de Datos - Lenguaje de Señas", 800, 600)

def process_landmarks(hand_landmarks):
    """Procesa los landmarks de una mano y devuelve características normalizadas"""
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = landmarks_array[0]
    normalized_landmarks = landmarks_array - wrist
    return normalized_landmarks.flatten()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error al leer la cámara")
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Dibujar landmarks si se detectan manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Mostrar número de manos detectadas
        cv2.putText(frame, f"Manos detectadas: {len(results.multi_hand_landmarks)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar instrucciones en pantalla
    cv2.putText(frame, "Presiona una letra (A-Z) para capturar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "ESPACIO para señas con dos manos", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "ESC para terminar", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Muestras capturadas: {len(data)}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Captura de Datos - Lenguaje de Señas", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # ESPACIO - para señas con dos manos
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            letra = "2H"  # Prefijo para señas con dos manos

            # Guardar imagen
            img_name = f"dataset/{letra}_{timestamp}.jpg"
            cv2.imwrite(img_name, frame)

            # Procesar ambas manos
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks.extend(process_landmarks(hand_landmarks).tolist())
            
            data.append(landmarks)
            labels.append(letra)
            print(f"Capturada seña con dos manos - Total muestras: {len(data)}")
        else:
            print("Se requieren exactamente dos manos para esta captura")
    elif (key >= ord('a') and key <= ord('z')) or (key >= ord('A') and key <= ord('Z')):
        if results.multi_hand_landmarks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            letra = chr(key).upper()

            # Guardar imagen
            img_name = f"dataset/{letra}_{timestamp}.jpg"
            cv2.imwrite(img_name, frame)

            # Procesar landmarks (solo primera mano)
            landmarks = process_landmarks(results.multi_hand_landmarks[0]).tolist()
            
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
    # Verificar que todos los vectores de características tengan la misma longitud
    feature_length = len(data[0])
    for i, sample in enumerate(data):
        if len(sample) != feature_length:
            print(f"Error: muestra {i} tiene longitud {len(sample)}, esperada {feature_length}")
            # Rellenar con ceros si es necesario
            data[i] = sample + [0] * (feature_length - len(sample))
    
    df = pd.DataFrame(data)
    df['label'] = labels
    
    # Generar nombres de columnas
    num_landmarks = 21  # 21 landmarks por mano
    num_coords = 3      # x, y, z
    single_hand_features = num_landmarks * num_coords
    
    # Nombres de columnas para landmarks
    column_names = []
    for hand in ['mano1', 'mano2']:  # Soporte para dos manos
        for i in range(num_landmarks):
            for coord in ['x', 'y', 'z']:
                column_names.append(f"{hand}_landmark{i}_{coord}")
    
    # Solo usar las columnas necesarias según la longitud de los datos
    df.columns = column_names[:len(data[0])] + ['label']
    
    # Guardar con comprobación de existencia
    csv_path = 'datos_senas.csv'
    if os.path.exists(csv_path):
        # Si el archivo existe, añadir los nuevos datos
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(csv_path, index=False)
        print(f"\nDatos actualizados en '{csv_path}' (Total muestras: {len(updated_df)})")
    else:
        df.to_csv(csv_path, index=False)
        print(f"\nDatos guardados en '{csv_path}' ({len(df)} muestras)")
else:
    print("\nNo se capturaron datos")