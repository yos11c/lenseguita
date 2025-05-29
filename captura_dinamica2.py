import cv2
import numpy as np
import os
import time
import mediapipe as mp

print("📸 Iniciando captura dinámica con temporizador preciso...")

# ==============================================
# CONFIGURACIONES EDITABLES
# ==============================================
TIEMPO_CAPTURA = 4.0  # Duración de captura
# ==============================================

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# Configuración de señas
acciones = np.array(['Miercoles'])  # Editar con tus señas
base_dir = 'MP_Data_dias'

# Configuraciones fijas
SECUENCIAS_POR_ACCION = 300
LONGITUD_SECUENCIA = 50  # 30 frames fijos

def asegurar_formato_datos(puntos):
    """Garantiza siempre 126 valores (63 x 2 manos)"""
    return (puntos + [0]*126)[:126]

def capturar_secuencia(cap, accion, secuencia_path, num_secuencia, total_secuencias):
    """Captura con temporizador preciso que muestra tiempo real"""
    data_secuencia = []
    start_time = time.time()
    last_display_time = start_time
    
    # Variable para controlar el tiempo real de captura
    tiempo_inicio_captura = time.time()
    
    while (time.time() - start_time) < TIEMPO_CAPTURA:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Procesamiento del frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Extracción de landmarks
        puntos = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    puntos.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                puntos.extend([0]*63)
        else:
            puntos = [0]*126
            
        puntos = asegurar_formato_datos(puntos)
        
        # Solo añadir frame si aún no hemos alcanzado el límite
        if len(data_secuencia) < LONGITUD_SECUENCIA:
            data_secuencia.append(puntos)
            tiempo_inicio_captura = time.time()  # Resetear el temporizador de captura
        
        # Calcular tiempo transcurrido y restante
        current_time = time.time()
        tiempo_transcurrido = current_time - start_time
        tiempo_restante = max(0, TIEMPO_CAPTURA - tiempo_transcurrido)
        
        # Mostrar información (actualizada cada 100ms para mejor fluidez)
        if current_time - last_display_time >= 0.1:
            cv2.putText(frame, f"Seña: {accion}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Captura: {num_secuencia}/{total_secuencias}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tiempo: {tiempo_restante:.1f}s", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frames: {len(data_secuencia)}/{LONGITUD_SECUENCIA}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Captura Dinámica', frame)
            last_display_time = current_time
        
        if cv2.waitKey(1) & 0xFF == 27:  # Permitir cancelar con ESC
            return None
    
    # Asegurar longitud exacta
    if len(data_secuencia) > LONGITUD_SECUENCIA:
        data_secuencia = data_secuencia[:LONGITUD_SECUENCIA]
    elif len(data_secuencia) < LONGITUD_SECUENCIA:
        # Repetir el último frame capturado si no alcanzamos el total
        if len(data_secuencia) > 0:
            data_secuencia.extend([data_secuencia[-1]] * (LONGITUD_SECUENCIA - len(data_secuencia)))
        else:
            # Caso extremo: no se capturó ningún frame
            data_secuencia = [[0]*126] * LONGITUD_SECUENCIA
    
    # Guardar datos
    np.save(os.path.join(secuencia_path, 'datos'), np.array(data_secuencia, dtype=np.float32))
    return len(data_secuencia)

def main():
    # Crear directorios
    for accion in acciones:
        base_path = os.path.join(base_dir, accion)
        os.makedirs(base_path, exist_ok=True)
        secuencias_existentes = len(os.listdir(base_path))
        total_secuencias = secuencias_existentes + SECUENCIAS_POR_ACCION
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara para {accion}")
            continue
            
        try:
            for seq in range(secuencias_existentes, total_secuencias):
                secuencia_path = os.path.join(base_path, str(seq))
                os.makedirs(secuencia_path, exist_ok=True)
                
                print(f"\n▶ Iniciando '{accion}' ({TIEMPO_CAPTURA}s) - {seq+1}/{total_secuencias}")
                frames_capturados = capturar_secuencia(cap, accion, secuencia_path, seq+1, total_secuencias)
                
                if frames_capturados is None:
                    print("⚠️ Captura cancelada por el usuario")
                    break
                    
                print(f"✅ Captura completada ({frames_capturados} frames)")
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    print("✨ Proceso de captura finalizado correctamente")