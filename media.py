from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from threading import Lock
import os
import json
from collections import deque

app = Flask(__name__)

# Inicializar mediapipe para manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Lock para evitar conflictos
camera_lock = Lock()
prediction_lock = Lock()

# Variables globales para las predicciones
latest_prediction = {'prediction': '---', 'confidence': 0}
prediction_history = deque(maxlen=10)  # Suavizar predicciones

# Cargar el modelo entrenado
model = None
scaler = None
model_path = 'modelo_senas.pkl'

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
        print("✅ Modelo cargado exitosamente")
        print(f"🔢 Clases disponibles: {model.classes_}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {str(e)}")
else:
    print(f"⚠️ Archivo de modelo no encontrado: {model_path}")

class Camera:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.cap = cv2.VideoCapture(0)
                if not cls._instance.cap.isOpened():
                    raise RuntimeError("No se pudo abrir la cámara")
                # Configurar resolución
                cls._instance.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cls._instance.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cls._instance

def update_prediction(pred, conf):
    """Actualiza las predicciones globales de manera segura"""
    with prediction_lock:
        global latest_prediction
        prediction_history.append((pred, conf))
        # Usar la moda para suavizar las predicciones
        if prediction_history:
            from collections import Counter
            preds = [p[0] for p in prediction_history]
            most_common = Counter(preds).most_common(1)[0][0]
            avg_conf = sum(p[1] for p in prediction_history if p[0] == most_common) / \
                      len([p for p in prediction_history if p[0] == most_common])
            latest_prediction = {
                'prediction': most_common,
                'confidence': round(avg_conf, 1)
            }

def gen_frames():
    camera = Camera()
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            with camera_lock:
                success, frame = camera.cap.read()
                if not success:
                    print("Error al leer el frame")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )

                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        landmarks = np.array(landmarks).reshape(1, -1)

                        if model is not None and scaler is not None:
                            try:
                                landmarks_scaled = scaler.transform(landmarks)
                                prediction = model.predict(landmarks_scaled)[0]
                                proba = model.predict_proba(landmarks_scaled)[0]
                                confidence = round(np.max(proba) * 100, 1)
                                
                                update_prediction(prediction, confidence)
                                
                                # Mostrar en el frame (opcional)
                                cv2.putText(frame, f'{prediction} ({confidence}%)', (10, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            except Exception as e:
                                print(f"Error al predecir: {str(e)}")

                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if not ret:
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with prediction_lock:
        return jsonify(latest_prediction)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)