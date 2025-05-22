# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
import time as pytime
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from threading import Thread, Lock

# Inicializar Flask y cargar modelo/clases
app = Flask(__name__)
modelo = load_model('modelo.h5')
clases = np.load('clases.npy')

# Inicializar MediaPipe para hasta 2 manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# Variables para secuencia e inferencia
longitud_secuencia = 30
secuencia = []                    # ventana de 30 frames
latest_frame = None               # último frame capturado
prediccion_actual = ''
confianza_actual = 0.0
lock = Lock()                     # para proteger shared state

# Hilo separado para inferencia continua
def prediction_thread():
    global latest_frame, secuencia, prediccion_actual, confianza_actual
    while True:
        with lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is not None:
            # Procesar con MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            # Extraer puntos
            if results.multi_hand_landmarks:
                puntos = []
                for hl in results.multi_hand_landmarks:
                    for lm in hl.landmark:
                        puntos.extend([lm.x, lm.y, lm.z])
                # completar si solo 1 mano
                if len(results.multi_hand_landmarks) == 1:
                    puntos.extend([0]*63)
            else:
                puntos = [0]*126
            # Ventana deslizante
            secuencia.append(puntos)
            if len(secuencia) > longitud_secuencia:
                secuencia.pop(0)
            # Predecir cuando tengamos 30 frames
            if len(secuencia) == longitud_secuencia:
                entrada = np.expand_dims(secuencia, axis=0)
                pred = modelo.predict(entrada, verbose=0)[0]
                idx = np.argmax(pred)
                with lock:
                    prediccion_actual = clases[idx]
                    confianza_actual = float(pred[idx])
        pytime.sleep(0.01)  # pequeño descanso entre iteraciones

# Arrancar el hilo de inferencia
thread = Thread(target=prediction_thread, daemon=True)
thread.start()

# Generador de frames para el stream
def generar_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Flip y recorte (85% ancho)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame = frame[:, :int(w*0.85)]
        frame = cv2.resize(frame, (640, 480))

        # Dibujar landmarks para feedback visual
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

        # Guardar el frame para la inferencia
        with lock:
            latest_frame = frame.copy()
            texto = f"Seña detectada: {prediccion_actual}"
            cv2.putText(frame, texto, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Codificar a JPEG y enviar
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# Rutas de Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generar_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_prediction')
def get_prediction():
    with lock:
        pred = prediccion_actual
        conf = int(confianza_actual * 100)
    return jsonify({
        "prediction": pred,
        "confidence": conf
    })

# Ejecutar
if __name__ == '__main__':
    app.run(debug=True)
