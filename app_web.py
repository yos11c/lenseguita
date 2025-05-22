# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model

# Inicializar Flask
app = Flask(__name__)

# Cargar modelo entrenado y clases
modelo = load_model('modelo.h5')
clases = np.load('clases.npy')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils

# Variables globales para predicciones
secuencia = []
longitud_secuencia = 30
prediccion_actual = ''
confianza_actual = 0.0

# Función para capturar frames y predecir
def generar_frames():
    global secuencia, prediccion_actual, confianza_actual
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Recortar el borde derecho (ajusta el 0.85 si es necesario)
        height, width, _ = frame.shape
        frame = frame[:, :int(width * 0.85)]
        frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(rgb)

        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                puntos = []
                for lm in hand_landmarks.landmark:
                    puntos.extend([lm.x, lm.y, lm.z])

                secuencia.append(puntos)

                if len(secuencia) > longitud_secuencia:
                    secuencia.pop(0)

                if len(secuencia) == longitud_secuencia:
                    entrada = np.expand_dims(secuencia, axis=0)
                    pred = modelo.predict(entrada, verbose=0)[0]
                    idx = np.argmax(pred)
                    prediccion_actual = clases[idx]
                    confianza_actual = float(pred[idx])
        else:
            prediccion_actual = ''
            confianza_actual = 0.0

        # Mostrar predicción sobre el frame
        texto = f"Sena detectada: {prediccion_actual}"
        cv2.putText(frame, texto, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta del stream de video
@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta de predicción actual
@app.route('/get_prediction')
def get_prediction():
    global prediccion_actual, confianza_actual
    return jsonify({
        "prediction": prediccion_actual,
        "confidence": int(confianza_actual * 100)
    })

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
