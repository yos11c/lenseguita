import cv2
import numpy as np
import mediapipe as mp
import time
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
from threading import Thread, Lock

app = Flask(__name__)

# 1) Carga de modelos independientes
modelos = {
    'abecedario': load_model('modelo_abecedario.h5'),
    'familia':    load_model('modelo_familia.h5'),
    'frases':     load_model('modelo_frases.h5'),
    'dias':       load_model('modelo_dias.h5'),
}
clases = {
    cat: np.load(f'clases_{cat}.npy')
    for cat in modelos
}

# 2) MediaPipe (hasta 2 manos)
mp_hands = mp.solutions.hands
hands_cfg = dict(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# Instanciar un solo objeto Hands para todo el proceso
hands = mp_hands.Hands(**hands_cfg)
# Utilidades de dibujo
drawing = mp.solutions.drawing_utils

# 3) Estado compartido e hilos
lock = Lock()
latest_frame = {cat: None for cat in modelos}
prediction = {cat: {'text': '', 'conf': 0.0} for cat in modelos}
SEQUENCE_LEN = 30

# Hilo de inferencia por categoría
def prediction_thread(cat):
    seq = []
    modelo = modelos[cat]
    # Reutilizar la misma instancia de MediaPipe Hands
    while True:
        with lock:
            frame = latest_frame[cat].copy() if latest_frame[cat] is not None else None
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            # Extraer coordenadas
            if results.multi_hand_landmarks:
                pts = []
                for hl in results.multi_hand_landmarks:
                    for lm in hl.landmark:
                        pts.extend([lm.x, lm.y, lm.z])
                if len(results.multi_hand_landmarks) == 1:
                    pts.extend([0] * 63)
            else:
                pts = [0] * 126
            # Ventana deslizante
            seq.append(pts)
            if len(seq) > SEQUENCE_LEN:
                seq.pop(0)
            # Predecir
            if len(seq) == SEQUENCE_LEN:
                entrada = np.expand_dims(seq, axis=0)
                pred = modelo.predict(entrada, verbose=0)[0]
                idx = np.argmax(pred)
                with lock:
                    prediction[cat]['text'] = clases[cat][idx]
                    prediction[cat]['conf'] = float(pred[idx])
        time.sleep(0.01)

# Arrancar hilos para cada categoría
for c in modelos:
    Thread(target=prediction_thread, args=(c,), daemon=True).start()

# Generador de frames por categoría
def generar_frames(cat):
    cap = cv2.VideoCapture(0)
    # Reutilizar la misma instancia de MediaPipe Hands
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        # Dibujar landmarks como feedback
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
        # Guardar frame para inferencia
        with lock:
            latest_frame[cat] = frame.copy()
            text = prediction[cat]['text']
            conf = int(prediction[cat]['conf'] * 100)
        # Mostrar texto en frame
        cv2.putText(frame, f"Seña: {text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # Enviar frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# 5) Rutas Flask dinámicas
from flask import abort

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<cat>')
def serve_cat(cat):
    # Página de categoría
    if cat not in modelos:
        abort(404)
    return render_template(f'{cat}.html')

@app.route('/video_feed/<cat>')
def video_feed(cat):
    # Stream de vídeo por categoría
    if cat not in modelos:
        abort(404)
    return Response(
        generar_frames(cat),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_prediction/<cat>')
def get_pred(cat):
    # Predicción SSE por categoría
    if cat not in modelos:
        abort(404)
    with lock:
        p = prediction[cat]
    return jsonify({'prediction': p['text'], 'confidence': int(p['conf']*100)})

if __name__ == '__main__':
    app.run(debug=True)

