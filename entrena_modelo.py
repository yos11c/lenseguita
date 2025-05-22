import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle

print("=== Entrenamiento del Modelo de Lenguaje de Señas ===")

try:
    # Cargar los datos
    df = pd.read_csv('datos_senas.csv')
    
    # Verificar datos
    if len(df) < 10:
        raise ValueError("❌ Insuficientes datos. Necesitas al menos 10 muestras.")
    
    print(f"\n📊 Datos cargados correctamente:")
    print(f"- Total muestras: {len(df)}")
    print(f"- Señas disponibles: {df['label'].unique()}")
    print(f"- Distribución:\n{df['label'].value_counts()}")

    # Preparar datos
    X = df.drop('label', axis=1)
    y = df['label']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Entrenar modelo
    print("\n🔄 Entrenando modelo...")
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    )
    model.fit(X_train, y_train)

    # Evaluación
    print("\n📝 Resultados de evaluación:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    model_data = {
        'model': model,
        'scaler': model.named_steps['standardscaler'],
        'classes': list(model.classes_)
    }

    with open('modelo_senas.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\n✅ Modelo guardado exitosamente como 'modelo_senas.pkl'")
    print(f"🔢 Clases aprendidas: {model.classes_}")

except FileNotFoundError:
    print("❌ Error: No se encontró 'datos_senas.csv'")
    print("💡 Ejecuta primero 'python captura_datos.py'")
except Exception as e:
    print(f"❌ Error inesperado: {str(e)}")