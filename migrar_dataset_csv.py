import os
import pandas as pd
import numpy as np

# Ruta del CSV original
csv_path = 'datos_senas.csv'

# Carpeta de destino donde se unificarán todos los datos
destino = 'MP_Data'

# Leer el CSV
df = pd.read_csv(csv_path)

# Contador por letra
conteos = {}

# Crear carpetas y guardar cada fila como .npy
for i, row in df.iterrows():
    label = row['label'].upper()
    datos = row.drop('label').values.astype(np.float32)

    if label not in conteos:
        conteos[label] = 0
    secuencia = conteos[label]

    carpeta = os.path.join(destino, label, str(secuencia))
    os.makedirs(carpeta, exist_ok=True)

    np.save(os.path.join(carpeta, 'datos.npy'), datos)
    conteos[label] += 1

print("✅ Migración completada. Se guardaron:")
for label, cantidad in conteos.items():
    print(f"- {label}: {cantidad} muestras")
