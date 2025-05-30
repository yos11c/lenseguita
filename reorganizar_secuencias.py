import os
import shutil
from pathlib import Path

def reorganizar_secuencias(base_dir='MP_Data_frases'):
    print("🔍 Revisando estructura de datos...")
    
    for accion in os.listdir(base_dir):
        accion_path = os.path.join(base_dir, accion)
        
        if not os.path.isdir(accion_path):
            continue  # Ignora archivos que no sean carpetas
            
        print(f"\n🔄 Procesando: {accion}")
        
        # Lista secuencias existentes (solo carpetas numéricas válidas)
        secuencias = []
        for sec_name in os.listdir(accion_path):
            sec_path = os.path.join(accion_path, sec_name)
            
            # Solo carpetas con números y que tengan 'datos.npy'
            if sec_name.isdigit() and os.path.exists(os.path.join(sec_path, "datos.npy")):
                secuencias.append((int(sec_name), sec_path))
        
        # Ordena por número de secuencia
        secuencias.sort()
        
        # Renumeración desde 0 sin saltos
        for new_number, (old_number, old_path) in enumerate(secuencias):
            new_path = os.path.join(accion_path, str(new_number))
            
            if old_path != new_path:
                print(f"  - Renumerando: {old_number} → {new_number}")
                shutil.move(old_path, new_path)
    
    print("\n✅ ¡Reorganización completada!")

if __name__ == "__main__":
    reorganizar_secuencias()