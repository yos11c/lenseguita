import numpy as np
import os

def verificar_capturas(base_dir='MP_Data_frases'):
    print("🔍 Buscando capturas problemáticas...")
    problemas = []
    
    for accion in os.listdir(base_dir):
        accion_path = os.path.join(base_dir, accion)
        
        if not os.path.isdir(accion_path):
            continue
            
        for sec in os.listdir(accion_path):
            sec_path = os.path.join(accion_path, sec)
            datos_path = os.path.join(sec_path, "datos.npy")
            
            if not os.path.exists(datos_path):
                problemas.append(f"❌ {accion}/{sec}: Falta 'datos.npy'")
                continue
                
            try:
                datos = np.load(datos_path)
                if datos.shape[0] != 30:  # Ajusta este número al esperado
                    problemas.append(f"⚠️ {accion}/{sec}: Shape incorrecto {datos.shape}")
            except:
                problemas.append(f"❌ {accion}/{sec}: Archivo corrupto")
    
    if problemas:
        print("\n".join(problemas))
        print(f"\n🚨 Se encontraron {len(problemas)} problemas.")
    else:
        print("✅ ¡Todo correcto! Todas las capturas son válidas.")

verificar_capturas()