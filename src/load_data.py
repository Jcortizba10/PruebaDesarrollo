import pandas as pd
import os

# Función para cargar los datos
def load_data(file_path):
    """
    Carga los datos del archivo CSV y maneja posibles errores.
    """
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Datos cargados correctamente.")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e.__class__.__name__} - {e}")
        return None
