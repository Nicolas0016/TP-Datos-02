"""
===============================================================================
TRABAJO PRÁCTICO N° 02 - Clasificación y Selección de Modelos
===============================================================================

Nombre del grupo: CRUD
Integrantes:
    - Argañaraz, Nicolás (652/25)
    - Frontera, Axel (753/25)
    - Rojas Emiliano (nose)

Fecha: 24 feb 2026
Materia: Laboratorio de Datos
Carrera: Ciencias de Datos y Ciencias de la Computación.

Descripción:
    Este script implementa la resolución del TP-02 sobre clasificación y 
    selección de modelos utilizando validación cruzada. Se trabaja con el 
    dataset de imágenes de caracteres tipeados (letras mayúsculas del inglés).

Contenido:
    1. Análisis exploratorio de datos
    2. Clasificación binaria (letras O vs L)
    3. Clasificación multiclase (todas las letras)

Estructura del código:
    El código está organizado en secciones delimitadas por #%%% para permitir
    su ejecución por fragmentos en editores como Spyder o VSCode.
===============================================================================
"""


#%% ============================================================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# ===============================================================================
"""
Módulos necesarios para el procesamiento de datos, visualización y modelado.
Se importan todas las bibliotecas requeridas y se configura el entorno de
visualización.
"""
import pandas as pd
import numpy as np

#%% ============================================================================
# CARGA DE DATOS
# ===============================================================================
"""
Carga del dataset desde el archivo CSV.
Se realiza una verificación inicial de la estructura de los datos.
"""
def cargar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    
    Parameters
    ----------
    ruta_archivo : str
        Ruta al archivo CSV con los datos
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(ruta_archivo)
        print(f"✅ Datos cargados exitosamente desde: {ruta_archivo}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {ruta_archivo}")
    
DATA_PATH = '../data/letras.csv'
df = cargar_datos(DATA_PATH)


#%% ============================================================================
# FUNCION PARA VISUALIZAR LETRAS (basado en el estilo del PDF)
# ===============================================================================

def mostrar_imagen_letra(df, indice, columna_label='label', dimension=(28, 28)):
    """
    Muestra una imagen de letra a partir de su índice en el DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    indice : int
        Índice de la fila a visualizar
    columna_label : str
        Nombre de la columna que contiene la etiqueta
    dimension : tuple
        Dimensiones de la imagen (alto, ancho)
    """
    # Obtener la letra
    letra = df.iloc[indice][columna_label]
    
    # Obtener los píxeles (excluyendo la columna de etiqueta)
    X = df.drop(columns=[columna_label])
    
    # Reconstruir y mostrar la imagen
    img = np.array(X.iloc[indice]).reshape(dimension[0], dimension[1])
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f'Letra: {letra} - Índice: {indice}', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    return img