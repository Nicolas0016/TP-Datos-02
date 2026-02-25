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
import matplotlib.pyplot as plt
#%% FUNCIONES UTILIZADAS DURANTE EL INFORME
def conteo_de_letters(df):
    conteo = {}
    for _, row in df.iterrows():
        if row['label'] not in conteo:
            conteo[row['label']] = 0
        conteo[row['label']] +=1
    return conteo
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
        print("Datos cargados")
        return df
    except FileNotFoundError:
        print("No se encontró el archivo")

def remane_labels(df):
    letters_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    index_label = list(range(len(letters_upper)))
    
    rename = dict(zip(index_label,letters_upper))
    
    df['label'] = df['label'].replace(rename)
    return df
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

#%% CARGA DE DATOS

DATA_PATH = './data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)

#%% Análisis exploratorio

# =============================================================================
#### TOTAL DE MUESTRAS
# =============================================================================
print(f"Total de muestras: {len(df)}")


# =============================================================================
#### TOTAL DE ATRIBUTOS
# =============================================================================
print(f"Total de atributos: 1 etiqueta + {len(df.iloc[0,:]) - 1} pixeles")


# =============================================================================
#### DIMENSIÓN DE LAS imágenes
# =============================================================================
print("Dimeision de las imagenes: 28 x 28 = 784 pixeles (Por enunciado son imágenes cuadradas)")

# =============================================================================
#### CANTIDAD DE VALORES NULOS
# =============================================================================
print(f"Cantidad de elementos nulos: {df.isnull().sum().sum()}")

# =============================================================================
#### Cantidad de clases de interes de la variable de interés (la letra representada) 
# =============================================================================

# =============================================================================
#### Distribución de letras
# =============================================================================
conteo_letras = conteo_de_letters(df)

total_de_muestras = len(df)
cantidad_letras = len(conteo_letras)

# En general
print(f"Cantidad de muestras {total_de_muestras}")
print(f"Cantidad de letras diferentes {cantidad_letras}")
print(f"Letras presentes: {list(conteo_letras.keys())}")

# Vemo el balance 
max_letra, max_cantidad = max(conteo_letras.items())
min_letra, min_cantidad = min(conteo_letras.items())
ratio = max_cantidad / min_cantidad

print("Análisis de balance:")
print(f"Letra con más muestras: {max_letra} ({max_cantidad} muestras)")
print(f"Letra con menos muestras: {min_letra} ({min_cantidad} muestras)")
print(f"Diferencia entre max/min: {max_cantidad - min_cantidad}")
print(f"Ratio max/min: {ratio}")

# =============================================================================
#### Visualización
# =============================================================================

plt.figure(figsize=(15,6))
letras = conteo_letras.keys()
cantidades = [conteo_letras[letra] for letra in letras]

barras = plt.bar(letras, cantidades)
plt.title("Distribución de letras en el df")
plt.xlabel("letra")
plt.ylabel("cantidad de muestras")
plt.figtext(0.5, 0.01, "FIGURA 1: Distribución de letras en el dataset", 
            ha="center", fontsize=10, style='italic')
plt.show()

