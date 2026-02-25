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
        print("Datos cargados")
        return df
    except FileNotFoundError:
        print("No se encontró el archivo")
DATA_PATH = '../data/letras.csv'
df = cargar_datos(DATA_PATH)
def remane_labels(df):
    letters_upper = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    index_label = list(range(len(letters_upper)))
    
    rename = dict(zip(index_label,letters_upper))
    
    df['label'] = df['label'].replace(rename)
    return df
df =  remane_labels(df)

#%% TOTAL DE MUESTRAS
print(f"Total de muestras: {len(df)}")

#%% Total de Atributos
print(f"Total de atributos: 1 etiqueta + {len(df.iloc[0,:]) - 1} pixeles")

#%% Dimiensión de las imágenes
print("Dimeision de las imagenes: 28 x 28 = 784 pixeles (Por enunciado son imágenes cuadradas)")

#%% Valores nulos
print(df.isnull().sum().sum())
#%% CANTIDAD DE CLASES DE INTERES
# =============================================================================
# Cantidad de clases de interes de la variable de interés (la letra representada) 
# =============================================================================
def conteo_de_letters(df):
    conteo = {}
    for _, row in df.iterrows():
        if row['label'] not in conteo:
            conteo[row['label']] = 0
        conteo[row['label']] +=1
    return conteo
#%% Distribución de letras
conteo_letras = conteo_de_letters(df)

total_de_muestras = len(df)
cantidad_letras = len(conteo_letras)

# En general
print(f"Cantidad de muestras {total_de_muestras}")
print(f"Cantidad de letras diferentes {cantidad_letras}")
print(f"Letras presentes: {conteo_letras.keys()}")

# Vemo el balance 
max_letra, max_cantidad = max(conteo_letras.items())
min_letra, min_cantidad = min(conteo_letras.items())
ratio = max_cantidad / min_cantidad

print("Análisis de balance:")
print(f"Letra con más muestras: {max_letra} ({max_cantidad} muestras)")
print(f"Letra con menos muestras: {min_letra} ({min_cantidad} muestras)")
print(f"Diferencia entre max/min: {max_cantidad - min_cantidad}")
print(f"Ratio max/min: {ratio}")

#%% Visualización
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


# %% PUNTO B

def comparar_letras_superposicion(df, letra1, letra2,dimension=(28, 28)):#le saqué el contador de figuras
    """
    Compara dos letras superponiéndolas.
    - Fondo blanco (255): píxeles iguales
    - Rojo: píxeles que pertenecen a letra1
    - Azul: píxeles que pertenecen a letra2
    """
    idx1 = df[df['label'] == letra1].index[0]
    idx2 = df[df['label'] == letra2].index[0]
    
    X = df.drop(columns=['label'])
    
    img1 = np.array(X.iloc[idx1]).reshape(dimension[0], dimension[1])
    img2 = np.array(X.iloc[idx2]).reshape(dimension[0], dimension[1])
    
    img_superpuesta = np.ones((dimension[0], dimension[1], 3))  # Fondo blanco (1,1,1)
    
    mask_ambas = (img1 < 128) & (img2 < 128)
    img_superpuesta[mask_ambas] = [0, 1, 0]  # Verde
    
    # Visualizar
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(f'Letra {letra1}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(f'Letra {letra2}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(img_superpuesta)
    axes[2].set_title('Superposición', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'Comparación {letra1} vs {letra2} - Superposición', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.figtext(0.5, 0.01, f"FIGURA 1.b: Comparación {letra1} vs {letra2}", 
                ha="center", fontsize=10, style='italic')
    plt.show()
    
    return img_superpuesta

#comparamos las letras pedidas
comparar_letras_superposicion(df, 'S', 'M')
comparar_letras_superposicion(df, 'O', 'Q')
# %%
