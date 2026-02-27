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


#%% IMPORTS Y CONFIGURACIÓN INICIAL
"""
Módulos necesarios para el procesamiento de datos, visualización y modelado.
Se importan todas las bibliotecas requeridas y se configura el entorno de
visualización.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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
    
    Parametros
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
    
    Parametros
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
    letra = df.iloc[indice][columna_label]
    
    X = df.drop(columns=[columna_label])
    
    img = np.array(X.iloc[indice]).reshape(dimension[0], dimension[1])
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f'Letra: {letra} - Índice: {indice}', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    return img

#%% CARGA DE DATOS
contador_figuras = 0
DATA_PATH = './data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)

#%% ANÁLISIS EXPLORATORIO

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
contador_figuras += 1
plt.figure(figsize=(15,6))
letras = conteo_letras.keys()
cantidades = [conteo_letras[letra] for letra in letras]

barras = plt.bar(letras, cantidades)
plt.title("Distribución de letras en el df")
plt.xlabel("letra")
plt.ylabel("cantidad de muestras")
plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Distribución de letras en el dataset", 
            ha="center", fontsize=10, style='italic')
plt.show()

# =============================================================================
#### VISUALIZAR TODAS LAS LETRAS (para comparar similitudes)
# =============================================================================

# =============================================================================
#### Función para Visualizar Grilla completa
# =============================================================================
def visualizar_todas_letras_grilla(df,contador_figuras, dimension=(28, 28), tipografia=0):
    """
    Muestra una grilla con todas las letras del alfabeto (primera muestra de cada una).
    """
    letras_unicas = df['label'].unique()
    cols = 6
    rows = 5
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows*2.5))
    axes = axes.flatten()
    
    for i, letra in enumerate(letras_unicas):
        idx = df[df['label'] == letra].index[tipografia]
        
        X = df.drop(columns=['label'])
        img = np.array(X.iloc[idx]).reshape(dimension[0], dimension[1])
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Letra {letra}', fontsize=12, fontweight='bold')
        
    
    for i in range(0, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Todas las Letras del Alfabeto', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Distribución de letras en el dataset", 
                ha="center", fontsize=10, style='italic')
    plt.show()
contador_figuras += 1
visualizar_todas_letras_grilla(df,contador_figuras)

# =============================================================================
#### FUNCIÓN PARA COMPARAR PARES DE LETRAS
# =============================================================================
def comparar_letras_superposicion(df, letra1, letra2, contador_figuras,dimension=(28, 28)):
    idx1 = df[df['label'] == letra1].index[0]
    idx2 = df[df['label'] == letra2].index[0]
    
    X = df.drop(columns=['label'])
    
    img1 = np.array(X.iloc[idx1]).reshape(dimension[0], dimension[1])
    img2 = np.array(X.iloc[idx2]).reshape(dimension[0], dimension[1])
    
    img_superpuesta = np.ones((dimension[0], dimension[1], 3))  # Fondo blanco (1,1,1)
    
    mask_ambas = (img1 < 200) & (img2 < 200)
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
    plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Comparación {letra1} vs {letra2}", 
                ha="center", fontsize=10, style='italic')
    plt.show()
    
    return img_superpuesta

contador_figuras += 1
comparar_letras_superposicion(df, 'O', 'Q', contador_figuras)
contador_figuras += 1
comparar_letras_superposicion(df, 'I', 'T', contador_figuras)
contador_figuras += 1
comparar_letras_superposicion(df, 'C', 'G', contador_figuras)
contador_figuras += 1
comparar_letras_superposicion(df, 'P', 'R', contador_figuras)
contador_figuras += 1
comparar_letras_superposicion(df, 'C', 'O', contador_figuras)
contador_figuras += 1
comparar_letras_superposicion(df, 'I', 'L', contador_figuras)
# =============================================================================
#### FUNCIÓN PARA VISUALIZAR MÚLTIPLES MUESTRAS DE UNA LETRA
# =============================================================================

def visualizar_tipografia_letra(df, letra, contador_figuras,n_muestras=49, dimension=(28, 28)):
    """
    Muestra múltiples muestras de una misma letra para ver variabilidad.
    """
    df_letra = df[df['label'] == letra]
    indices = df_letra.head(n_muestras).index.tolist()
    
    cols = int(n_muestras**(1/2)) # sqrt 
    rows = int(n_muestras/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows*3))
    axes = axes.flatten()
    
    X = df.drop(columns=['label'])
    
    for i, idx in enumerate(indices):
        img = np.array(X.iloc[idx]).reshape(dimension[0], dimension[1])
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Muestra {i+1}', fontsize=10)
        axes[i].axis('off')
    
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Variabilidad en Letra {letra} - {len(indices)} muestras de 1016', 
                 fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Variabilidad en letra {letra}", 
                ha="center", fontsize=10, style='italic')
    plt.show()
contador_figuras += 1
visualizar_tipografia_letra(df, 'O', contador_figuras)
#%% CLASIFICACIÓN BINARIA

# =============================================================================
#### Subconjunto de imágenes correspondientes a las letras O y L
# =============================================================================
df_letra_O = df[(df['label'] == 'O')]
df_letra_L = df[(df['label'] == 'L')]

df_letras_OL = pd.concat([df_letra_O,df_letra_L])

print("Muestras de O:",len(df_letra_O))
print("Muestras de L:",len(df_letra_L))
print("Total de muestras:",len(df_letra_O)+len(df_letra_L))
# =============================================================================
# 
# # Falta: determinar si está balanceado con respecto a las 
# # dos clases a predecir (si la imagen es de la letra O o de la letra L).
# 
# =============================================================================

# =============================================================================
#### Separar los datos en conjuntos de train y test
# =============================================================================

X = df_letras_OL

y = df_letras_OL['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,stratify=y)

print(f"Train: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")
print("\nDistribución en train:")
print(y_train.value_counts())
print("\nDistribución en test:")
print(y_test.value_counts())

# =============================================================================
#### modelo de KNN sobre los datos de entrenamiento utilizando una cantidad 
### reducida de atributos(3)
# =============================================================================
#desviaciones = X_train.std().sort_values(ascending = False)

df_letra_O = X_train[X_train['label']=='O'].drop(columns=['label'])
df_letra_L = X_train[X_train['label']=='L'].drop(columns=['label'])
def rango_intercuartil(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    # Calcular el Rango Intercuartil (IQR)
    IQR = Q3 - Q1
    return IQR

IQRO = rango_intercuartil(df_letra_O)
IQRL = rango_intercuartil(df_letra_L)

media_O = df_letra_O.mean()
media_L = df_letra_L.mean()
diferencia = abs(media_O - media_L)

#IQR = ((IQRL-IQRO)**2)**(1/2)
#IQR_con_mayor = IQR.sort_values(ascending = False)

img = np.array(IQRO).reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

img = np.array(media_O).reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


IQR = abs(IQRL-IQRO)

img = np.array(IQR).reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

IQR_con_mayor = IQR.sort_values(ascending = False)
accuracies = []
diferencia = diferencia.sort_values(ascending = True)

for rango in range(1,784, 1):
    
    atributos = diferencia.iloc[rango- 1: rango].index.tolist()
    clasificador = KNeighborsClassifier(n_neighbors= 5)
    clasificador.fit(X_train[atributos], y_train)
    
    y_pred = clasificador.predict(X_test[atributos])
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((acc, list(atributos)))
    print(f"\nConjunto {rango - 3} - {rango} - Atributos: {atributos}")
    print(f"Accuracy: {round(acc, 4)}")


punto_maximo = max(accuracies, key = lambda x: x[0])
acurracies = [items[0] for items in accuracies]
plt.figure(figsize=(25, 5))

plt.plot(acurracies, 'bo-', linewidth=1, markersize=2)
plt.title('Accuracy para 5 conjuntos de atributos', fontsize=13, fontweight='bold')
plt.xlabel('Conjunto')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)