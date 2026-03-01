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
from sklearn.metrics import precision_score
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
def limpiar_ruido(df, umbral=200):
    """Versión de una línea para filtrar ruido."""
    df_limpio = df.copy()
    if 'label' in df_limpio.columns:
        df_limpio[df_limpio.columns[1:]] = df_limpio[df_limpio.columns[1:]].clip(upper=umbral)
        df_limpio[df_limpio.columns[1:]] = df_limpio[df_limpio.columns[1:]].replace(umbral, 255)
    else:
        df_limpio = df_limpio.clip(upper=umbral)
        df_limpio = df_limpio.replace(umbral, 255)
    return df_limpio

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
def comparar_letras_superposicion(df, letra1, letra2, contador_figuras,dimension=(28, 28)):
    idx1 = df[df['label'] == letra1].index[0]
    idx2 = df[df['label'] == letra2].index[0]
    
    X = df.drop(columns=['label'])
    
    img1 = np.array(X.iloc[idx1]).reshape(dimension[0], dimension[1])
    img2 = np.array(X.iloc[idx2]).reshape(dimension[0], dimension[1])
    
    img_superpuesta = np.ones((dimension[0], dimension[1], 3))  # Fondo blanco (1,1,1)
    
    mask_ambas = ((img1 < 200) & (img2 < 200)) | ((img1==255) & (img2==255))
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
def rango_intercuartil(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    # Calcular el Rango Intercuartil (IQR)
    IQR = Q3 - Q1
    return IQR
def mostrar_letra(letra, title):
    img = np.array(letra).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontweight='bold')
    plt.axis('off')
    plt.show()
#%% CARGA DE DATOS
contador_figuras = 0
DATA_PATH = './data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)
df = limpiar_ruido(df)
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
plt.figure(figsize=(9,6))
letras = conteo_letras.keys()
cantidades = [conteo_letras[letra] for letra in letras]
promedio = np.mean(cantidades)
desvio = np.std(cantidades)
# Histograma
plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(cantidades, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Histograma: Distribución de Frecuencias', fontweight='bold')
plt.xlabel('Cantidad de Muestras')
plt.ylabel('Cantidad de Letras')
plt.xticks([1015, 1016, 1017])
plt.yticks([0,9,18,27])
plt.grid(True, alpha=0.3, axis='y')
plt.legend()
plt.figtext(0.705, -0.05, f"FIGURA {contador_figuras}", 
            ha="center", fontsize=10, style='italic')
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
visualizar_todas_letras_grilla(df,contador_figuras, tipografia=8)
# =============================================================================
#### FUNCIÓN PARA COMPARAR PARES DE LETRAS
# =============================================================================

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

def visualizar_tipografia_letra(df, letra, contador_figuras,n_muestras=9, dimension=(28, 28)):
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

print("\n--- ANÁLISIS DE BALANCE ---")
print(f"Proporción O: {len(df_letra_O)/len(df_letras_OL)}")
print(f"Proporción L: {len(df_letra_L)/len(df_letras_OL)}")
if abs(len(df_letra_O) - len(df_letra_L)) / min(len(df_letra_O), len(df_letra_L)) < 0.1:
    print("El dataset está balanceado")
else:
    print("El dataset no está perfectamente balanceado")


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

df_letra_O_train = X_train[y_train == 'O'].drop(columns = ['label'])
df_letra_L_train =  X_train[y_train == 'L'].drop(columns = ['label'])


# Mediana de clase de letra por pixel.
media_O = df_letra_O_train.mean()
media_L = df_letra_L_train.mean()
diferencia_medias = abs(media_O - media_L)

# Variacion de pixeles por clase de letra
IQR_O = rango_intercuartil(df_letra_O_train)
IQR_L = rango_intercuartil(df_letra_L_train)

diferencia_IQR = abs(IQR_O - IQR_L)


# Visualizar las diferencias (mediana y rango interquartil)
mostrar_letra(media_O, 'Media de O') 
mostrar_letra(IQR_O, 'Rango Intercuartílico de O') 

mostrar_letra(media_L, 'Media de L') 
mostrar_letra(IQR_L, 'Rango Intercuartílico de L')

mostrar_letra(diferencia_medias, 'Diferencia de Medias\n(O vs L)')
mostrar_letra(diferencia_IQR, 'Diferencia de IQR\n(O vs L)')

# =============================================================================
# Comparación de criterios de selección (IQR vs Diferencia de Medias)
# =============================================================================
atributos_por_IQR = diferencia_IQR.sort_values(ascending=False).index.tolist()
atributos_por_media = diferencia_medias.sort_values(ascending=False).index.tolist()

tamanos_atributos = [3, 5, 10, 20, 50, 100, 200, 400, 784]
resultados_IQR = []
resultados_media = []

for n_atributos in tamanos_atributos:
    # Entrenamiento por IQR
    atributos_IQR = atributos_por_IQR[:n_atributos]
    knn_IQR = KNeighborsClassifier(n_neighbors=5)
    knn_IQR.fit(X_train[atributos_IQR], y_train)
    y_pred_IQR = knn_IQR.predict(X_test[atributos_IQR])
    acc_IQR = accuracy_score(y_test, y_pred_IQR)
    resultados_IQR.append(acc_IQR)
    
    # Entrenamiento por Mediana
    atributos_media = atributos_por_media[:n_atributos]
    knn_media = KNeighborsClassifier(n_neighbors=5)
    knn_media.fit(X_train[atributos_media], y_train)
    y_pred_media = knn_media.predict(X_test[atributos_media])
    acc_media = accuracy_score(y_test, y_pred_media)
    resultados_media.append(acc_media)
    
# =============================================================================
## Comparación de criterios
# =============================================================================
contador_figuras += 1
plt.figure(figsize=(12, 8))

plt.plot(tamanos_atributos, resultados_IQR, 'bo-', linewidth=2, markersize=8, 
         label='Selección por IQR', markeredgecolor='black', markeredgewidth=1)
plt.plot(tamanos_atributos, resultados_media, 'rs-', linewidth=2, markersize=8, 
         label='Selección por Diferencia de Medias', markeredgecolor='black', markeredgewidth=1)

# =============================================================================
## marcar el mejor punto de cada curva
# =============================================================================
mejor_IQR_idx = np.argmax(resultados_IQR)
mejor_media_idx = np.argmax(resultados_media)
plt.plot(tamanos_atributos[mejor_IQR_idx], resultados_IQR[mejor_IQR_idx], 'o', 
         markersize=15, markeredgecolor='gold', markeredgewidth=3, markerfacecolor='blue')
plt.plot(tamanos_atributos[mejor_media_idx], resultados_media[mejor_media_idx], 's', 
         markersize=15, markeredgecolor='gold', markeredgewidth=3, markerfacecolor='red')
plt.legend()


# CUADRADO AMARILLO CON INFORMACIÓN
plt.annotate(f'{round(resultados_IQR[mejor_IQR_idx], 3)}', 
             (tamanos_atributos[mejor_IQR_idx], resultados_IQR[mejor_IQR_idx]),
             xytext=(10, 10), textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.annotate(f'{round(resultados_media[mejor_media_idx],3)}', 
             (tamanos_atributos[mejor_media_idx], resultados_media[mejor_media_idx]),
             xytext=(10, -20), textcoords='offset points', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.xlabel('Cantidad de Atributos', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparación de Criterios de Selección de Atributos\nIQR vs Diferencia de Medias (k=5)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=12)
plt.xscale('log')
plt.xlim(2, 1000)
plt.ylim(0.85, 1.005)
plt.xticks(tamanos_atributos)
plt.tight_layout()
plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Comparación IQR vs Diferencia de Medias", 
            ha="center", fontsize=10, style='italic')
plt.show()

# =============================================================================
## Modelos de KNN utilizando distintos atributos y distintos valores de k (vecinos).
# =============================================================================
contador_figuras += 1

# Crear matriz de resultados para diferentes k y cantidades de atributos
valores_k = [1, 3, 5, 7, 9, 11, 15]
tamanos_atributos_exp = [3, 5, 10, 20, 50, 100, 200, 400, 784]
resultados_completos = []

print("\n--- Variando k y cantidad de atributos... ---")
for n_atributos in tamanos_atributos_exp:
    atributos_seleccionados = atributos_por_IQR[:n_atributos]
    for k in valores_k:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[atributos_seleccionados], y_train)
        y_pred = knn.predict(X_test[atributos_seleccionados])
        acc = accuracy_score(y_test, y_pred)
        resultados_completos.append({
            'n_atributos': n_atributos,
            'k': k,
            'accuracy': acc
        })
print(":D")
df_resultados = pd.DataFrame(resultados_completos)
# Heatmap
plt.figure(figsize=(14, 8))
heatmap_data = df_resultados.pivot_table(
    values='accuracy', 
    index='n_atributos', 
    columns='k'
)

plt.imshow(heatmap_data, aspect='auto', cmap='Oranges', interpolation='nearest')
plt.colorbar(label='Accuracy', shrink=0.8)

# Configurar ejes
plt.xticks(range(len(valores_k)), valores_k, fontsize=11)
plt.yticks(range(len(tamanos_atributos_exp)), tamanos_atributos_exp, fontsize=11)

plt.xlabel('Valor de k (vecinos)', fontsize=12)
plt.ylabel('Cantidad de Atributos', fontsize=12)
plt.title('Heatmap de Accuracy: KNN con diferentes k y cantidad de atributos\n(Selección por IQR)', 
          fontsize=14, fontweight='bold')

# Agregar valores en las celdas
for i in range(len(tamanos_atributos_exp)):
    for j in range(len(valores_k)):
        valor = heatmap_data.iloc[i, j]
        color_texto = 'white' if valor > 0.85 else 'black'
        plt.text(j, i, f'{round(valor,3)}', 
                ha='center', va='center', color=color_texto, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Heatmap de resultados KNN", 
            ha="center", fontsize=10, style='italic')
plt.show()

