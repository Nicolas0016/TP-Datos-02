"""
===============================================================================
TRABAJO PRÁCTICO N° 02 - Clasificación y Selección de Modelos
===============================================================================

Nombre del grupo: CRUD
Integrantes:
    - Argañaraz, Nicolás
    - Frontera, Axel 
    - Rojas, Emiliano 

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
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
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
    df_limpio[df_limpio.columns[1:]] = df_limpio[df_limpio.columns[1:]].clip(upper=umbral)
    df_limpio[df_limpio.columns[1:]] = df_limpio[df_limpio.columns[1:]].replace(umbral, 255)
    return df_limpio
def incrementar_contador_figuras():
    global contador_figuras
    contador_figuras += 1 
    return contador_figuras
def comparar_letras_superposicion(df, letra1, letra2,dimension=(28, 28)):
    contador_figuras = incrementar_contador_figuras()
    # Agarro solo las letras de interes.
    idx1 = df[df['label'] == letra1].index[0]
    idx2 = df[df['label'] == letra2].index[0]
    
    # saco lo molesto
    X = df.drop(columns=['label'])
    
    img1 = np.array(X.iloc[idx1]).reshape(dimension[0], dimension[1])
    img2 = np.array(X.iloc[idx2]).reshape(dimension[0], dimension[1])
    
    
    img_superpuesta = np.ones((dimension[0], dimension[1], 3))  # Fondo blanco (1,1,1)
    # me quedo con todos los pixeles que tengan estas caracteristicas:
        # necesito que sean más oscuros que un gris
        # y además en ambas imagenes debe habler un blanco blanco
    mask_ambas = ((img1 < 200) & (img2 < 200)) | ((img1==255) & (img2==255)) 
    img_superpuesta[mask_ambas] = [0, 1, 0]  # Verde (Red, Green, Blue) 
    
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
                ha="center",fontsize=23, style='italic')
    plt.show()
    
    return img_superpuesta
def rango_intercuartil(df):
    # cosa que dijo clara durante la corrección del tp 2 (no le digan a pablito)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)

    # Calcular el Rango Intercuartil (IQR)
    IQR = Q3 - Q1
    return IQR
def mostrar_letra(letra, title):
    contador_figuras = incrementar_contador_figuras()
    img = np.array(letra).reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontweight='bold')
    plt.axis('off')
    plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}", 
                ha="center", fontsize=10, style='italic')
    plt.show()
def visualizar_tipografia_letra(df, letra,n_muestras=9, dimension=(28, 28)):
    """
    Muestra múltiples muestras de una misma letra para ver variabilidad.
    """
    contador_figuras = incrementar_contador_figuras()
    df_letra = df[df['label'] == letra]
    indices = df_letra.head(n_muestras).index.tolist()
    
    cols = int(n_muestras**(1/2)) # sqrt 
    rows = int(n_muestras/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows*3))
    # aplana la matriz de ejes para iterar sobre ellos con solo el índice i 
    axes = axes.flatten()
    
    X = df.drop(columns=['label'])
    
    for i, idx in enumerate(indices):
        img = np.array(X.iloc[idx]).reshape(dimension[0], dimension[1])
        # Mete la imagen en su espacio que le corresponde.
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Muestra {i+1}', fontsize=10)
        axes[i].axis('off') # Oculta los ejes (números y ticks) para limpieza 
    
    for i in range(len(indices), len(axes)):
        axes[i].axis('off') # Oculta los ejes
    
    plt.suptitle(f'Variabilidad en Letra {letra} - {len(indices)} muestras de 1016', 
                 fontsize=16, fontweight='bold')
    plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Variabilidad en letra {letra}", 
                ha="center", fontsize=23, style='italic')
    plt.show()
def visualizar_todas_letras_grilla(df, dimension=(28, 28), tipografia=0):
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
    plt.figtext(0.5, 0, f"FIGURA {incrementar_contador_figuras()}: Distribución de letras en el dataset", 
                ha="center", fontsize=23, style='italic')
    plt.show()
#%% CARGA DE DATOS
contador_figuras = 0
DATA_PATH = './data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)

#%% MÉTRICAS PARA VER LA CALIDAD DE DATOS

pixeles = df.drop(columns=['label'])
umbral = 200
pixeles_totales = 784

# DF booleano que indica True si es pixel es > 200
df_fondo = pixeles > umbral
pixeles_fondo_por_imagen = df_fondo.sum(axis=1)
porcentaje_fondo = (pixeles_fondo_por_imagen / pixeles_totales) * 100

porcentaje_global = porcentaje_fondo.mean()
print(f"M1: Porcentaje promedio de fondo/ruido en el dataset: {porcentaje_global:.1f}%")

df_gris = (pixeles >= 150) & (pixeles <= 200)
pixeles_gris_por_imagen = df_gris.sum(axis=1)

porcentaje_gris_por_imagen = (pixeles_gris_por_imagen / pixeles_totales) * 100
porcentaje_grises = porcentaje_gris_por_imagen.mean()

print(f"M2: Porcentaje promedio de píxeles en gris: {porcentaje_grises:.1f}%")


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
incrementar_contador_figuras()
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

visualizar_todas_letras_grilla(df)
visualizar_todas_letras_grilla(df, tipografia=8)

# =============================================================================
#### FUNCIÓN PARA VISUALIZAR MÚLTIPLES MUESTRAS DE UNA LETRA
# =============================================================================

visualizar_tipografia_letra(df, 'O')

# =============================================================================
#### COMPARAR PARES DE LETRAS
# =============================================================================

comparar_letras_superposicion(df, 'O', 'Q')
comparar_letras_superposicion(df, 'P', 'R')
comparar_letras_superposicion(df, 'S', 'M')
df = limpiar_ruido(df)
comparar_letras_superposicion(df, 'O', 'Q')
comparar_letras_superposicion(df, 'P', 'R')
comparar_letras_superposicion(df, 'S', 'M')


#%% CLASIFICACIÓN BINARIA

# =============================================================================
#### Subconjunto de imágenes correspondientes a las letras O y L
# =============================================================================
df_letra_O = df[(df['label'] == 'O')]
df_letra_L = df[(df['label'] == 'L')]

df_letras_OL = pd.concat([df_letra_O,df_letra_L])
print("Muestras de O:",len(df_letra_O))
print("Muestras de L:",len(df_letra_L))
print("Total de muestras:",len(df_letras_OL))

print("\n--- ANÁLISIS DE BALANCE ---")
print(f"Proporción O: {len(df_letra_O)/len(df_letras_OL)}")
print(f"Proporción L: {len(df_letra_L)/len(df_letras_OL)}")
if abs(len(df_letra_O) - len(df_letra_L)) / min(len(df_letra_O), len(df_letra_L)) < 0.1:
    print("El dataset está balanceado")
else:
    print("El dataset no está perfectamente balanceado")


#%% =============================================================================
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
mostrar_letra(media_L, 'Media de L')
mostrar_letra(IQR_O, 'Rango Intercuartílico de O')  
mostrar_letra(IQR_L, 'Rango Intercuartílico de L')

mostrar_letra(diferencia_medias, 'Diferencia de Medias\n(O vs L)')
mostrar_letra(diferencia_IQR, 'Diferencia de IQR\n(O vs L)')

#%% =============================================================================
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
incrementar_contador_figuras()
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
plt.figtext(0.5, 0, f"FIGURA {contador_figuras}: Comparación IQR vs Diferencia de Medias", 
            ha="center", fontsize=15, style='italic')
plt.show()

# =============================================================================
## Modelos de KNN utilizando distintos atributos y distintos valores de k (vecinos).
# =============================================================================
incrementar_contador_figuras()

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
#%% 
# Heatmap
plt.figure(figsize=(14, 8))
heatmap_data = df_resultados.pivot_table(
    values='accuracy', 
    index='n_atributos', 
    columns='k'
)

plt.imshow(heatmap_data, aspect='auto', cmap='Oranges', interpolation='nearest')
plt.colorbar(label='Accuracy', shrink=0.8)

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
plt.figtext(0.43, 0, f"FIGURA {contador_figuras}: Heatmap de resultados KNN", 
            ha="center", fontsize=15, style='italic')
plt.show()


# %% CLASIFICACIÓN MULTICLASE

# =============================================================================
#### Separar el conjunto de datos en desarrollo (dev) y validación (held-out)
# =============================================================================
#vuelvo a definir x e y
X = df.drop(columns=['label'])
y= df['label']

X_dev, X_held_out, y_dev, y_held_out = train_test_split(X, y,test_size=0.2,stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev,test_size=0.3,stratify=y_dev)


#%% =============================================================================
#### Ajustar un modelo de árbol. Probar con distintas profundidades.
# =============================================================================
scores = []

print("--------- Acuracies (por profundidad) ---------")
for profundidad in range(1,21):
    # Entrenamiento
    model = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)

    print(f"{profundidad}: {accuracy}")

incrementar_contador_figuras()
plt.plot(range(1,21), scores, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy del arbol por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(0,21,2))
plt.figtext(0.1,0.03,f"Figura {contador_figuras}")

indx_de_max = np.argmax(scores) #buscamos el indice del mayor accuracy para resaltarlo
plt.scatter(range(1,21)[indx_de_max],scores[indx_de_max],color = 'magenta',zorder = 2)

plt.show()


#%% =============================================================================
#### Comparar y seleccionar distintos árboles de decisión, con distintos hiperparámetos. 
# =============================================================================
model = DecisionTreeClassifier(random_state=42)
param_grid = {'max_depth':range(1,11)}#nos piden usar profundidades de 1 a 10

grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1) #si se les traba todo pongan el n_jobs en 1 (-1 es que usa todos los nucleos del cpu, 1 es uno solo)
#cv = 5 es que se van dividir en 5 grupos

grid.fit(X_train,y_train)

mejor_modelo = grid.best_estimator_
mejor_profundidad = grid.best_params_['max_depth']

#GRAFICO (pongo estos pero habria que dejar solo uno o dos, o combinarlos)
resultados = pd.DataFrame(grid.cv_results_)

profundidades = resultados['param_max_depth']
accuracy_promedio = resultados['mean_test_score']
tiempo_promedio = resultados['mean_fit_time']
#accracy_deriv_estandar = resultados['std_test_score']

plt.figure(figsize=(8,6))
plt.plot(profundidades, tiempo_promedio, marker='o', linestyle='-',color = 'Red')
plt.title('Tiempo de entrenamiento por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Tiempo (segundos)')
plt.grid(True)
plt.figtext(0.1,0.04,'Figura 22')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(profundidades, accuracy_promedio, marker='o', linestyle='-',color = 'Green')
plt.title('Accuracy por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figtext(0.1,0.04,'Figura 21')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(tiempo_promedio, accuracy_promedio, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy por tiempo')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figtext(0.1,0.04,'Figura 20')
plt.show()

print(f"Modelo ganador: {mejor_modelo} \nAccuracy: {round(grid.best_score_,2)} \nProfundidad {mejor_profundidad}")


# =============================================================================
#### Entrenar el modelo elegido a partir del inciso previo, ahora en todo el conjunto de desarrollo.
# =============================================================================
#entreno al modelo ganador con todos los datos
mejor_modelo.fit(X_dev,y_dev)

# lo evaluamos
y_pred = mejor_modelo.predict(X_held_out)
accuracy = round(accuracy_score(y_held_out, y_pred),2)

matriz_confusion = pd.DataFrame(confusion_matrix(y_held_out,y_pred))

plt.figure(figsize=(8,6))
grafico = sns.heatmap(matriz_confusion,xticklabels=mejor_modelo.classes_,yticklabels=mejor_modelo.classes_,cmap="Greens") #le podemos cambiar el color con cmap, tipo "Reds" o "Blues"
plt.xlabel("Predicción")
grafico.xaxis.set_label_position("top")
plt.ylabel("Respuesta correcta")

plt.text(x = 20,y = 1,s=f"Accuracy: {accuracy}", fontsize = 11,color = 'Darkgreen')
plt.show()

precision_por_letra = precision_score(y_held_out, y_pred, average=None)
plt.figure(figsize=(14, 6))
barras = plt.bar( ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'], precision_por_letra,edgecolor='black', linewidth=1)

# Añadir los valores en las barras
for i, (barra, valor) in enumerate(zip(barras, precision_por_letra)):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01, 
             f'{round(valor, 3)}', ha='center', va='bottom', fontsize=9, rotation=0)

plt.axhline(y=np.mean(precision_por_letra), color='red', linestyle='--', linewidth=2, 
            label=f'Precisión promedio: {round(np.mean(precision_por_letra),3)}')

plt.title('Precisión del modelo por letra (árbol de decisión)', fontsize=16, fontweight='bold')
plt.xlabel('Letra', fontsize=12)
plt.ylabel('Precisión', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.legend(loc='lower right')
plt.figtext(0.5, -0.05, 'Figura 14 - Precisión por letra', ha='center', fontsize=10)
plt.tight_layout()
plt.show()
# %%
