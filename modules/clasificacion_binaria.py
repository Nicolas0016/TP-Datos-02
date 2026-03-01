#%% LIBRERIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#%% CARGA DE DATOS
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

#%% PUNTO A
df_letra_O = df[(df['label'] == 'O')]
df_letra_L = df[(df['label'] == 'L')]

df_letras_OL = pd.concat([df_letra_O,df_letra_L])

print("Muestras de O:",len(df_letra_O))
print("Muestras de L:",len(df_letra_L))
print("Total de muestras:",len(df_letras_OL))
#%% 2.b: SEPARAR TRAIN Y TEST
# ===============================================================================
X = df_letras_OL.drop(columns=['label'])
y= df_letras_OL['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,stratify=y)

print(f"Train: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")
print("\nDistribución en train:")
print(y_train.value_counts())
print("\nDistribución en test:")
print(y_test.value_counts())

#%% PUNTO C
# Conjunto con tres atributos

# Desviación Estándar representa cuánto se apartan los valores del valor promedio
desviaciones = X_train.std().sort_values(ascending = False)

# Agaramos los 3 conjuntos con mayor tinta
# lo podriamos seleccionar nosotros.
conjunto1 = desviaciones.iloc[0:3].index.tolist()    
conjunto2 = desviaciones.iloc[3:6].index.tolist()   
conjunto3 = desviaciones.iloc[6:9].index.tolist()  
conjunto4 = desviaciones.iloc[50:53].index.tolist()  
conjunto5 = desviaciones.iloc[100:103].index.tolist()
conjunto6 = desviaciones.tail(3).index.tolist() 
conjuntos = [conjunto1, conjunto2, conjunto3,conjunto4,conjunto5,conjunto6]

nombres_cortos = [f'Grupo {numero}' for numero in range(1,7)]
accuracies = []
for i, attrs in enumerate(conjuntos, 1):
    clasificador = KNeighborsClassifier(n_neighbors= 5)
    clasificador.fit(X_train[attrs], y_train)
    
    y_pred = clasificador.predict(X_test[attrs])
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"\nConjunto {i} - Atributos: {attrs}")
    print(f"Accuracy: {round(acc, 4)}")
    
plt.figure(figsize=(8, 5))

plt.plot([1, 2, 3,4,5,6], accuracies, 'bo-', linewidth=2, markersize=10)
plt.title('Accuracy para 5 conjuntos de atributos', fontsize=13, fontweight='bold')
plt.xlabel('Conjunto')
plt.ylabel('Accuracy')
plt.xticks([1, 2, 3,4,5,6], nombres_cortos)
plt.ylim([0.5, 1.0])
plt.grid(True, alpha=0.3)

for i, acc in enumerate(accuracies, 1):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

#%% DIFERENTES CANTIDADES DE ATRIBUTOS
cantidades = [3, 5, 10, 20, 25, 50, 100,150, 200,250,300,350, 400,450, 500,784]  # 784 son todos
resultados_cantidades = []
for n in cantidades:
    pixeles_con_mayor_cambio = desviaciones.head(n).index.tolist()
    X_train_actual = X_train[pixeles_con_mayor_cambio].values
    X_test_actual = X_test[pixeles_con_mayor_cambio].values
    
    clasificador = KNeighborsClassifier(n_neighbors=5)
    clasificador.fit(X_train_actual, y_train)
    
    y_pred = clasificador.predict(X_test_actual)
    accuracy = accuracy_score(y_test, y_pred)
    
    resultados_cantidades.append((n, accuracy))
    
mejor_numero_variables, mejor_accuracy  = max(resultados_cantidades, key=lambda x: x[1])
peor_numero_variables, peor_accuracy = min(resultados_cantidades, key=lambda x: x[1])

plt.figure(figsize=(25, 5))
plt.subplot(1, 2, 2)

n_plot = [X[0] for X in resultados_cantidades]
acc_plot = [X[1] for X in resultados_cantidades]

plt.plot(n_plot, acc_plot, marker='o', linestyle='-')
# Punto con mejor accuracy
plt.plot(mejor_numero_variables, mejor_accuracy, 'ro', markersize=12, markeredgecolor='black')

plt.plot(peor_numero_variables, peor_accuracy, 'ro', markersize=12, markeredgecolor='black')


plt.title('Accuracy vs Cantidad de Atributos')

plt.text(mejor_numero_variables + 10, mejor_accuracy-0.015, f'Accuracy: {round(mejor_accuracy,3)}\nCantidad Atributos: {mejor_numero_variables}', 
         ha='center', va='bottom', fontsize=11)


plt.xlabel('Cantidad de atributos')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.show()
#%%
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


img = np.array(media_L).reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


img = np.array(IQRL).reshape(28, 28)
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


img = np.array(diferencia).reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

IQR_con_mayor = IQR.sort_values(ascending = False)
accuracies = []
diferencia = diferencia.sort_values(ascending = False)
exactitudes = []
for rango in range(3,784, 3):
    
    atributos = IQR_con_mayor.iloc[rango- 3: rango].index.tolist()
    clasificador = KNeighborsClassifier(n_neighbors= 5)
    clasificador.fit(X_train[atributos], y_train)
    
    y_pred = clasificador.predict(X_test[atributos])
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((acc, list(atributos)))
    #exactitud = precision_score(y_test, y_pred)
   # exactitudes.append((exactitud, list(atributos)))
    
    #print(f"Exactitud del modelo: {exactitud:.2f}")
    print(f"\nConjunto {rango - 3} - {rango} - Atributos: {atributos}")
    print(f"Accuracy: {round(acc, 4)}")


punto_maximo = max(accuracies, key = lambda x: x[0])
acurracies = [items[0] for items in accuracies]
exact = [items[0] for items in exactitudes] 
plt.figure(figsize=(25, 5))

plt.plot(acurracies, 'bo-', linewidth=1, markersize=2)
plt.title('Accuracy para 5 conjuntos de atributos', fontsize=13, fontweight='bold')
plt.xlabel('Conjunto')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.figure(figsize=(25, 5))

plt.plot(exact, 'bo-', linewidth=1, markersize=2)
plt.title('Accuracy para 5 conjuntos de atributos', fontsize=13, fontweight='bold')
plt.xlabel('Conjunto')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)


#%%
valores_k = [1, 5,9, 11, 15]
tamanos_atributos = [3, 10, 50, 100, 200, 400, 784]
resultados = []
for n_atributos in tamanos_atributos:
    atributos_seleccionados = IQR_con_mayor[:n_atributos].index.tolist()
    
    for k in valores_k:
        # Entrenar modelo
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[atributos_seleccionados], y_train)
        
        # Evaluar
        y_pred = knn.predict(X_test[atributos_seleccionados])
        accuracy = accuracy_score(y_test, y_pred)
        
        resultados.append({
            'n_atributos': n_atributos,
            'k': k,
            'accuracy': accuracy
        })
        
        print(f"Atributos: {n_atributos:3d} | k: {k:2d} | Accuracy: {accuracy:.4f}")

df_resultados = pd.DataFrame(resultados)
#%%
contador_figuras += 1

# Crear figura correctamente
fig, ax = plt.subplots(figsize=(10, 9))

# Graficar cada curva de k
for k in valores_k:
    # Filtrar datos para este k
    df_k = df_resultados[df_resultados['k'] == k]
    df_k = df_k.sort_values('n_atributos')
    
    ax.plot(df_k['n_atributos'], df_k['accuracy'], 
            'o-', label=f'k={k}', markersize=4, linewidth=1.5)

ax.set_xlabel('Cantidad de Atributos', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Comparación de Modelos KNN: Diferentes Atributos y Diferentes k', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.set_xscale('log')
plt.tight_layout()
plt.figtext(0.5, 0, f"FIGURA {contador_figuras}: Comparación de modelos KNN", 
            ha="center", fontsize=10, style='italic')
plt.show()
#%%
contador_figuras += 1

# Crear matriz para heatmap
heatmap_data = df_resultados.pivot_table(
    values='accuracy', 
    index='n_atributos', 
    columns='k'
)

plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar(label='Accuracy')
# Configurar ejes
plt.xticks(range(len(valores_k)), valores_k)
plt.yticks(range(len(tamanos_atributos)), tamanos_atributos)

plt.xlabel('Valor de k')
plt.ylabel('Cantidad de Atributos')

# Agregar valores en las celdas
for i in range(len(tamanos_atributos)):
    for j in range(len(valores_k)):
        plt.text(j, i, f'{heatmap_data.iloc[i, j]:.3f}', 
                ha='center', va='center', color='white', fontsize=9)

plt.tight_layout()
plt.show()
#%%
mejor_modelo = df_resultados.loc[df_resultados['accuracy'].idxmax()]
contador_figuras += 1
k_optimo_global = mejor_modelo['k']
plt.figure(figsize=(12, 6))
df_filtrado = df_resultados.sort_values('n_atributos')
plt.plot(df_filtrado['n_atributos'], df_filtrado['accuracy'], linewidth=2, markersize=6)

plt.xlabel('Cantidad de Atributos')
plt.ylabel('Accuracy')
plt.title(f'Comparación de Criterios de Selección (k={k_optimo_global})', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xscale('log')
plt.tight_layout()
plt.figtext(0.5, 0.01, f"FIGURA {contador_figuras}: Comparación de criterios con k={k_optimo_global}", 
            ha="center", fontsize=10, style='italic')
plt.show()