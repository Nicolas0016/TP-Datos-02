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
print("Total de muestras:",len(df_letra_O)+len(df_letra_L))
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

nombres_cortos = ['Top 3', 'Sig 3', 'Med 1', 'Med 2', 'Med 3', 'Min 3']
accuracies = []
for i, attrs in enumerate(conjuntos, 1):
    clasificador = KNeighborsClassifier()
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
