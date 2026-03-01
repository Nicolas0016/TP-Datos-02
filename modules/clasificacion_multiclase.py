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
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

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
DATA_PATH = '../data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)
#%%PUNTO 1
X = df.drop(columns=['label'])
y= df['label']

X_dev, X_held_out, y_dev, y_held_out = train_test_split(X, y,test_size=0.2,stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev,test_size=0.3,stratify=y_dev)

#resultado = pd.concat([y_dev,X_dev],axis=1)
#%% PUNTO 2
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

#%%BORRARESTECOMENTARUI
#Grafico del accuracy por profundidad

plt.plot(range(1,21), scores, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy del arbol por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(0,21,2))
plt.figtext(0,0.03,'Figura 10')

indx_de_max = np.argmax(scores) #buscamos el indice del mayor accuracy para resaltarlo
plt.scatter(range(1,21)[indx_de_max],scores[indx_de_max],color = 'magenta',zorder = 2)

plt.show()
# %% PUNTO 3

model = DecisionTreeClassifier(random_state=42)
param_grid = {'max_depth':range(1,11)}#nos piden usar profundidades de 1 a 10

grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1) #si se les traba todo pongan el n_jobs en 1 (-1 es que usa todos los nucleos del cpu, 1 es uno solo)
#cv = 5 es que se van dividir en 5 grupos

grid.fit(X_dev,y_dev)

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
plt.show()

plt.figure(figsize=(8,6))
plt.plot(profundidades, accuracy_promedio, marker='o', linestyle='-',color = 'Green')
plt.title('Accuracy por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy promedio')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(tiempo_promedio, accuracy_promedio, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy por tiempo')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

print(f"Modelo ganador: {mejor_modelo} \nAccuracy: {round(grid.best_score_,2)} \nProfundidad {mejor_profundidad}")


#%% PUNTO 4
# evalúo con el mejor modelo (el del punto anterior)
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
