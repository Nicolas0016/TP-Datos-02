#%% IMPORTS Y CONFIGURACIÓN INICIAL
"""
Módulos necesarios para el procesamiento de datos, visualización y modelado.
Se importan todas las bibliotecas requeridas y se configura el entorno de
visualización.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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
    y_pred = model.predict(X_held_out)#cambio X_test a x_held_out
    accuracy = accuracy_score(y_held_out, y_pred)#cambio y_test a y_held_out
    scores.append(accuracy)

    print(f"{profundidad}: {accuracy}")

#Grafico del accuracy por profundidad
plt.plot(range(1,21), scores, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy del arbol por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

"""
#grafico del arbol
plt.figure(figsize=(50,profundidad * 15))
plot_tree(model, filled=True,fontsize=10,rounded=True)
plt.show()
"""
# %% PUNTO 3
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y) #capaz que hay que usar held out

model = DecisionTreeClassifier(random_state=42)
param_grid = {'max_depth':range(1,10)}#nos piden usar profundidades de 1 a 10

grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1) #si se les traba todo pongan el n_jobs en 1 (-1 es que usa todos los nucleos del cpu, 1 es uno solo)
#cv = 5 es que se van dividir en 5 grupos

grid.fit(X_train,y_train)

#print(grid.cv_results_)
#de grid.cv_results_ seguro se puede sacar un grafico
#no se si cumplimos lo de mostrar la configuracion de hiperparametros
print(f"Modelo ganador: {grid.best_estimator_} \nAccuracy: {round(grid.best_score_,2)} \nProfundidad {grid.best_params_['max_depth']}")
# %%
