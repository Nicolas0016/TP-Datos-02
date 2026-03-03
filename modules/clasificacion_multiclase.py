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
from sklearn.metrics import confusion_matrix,precision_score
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
#%% CARGA DE DATOS
contador_figuras = 0
DATA_PATH = '../data/letras.csv'
df = cargar_datos(DATA_PATH)
df =  remane_labels(df)
df = limpiar_ruido(df)
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

#Grafico del accuracy por profundidad
plt.plot(range(1,21), scores, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy del arbol por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(0,21,2))
plt.figtext(0.1,0.03,'Figura 10')

indx_de_max = np.argmax(scores) #buscamos el indice del mayor accuracy para resaltarlo
plt.scatter(range(1,21)[indx_de_max],scores[indx_de_max],color = 'magenta',zorder = 2)

plt.show()
# %% PUNTO 3

model = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
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
plt.figtext(0.1,0.04,'Figura 11')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(profundidades, accuracy_promedio, marker='o', linestyle='-',color = 'Green')
plt.title('Accuracy por profundidad')
plt.xlabel('Profundidad')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figtext(0.1,0.04,'Figura 12')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(tiempo_promedio, accuracy_promedio, marker='o', linestyle='-',color = 'Blue')
plt.title('Accuracy por tiempo')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.figtext(0.1,0.04,'Figura 13')
plt.show()

print(f"Modelo ganador: {mejor_modelo} \nAccuracy: {round(grid.best_score_,3)} \nProfundidad {mejor_profundidad}")

#%% PUNTO 4
#entreno al modelo ganador con todos los datos
mejor_modelo.fit(X_dev,y_dev)

# lo evaluamos
y_pred = mejor_modelo.predict(X_held_out)
accuracy = round(accuracy_score(y_held_out, y_pred),2)

matriz_confusion = pd.DataFrame(confusion_matrix(y_held_out,y_pred))

# Calcular matriz de porcentajes
cm = confusion_matrix(y_held_out, y_pred)
cm_porcentaje = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Crear anotaciones combinadas
annot_matrix = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot_matrix[i, j] = f'{round(cm_porcentaje[i, j],1)}'

plt.figure(figsize=(14,12))
grafico = sns.heatmap(matriz_confusion, 
                      annot=annot_matrix, 
                      fmt='',
                      xticklabels=mejor_modelo.classes_,
                      yticklabels=mejor_modelo.classes_,
                      cmap="Greens")
plt.xlabel("Predicción")
grafico.xaxis.set_label_position("top")
plt.ylabel("Respuesta correcta")

plt.text(x = 20,y = 1,s=f"Accuracy: {accuracy}", fontsize = 11,color = 'Darkgreen')
plt.show()

#%% Barras de precisión por letra

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