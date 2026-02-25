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
DATA_PATH = './data/letras.csv'
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