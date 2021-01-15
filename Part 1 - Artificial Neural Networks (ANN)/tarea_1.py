#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:54:54 2020

@author: Aronnvega
"""


#Parte 1 - Pre procesado de datos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# codificador de datos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Este es para pasar el pais 
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# Este es para pasar al sexo
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parte 2 - Construccion de la red neuronal(RNA)

# Importar Keras y librerias adicionales
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la Red Neuronal(RNA)
classifier =  Sequential()
 
# Anadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer="uniform",
                     activation = "relu", input_dim = 11)) 
# Anadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

# Anadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compilar la red neuronal 
classifier.compile(optimizer="adam",  loss="binary_crossentropy", 
                   metrics = ["accuracy"])

#Ajustamos la red neuronal al conjunto de entrenamiento

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Parte 3 - Evaluar el modelo y calcular predicciones finales 

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Utilizar nuestro modelo de RNA para predecir si el cliente con la siguiente 
#informacion abandonara el banco:
    #Geografia: Francia
    #Puntaje de credito: 600
    #Genero: masculino
    #edad: 40
    #tenencia:3 
    #saldo:$6000
    #numero de productos:2
    #Este cliente tiene tarjeta: Si
    #Es este cliente un miembro activo: Si
    #salario estimado: $50000
    #entonces deberiamos decir adios a este cliente?
 
new_predection = classifier.predict(sc_X.transform(np.array([[0,0,600, 1, 40, 
                                                              3, 60000, 2, 1, 1, 50000]])))
print(new_predection)
print(new_predection>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Para rectificar que probabilidad de exito hay 
(cm[0][0]+cm[1][1])/cm.sum()