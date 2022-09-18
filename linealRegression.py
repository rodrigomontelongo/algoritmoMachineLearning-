#librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
#clase del modelo de regresión lineal
class LinearRegression:
    def __init__(self, x , y):

        #datos de entrada
        self.data = x

        #datos de salida
        self.label = y

        #coeficiente b1 
        self.m = 0

        #coeficiente b0
        self.b = 0

        #tamaño de los datos
        self.n = len(x)
    
    #función para entrenar modelo
    def fit(self , epochs , lr):
         
        #gradient descent
        for i in range(epochs):

            #predecimos y con los coeficientes actuales
            y_pred = self.m * self.data + self.b
             
            #Calculamos las derivadas de los parámetros 
            D_m = (-2/self.n)*sum(self.data * (self.label - y_pred))
            D_b = (-1/self.n)*sum(self.label-y_pred)
             
            #actualizamos parámetros
            self.m = self.m - lr * D_m
            self.c = self.b - lr * D_b
             
    #función para predecir valor de y según entrada x
    def predict(self , inp):
        y_pred = self.m * inp + self.b 
        return y_pred

#cargamos datos en un dataframe
df = pd.read_csv('datos.csv')
 
#separamos datos en variable de entrada y salida
x = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])
 
#creamos objeto del modelo
modeloLineal = LinearRegression(x,y)
 
#entrenamos el modelo con 1000 iteraciones y un learning rate de 0.0001
modeloLineal.fit(1000 , 0.0001) 
 
#usamos el modelo entrenado para predecir los valores de y
y_pred = modeloLineal.predict(x)
 
#graficamos predicciones del modelo comparados con valor real
plt.figure(figsize = (10,6))

#gráfica con valores reales
plt.scatter(x,y , color = 'green')

#gráfica con valores que el modelo predijo
plt.plot(x , y_pred , color = 'k' , lw = 3)
plt.xlabel('x' , size = 20)
plt.ylabel('y', size = 20)
plt.show()
