import pandas as pd 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units= 16, activation= 'relu', #Units = Entrada de dados + saidas /2
                        kernel_initializer= 'random_uniform', input_dim= 30)) 
classificador.add(Dense(units= 1, activation= 'sigmoid'))

classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy', 
                      metrics= ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size= 10, epochs= 100) #Batch_size calcula o erro para 10 registros e dpois ajusta os pesos (descida do gradiente estocástico)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

matrix = confusion_matrix(classe_teste, previsoes)
print(matrix)
#benigno = 0, maligno = 1

resultado = classificador.evaluate(previsores_teste, classe_teste)