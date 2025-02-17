# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 04:48:43 2021

@author: msantamaria
"""

# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2

# Creación del Cerebro

# Importar las librerías
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

# CONSTRUCCIÓN DEL CEREBRO

class Brain(object):
    
    def __init__(self, learning_rate = 0.001, number_of_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape = (3,))
        
        x = Dense(units = 64, activation = 'sigmoid')(states)
        x = Dropout(rate = 0.1)(x)
        
        y = Dense(units = 32, activation = 'sigmoid')(x)
        y = Dropout(rate = 0.1)(y)
        
        q_values = Dense(units = number_of_actions, activation = 'softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        
        self.model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate)) 
    