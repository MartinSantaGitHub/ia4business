# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 04:49:22 2021

@author: msantamaria
"""

# Inteligencia Artifical aplicada a Negocios y Empresas - Caso Práctico 2

# Creación de la red Q profunda

# Importar las librerías
import numpy as np

# IMPLEMENTAR EL ALGORITMO DE DEEP Q-LEARNING CON REPETICIÓN DE EXPERIENCIA

class DQN(object):
    
    # INTRODUCCIÓN E INICIALIZACIÓN DE LOS PARÁMETROS Y VARIABLES DEL DQN
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        self.memory = []
        #self.memory = list()
        self.max_memory = max_memory
        self.discount_factor = discount_factor
        
    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA LA MEMORIA DE LA REPETICIÓN DE EXPERIENCIA
    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # CREACIÓN DE UN MÉTODO QUE CONSTRUYA DOS BLOQUES DE ENTRADAS Y TARGETS EXTRAYENDO TRANSICIONES
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        num_outputs = model.output_shape[-1]
        inputs = np.zeros((min(batch_size, len_memory), num_inputs))
        targets = np.zeros((min(batch_size, len_memory), num_outputs))
        
        for idx, i in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[i][0]
            game_over = self.memory[i][1]
            inputs[idx] = current_state
            targets[idx] = model.predict(current_state.reshape((1,-1)))[0]            
            Q_sa = np.max(model.predict(next_state.reshape((1,-1)))[0])
            
            if game_over:
                targets[idx, action] = reward
            else:
                targets[idx, action] = reward + self.discount_factor + Q_sa
                
        return inputs, targets
