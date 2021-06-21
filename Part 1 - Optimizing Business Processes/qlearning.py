# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:49:36 2021

@author: msantamaria
"""

# Inteligencia Artificial Aplicada a Negocios y Empresas

# Parte 1 - Optimización de los flujos de trabajo en un almacen con Q-Learning

# Importación de las librerí­as
import numpy as np

# Configuración de los parámetros gamma y alpha para el algoritmo de Q-Learning
gamma = 0.75
alpha = 0.9

# PARTE 1 - DEFINICIÓN DEL ENTORNO

# Definición de los estados
location_to_state = {'A': 0, 
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

priority_rank = {'A': 5, 
                 'B': 9,
                 'C': 8,
                 'D': 10,
                 'E': 12,
                 'F': 11,
                 'G': 1,
                 'H': 7,
                 'I': 6,
                 'J': 4,
                 'K': 2,
                 'L': 3}

# Definición de las acciones
actions = [i for i in range(0,12)]

# Definición de las recompensas
# Columnas:    A,B,C,D,E,F,G,H,I,J,K,L
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0], # A
              [1,0,1,0,0,1,0,0,0,0,0,0], # B
              [0,1,0,0,0,0,1,0,0,0,0,0], # C
              [0,0,0,0,0,0,0,1,0,0,0,0], # D
              [0,0,0,0,0,0,0,0,1,0,0,0], # E
              [0,1,0,0,0,0,0,0,0,1,0,0], # F
              [0,0,1,0,0,0,1,1,0,0,0,0], # G
              [0,0,0,1,0,0,1,0,0,0,0,1], # H
              [0,0,0,0,1,0,0,0,0,1,0,0], # I
              [0,0,0,0,0,1,0,0,1,0,1,0], # J
              [0,0,0,0,0,0,0,0,0,1,0,1], # K
              [0,0,0,0,0,0,0,1,0,0,1,0]],# L
             dtype = 'float64')

# PARTE 2 - CONSTRUCCIóN DE LA SOLUCIÓN DE IA CON Q-LEARNING

# Transformación inversa de estados a ubicaciones
state_to_location = {state: location for location, state in location_to_state.items()}

# Crear la función final que nos devuelve la ruta óptima
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]    
    R_new[ending_state, ending_state] = 1000
    route = [starting_location]
    next_location = starting_location    
    
    # for i in range(12):
    #     location = state_to_location[i]
    #     pr = priority_rank[location]
    #     R_new[:,i] *= (1 / pr)
    
    # Inicialización de los valores Q
    Q = np.array(np.zeros(shape = (12,12)))

    # Implementación del proceso de Q-Learning
    for i in range(1000):
        current_state = np.random.randint(low = 0, high = 12)
        playable_actions = []
        
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
    
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    
    return route

# PARTE 3 - PONER EL MODELO EN PRODUCCIÓN
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

# Imprimir la ruta final
print("Ruta Elegida: ")
print(best_route('B','K','G'))
