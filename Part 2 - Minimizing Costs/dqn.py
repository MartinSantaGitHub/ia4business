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
    
