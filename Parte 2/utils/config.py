"""
Configuración centralizada para experimentos de clasificación.
Define parámetros de modelos, distribuciones de datos y configuraciones de visualización.
"""

import numpy as np

# =============================================================================
# CONFIGURACIONES DE VISUALIZACIÓN
# =============================================================================
POINT_SIZE = 0.3  # Tamaño de puntos en gráficos

# =============================================================================
# PARÁMETROS DE MODELOS
# =============================================================================
KNN_WEIGHTS = 'distance'  # Función de peso para KNN
KS = [1, 3, 5, 11, 21]   # Valores de k para KNN
REPS = 20                 # Réplicas para experimentos

# =============================================================================
# CONFIGURACIONES DE DISTRIBUCIONES
# =============================================================================
# Medias para diferentes escenarios
DIFF_MEANS = [np.array([-1, 0]), np.array([0, 1])]
SAME_MEANS = [np.zeros(2), np.zeros(2)]
SEPARATED_MEANS = [np.array([-2, 0]), np.array([2, 0])]

# Matrices de covarianza para diferentes escenarios
SAME_COVS = [
    np.identity(2),
    np.identity(2)
]

LOW_COVS = [
    np.identity(2) * 0.1,
    np.identity(2) * 0.1
]

DIFF_COVS = [
    np.array([[1, 0], [0, 2]]),
    np.array([[2, 0], [0, 1]])
]

# =============================================================================
# CONFIGURACIONES DE MUESTREO
# =============================================================================
# Tamaños de muestra para clases balanceadas
SAMPLE_SIZES_BALANCED = [50, 100, 200, 500]

# Tamaños de muestra para clases desbalanceadas (clase 0, clase 1)
SAMPLE_SIZES_UNBALANCED = [
    [50, 200],
    [100, 400], 
    [200, 800],
    [500, 2000]
]