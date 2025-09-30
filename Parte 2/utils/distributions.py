"""
Generación de datos sintéticos para experimentos de clasificación.
"""

import numpy as np
from typing import Union, List, Tuple


def generate_data(
    means: List[np.ndarray],
    covs: List[np.ndarray], 
    n_samples: Union[int, List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos sintéticos a partir de distribuciones normales multivariadas.
    
    Args:
        means: Lista de vectores de media para cada clase
        covs: Lista de matrices de covarianza para cada clase
        n_samples: Número de muestras por clase (int) o lista específica por clase
        
    Returns:
        Tuple con:
            - X: Array de características (n_muestras, n_características)
            - y: Array de etiquetas (n_muestras,)
            
    Raises:
        ValueError: Si los tamaños de means, covs y n_samples no coinciden
    """
    if len(means) != len(covs):
        raise ValueError("means y covs deben tener la misma longitud")
    
    # Convertir n_samples a lista si es un entero
    if isinstance(n_samples, int):
        n_samples = [n_samples] * len(means)
    elif len(n_samples) != len(means):
        raise ValueError("n_samples debe ser int o lista de la misma longitud que means")
    
    X_list, y_list = [], []
    
    for i, (mean, cov) in enumerate(zip(means, covs)):
        n = n_samples[i]
        samples = np.random.multivariate_normal(mean, cov, n)
        X_list.append(samples)
        y_list.append(np.full(n, i))
    
    return np.vstack(X_list), np.concatenate(y_list)