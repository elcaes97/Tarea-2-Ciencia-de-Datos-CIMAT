import numpy as np

def generata_data(means, covs, N):
    """ Genera N muestras de múltiples distribuciones normales.

    Parámetros:
    -----------
    means : list of ndarray
        Lista con los vectores de medias de cada clase.
    covs : list of ndarray
        Lista con las matrices de covarianza de cada clase.
    N : int or list
        Número de muestras a generar por clase.

    Retorna:
    --------
    X : ndarray (n, d)
        Datos a clasificar (n puntos de dimensión d).
    y : ndarray (n,)
        Etiquetas de clase.
    """
    X_list, y_list = [], []  # para los datos y las etiquetas de cada clase
    
    # Determinar el número de muestras por clase
    N_samples = [N] * len(means) if isinstance(N, int) else N
    
    # Generar datos para cada clase
    for i, (mean, cov) in enumerate(zip(means, covs)):
        # Generar muestras de la distribución normal multivariada
        samples = np.random.multivariate_normal(mean, cov, N_samples[i])
        X_list.append(samples)
        y_list.append(np.full(N_samples[i], i))  # Etiquetas con el índice de la clase
    
    # Concatenar todos los datos y etiquetas
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y