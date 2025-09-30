import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def show_data(X:np.ndarray, y:np.ndarray, ax=None, **ax_kwargs) -> None:
    """ Muestra los datos con sus etiquetas de clase.

    Parámetros:
    -----------
    X : ndarray (n, d)
        Datos a clasificar (n puntos de dimensión d).
    y : ndarray (n,)
        Etiquetas de clase.
    title : str
        Título de la figura.
    ax : Matplotlib axis
        Axis de la figura.

    Retorna:
    --------
    None
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep", alpha=0.75,
    ax=ax)
    ax.set(**ax_kwargs) 

    if ax is None: plt.show()
    return