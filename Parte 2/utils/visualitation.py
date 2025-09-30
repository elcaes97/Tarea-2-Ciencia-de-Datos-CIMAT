import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from utils.config import points_size

def show_data(X:np.ndarray, y:np.ndarray, size_points=0.5, ax=None, model=None,\
    **ax_kwargs) -> None:
    """ Muestra los datos en un grafico de dispersión.

    Parámetros:
    -----------
    X : ndarray (n, d)
        Datos a clasificar (n puntos de dimensión d).
    y : ndarray (n,)
        Etiquetas de clase.
    ax : Axis
        Axis de matplotlib.
    model : sklearn model
        Clasificador.
    ax_kwargs : dict
        Argumentos para el axis de matplotlib.

    Retorna:
    --------
    None
    """
    # crear ejes si no se proporcionan
    if ax is None: _, ax = plt.subplots()

    if model is not None:   # fronteras de decision
        DecisionBoundaryDisplay.from_estimator(model, X, ax=ax,
        response_method="predict", alpha=0.2)

    # Mostrar datos
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, alpha=0.9, ax=ax,\
        palette="viridis", size=points_size, legend=False)
    ax.set(**ax_kwargs) # configuraciones adicionales
    return ax