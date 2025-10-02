"""
Utilidades de visualización para resultados de clasificación.
"""

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from utils.config import POINT_SIZE


def plot_classification_results(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[object] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    show_decision_boundary: bool = True
) -> plt.Axes:
    """
    Visualiza resultados de clasificación con o sin frontera de decisión.
    
    Args:
        X: Datos de entrada (n_muestras, n_características)
        y: Etiquetas verdaderas o predichas
        model: Modelo entrenado para mostrar frontera de decisión
        ax: Ejes de matplotlib donde graficar
        title: Título del gráfico
        show_decision_boundary: Si mostrar la frontera de decisión
        
    Returns:
        Ejes de matplotlib con el gráfico
    """
    if ax is None: _, ax = plt.subplots()
    
    # Mostrar frontera de decisión si hay modelo
    if model is not None and show_decision_boundary:
        DecisionBoundaryDisplay.from_estimator(
            model, X, ax=ax,
            response_method="predict",
            alpha=0.2,
            grid_resolution=100,
            cmap="viridis"
        )
    
    # Graficar puntos de datos
    sns.scatterplot(
        x=X[:, 0], y=X[:, 1], hue=y, alpha=0.9, ax=ax,
        palette="viridis", size=POINT_SIZE, legend=False
    )
    
    ax.set_title(title, fontsize=10)
    
    return ax