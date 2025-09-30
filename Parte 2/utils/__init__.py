
from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from utils.distributions import generate_data
from utils.visualitation import plot_classification_results
from utils.clasificadores import GaussianBayesClassifier
from utils.config import *
import os


def setup_environment():
    """Configura el entorno para reproducibilidad y calidad de gráficos."""
    np.random.seed(0)  # Para reproducibilidad
    plt.rcParams['figure.constrained_layout.use'] = True
    # Crear directorio para figuras si no existe
    os.makedirs('./figures', exist_ok=True)


def get_classifiers(means, covs):
    """
    Retorna lista de clasificadores a comparar.
    
    Returns:
        Lista de tuplas (nombre, instancia del clasificador)
    """
    return [
        ('GNB', GaussianNB()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier(weights=KNN_WEIGHTS)),
        ('GB', GaussianBayesClassifier(means, covs))
    ]


def create_comparison_figure(means, covs, sample_sizes, title_suffix, filename_suffix):
    """
    Crea figura comparativa de clasificadores para diferentes tamaños de muestra.
    
    Args:
        means: Medias para generación de datos
        covs: Covarianzas para generación de datos  
        sample_sizes: Lista de tamaños de muestra a evaluar
        title_suffix: Sufijo para el título de la figura
        filename_suffix: Sufijo para el nombre del archivo
    """
    classifiers = get_classifiers(means, covs)
    n_rows = len(classifiers) + 1  # +1 para datos originales
    n_cols = len(sample_sizes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    
    for col_idx, n in enumerate(sample_sizes):
        # Generar datos
        X, y = generate_data(means, covs, n_samples=n)
        
        # Mostrar datos originales
        plot_classification_results(
            X, y, 
            title=f'Datos Originales\nN={n}', 
            ax=axes[0, col_idx] if n_cols > 1 else axes[0]
        )
        
        # Entrenar y evaluar clasificadores
        for row_idx, (name, model) in enumerate(classifiers, start=1):
            model.fit(X, y)
            y_pred = model.predict(X)
            
            current_ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            plot_classification_results(
                X, y_pred, model=model,
                title=f'{name}\nN={n}', 
                ax=current_ax
            )
    
    fig.suptitle(f'Comparación de Clasificadores - {title_suffix}', fontsize=16)
    plt.savefig(f'./figures/classifier_comparison_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_knn_study_figure(means, covs, sample_sizes, title_suffix, filename_suffix):
    """
    Crea estudio de sensibilidad del parámetro k en KNN.
    
    Args:
        means: Medias para generación de datos
        covs: Covarianzas para generación de datos
        sample_sizes: Lista de tamaños de muestra a evaluar
        title_suffix: Sufijo para el título de la figura
        filename_suffix: Sufijo para el nombre del archivo
    """
    n_rows = len(KS) + 1  # +1 para datos originales
    n_cols = len(sample_sizes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    
    for col_idx, n in enumerate(sample_sizes):
        X, y = generate_data(means, covs, n_samples=n)
        
        # Mostrar datos originales
        plot_classification_results(
            X, y,
            title=f'Datos Originales\nN={n}',
            ax=axes[0, col_idx] if n_cols > 1 else axes[0]
        )
        
        # Probar diferentes valores de k
        for row_idx, k in enumerate(KS, start=1):
            knn = KNeighborsClassifier(weights=KNN_WEIGHTS, n_neighbors=k)
            knn.fit(X, y)
            y_pred = knn.predict(X)
            
            current_ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
            plot_classification_results(
                X, y_pred, model=knn,
                title=f'KNN (k={k})\nN={n}',
                ax=current_ax
            )
    
    fig.suptitle(f'Estudio de Sensibilidad de K en KNN - {title_suffix}', fontsize=16)
    plt.savefig(f'./figures/knn_study_{filename_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
