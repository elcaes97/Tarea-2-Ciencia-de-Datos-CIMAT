"""
Implementación de clasificador Bayesiano Gaussiano personalizado.
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Optional, List


class GaussianBayesClassifier:
    """
    Clasificador Bayesiano Gaussiano con parámetros predefinidos o estimados.
    
    Este clasificador puede funcionar en dos modos:
    1. Con parámetros predefinidos (medias y covarianzas)
    2. Estimando parámetros a partir de los datos (si no se proporcionan)
    """
    
    def __init__(
        self,
        means: Optional[List[np.ndarray]] = None,
        covs: Optional[List[np.ndarray]] = None,
        priors: Optional[List[float]] = None
    ):
        """
        Inicializa el clasificador Bayesiano Gaussiano.
        
        Args:
            means: Lista de medias para cada clase. Si es None, se estiman de los datos.
            covs: Lista de matrices de covarianza para cada clase. Si es None, se estiman.
            priors: Priors para cada clase. Si es None, se estiman de las frecuencias.
        """
        self.means = means
        self.covs = covs
        self.priors = priors
        self.distributions_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianBayesClassifier':
        """
        Ajusta el clasificador a los datos.
        
        Args:
            X: Datos de entrenamiento (n_muestras, n_características)
            y: Etiquetas (n_muestras,)
            
        Returns:
            self: Clasificador ajustado
        """
        self.classes_, counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        
        # Estimar priors si no se proporcionan
        if self.priors is None:
            self.priors = counts / len(y)
        
        # Estimar medias y covarianzas si no se proporcionan
        if self.means is None or self.covs is None:
            self.means_ = []
            self.covs_ = []
            for cls in self.classes_:
                X_cls = X[y == cls]
                self.means_.append(np.mean(X_cls, axis=0))
                self.covs_.append(np.cov(X_cls, rowvar=False))
        else:
            self.means_ = self.means
            self.covs_ = self.covs
        
        # Crear distribuciones normales multivariadas
        self.distributions_ = [
            multivariate_normal(
                mean=self.means_[i], 
                cov=self.covs_[i], 
                allow_singular=True
            )
            for i in range(self.n_classes_)
        ]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice las etiquetas de clase para las muestras dadas.
        
        Args:
            X: Datos a predecir (n_muestras, n_características)
            
        Returns:
            Etiquetas predichas (n_muestras,)
        """
        if self.distributions_ is None:
            raise ValueError("El clasificador no ha sido ajustado. Llame a fit() primero.")
        
        # Calcular probabilidades posteriores
        probabilities = np.zeros((len(X), self.n_classes_))
        
        for i in range(self.n_classes_):
            probabilities[:, i] = self.distributions_[i].logpdf(X) + np.log(self.priors[i])
        
        return self.classes_[np.argmax(probabilities, axis=1)]
    
    def set_params(self, **params) -> None:
        """Establece parámetros del clasificador."""
        for param, value in params.items():
            setattr(self, param, value)