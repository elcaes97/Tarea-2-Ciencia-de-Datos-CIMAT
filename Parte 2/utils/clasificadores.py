import numpy as np
from scipy.stats import multivariate_normal

class GaussianBayesClassifier:
    def __init__(self, means, covs, priors=None):
        """
        Inicializa el clasificador.
        
        Args:
            means (list): Lista de arrays con las medias de cada clase.
            covs (list): Lista de matrices de covarianza de cada clase.
            priors (list): Lista de priors de cada clase.
        """
        self.means = means
        self.covs = covs
        self.priors = priors

    def fit(self, X, y):
        """
        Entrena el clasificador con los datos y las etiquetas de clase.
        
        Args:
            X (array): Matriz de muestras (n_samples, n_features).
            y (array): Etiquetas de clase (n_samples,).
        """
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        self.data_size = len(X)
        self.n_classes = len(self.classes_)

        self.priors = self.counts_ / self.data_size if self.priors is None\
            else self.priors

        self.distributions = [ multivariate_normal(mean=self.means[i],
            cov=self.covs[i],allow_singular=True)\
            for i in range(self.n_classes) ]


    def predict(self, X):
        probs = np.zeros((len(X), self.n_classes))
        
        for i in range(self.n_classes):
            probs[:, i] = self.distributions[i].pdf(X) * self.priors[i]
        
        # Seleccionar la clase con mayor probabilidad
        return np.argmax(probs, axis=1)
            
        