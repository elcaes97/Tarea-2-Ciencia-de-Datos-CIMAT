from matplotlib import pyplot as plt
from utils.distributions import generata_data
from utils.visualitation import show_data
from utils.clasificadores import GaussianBayesClassifier
import numpy as np

np.random.seed(0)

# crear muestras
means = [np.array([-1, -1]), np.array([1, 1])]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]

X, y = generata_data(means, covs, N=[100, 500])

# clasificar muestras
model = GaussianBayesClassifier(means, covs)
model.fit(X, y)
y_pred = model.predict(X)

# mostrar muestras
fig, axes = plt.subplots(ncols=2, nrows=1)
show_data(X, y, ax=axes[0], title="Muestras Generadas")
show_data(X, y_pred, ax=axes[1], title="Muestras Clasificadas")
plt.show()



