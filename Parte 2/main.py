from matplotlib import pyplot as plt
from utils.distributions import generate_data
from utils.visualitation import show_data
from utils.clasificadores import GaussianBayesClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)   # Para la reproducibilidad
plt.rcParams['figure.constrained_layout.use'] = True # Para ajustar figuras

# Paramnetros de distribuciones
means = [np.array([-1, -1]), np.array([1, 1]), np.array([1, -1])]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
sizes = [400, 400, 450] # tamanios de cada muestra

X, y = generate_data(means, covs, N=sizes)  # Generar datos

# Mostrar la data
show_data(X, y, title='DataGenerated')
plt.savefig('./figures/data.png')
# plt.show()


# Pruebas visuales de los clasificadores

# Parametros de knn
kneighbors = 3
knn_weights = 'distance'

# Generar figura y ejes
fig, axs = plt.subplots(3, 2)

show_data(X, y, title='DataGenerated', ax=axs[0, 0])  # Mostrar datos generados

# Para mostrar clasificación bayes optimo
model = GaussianBayesClassifier(means, covs)
model.fit(X, y)
y_pred = model.predict(X)
show_data(X, y_pred, title=model.__class__.__name__, ax=axs[0, 1], model=model)

# Para mostrar clasificación con los otros clasificadores
models = [
    GaussianNB(), LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=kneighbors, weights=knn_weights)
    ]

for model in models:
    model.fit(X, y)
    y_pred = model.predict(X)
    show_data(X, y_pred, title=model.__class__.__name__,
        ax=axs[1+models.index(model)//2, models.index(model)%2], model=model)


fig.suptitle('Clasificación de datos con distribuciones normales')
#plt.show()
plt.savefig('./figures/classifiers.png')



# variando k en knn
knns_clasifiers = [
    KNeighborsClassifier(n_neighbors=k+1, weights='distance') for k in range(10)
]

fig, axes = plt.subplots(2, 5)

for i, knn in enumerate(knns_clasifiers):
    knn.fit(X, y)
    y_pred = knn.predict(X)
    show_data(X, y_pred, title=f'K={1+i}', ax=axes[i//5, i%5], model=model)

fig.suptitle('KNN con diferentes k')
#plt.show()
plt.savefig('./figures/knns.png')