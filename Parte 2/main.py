from matplotlib import pyplot as plt
from utils.distributions import generate_data
from utils.visualitation import show_data
from utils.clasificadores import GaussianBayesClassifier
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from utils.config import *
import os as os

np.random.seed(0)   # Para la reproducibilidad
plt.rcParams['figure.constrained_layout.use'] = True # Para ajustar figuras



# Empezar con las pruebas

## Datos con distintas medias y mismas covs (Clases proporcionales)

models = [  # Los modelos para estas pruebas
    GaussianNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(weights=knn_weights),
    GaussianBayesClassifier(diff_means, same_covs)
]

titles = ['DataGen','GNB','LDA','QDA','KNN','GB']

fig, axes = plt.subplots(len(titles), len(Ns_prop))

for n in Ns_prop:
    X, y = generate_data(diff_means, same_covs, N=n)

    show_data(X, y, title=titles[0]+f' N={n}', ax=axes[0, Ns_prop.index(n)])

    for j, model in enumerate(models):
        model.fit(X, y)
        y_pred = model.predict(X)
        show_data(X, y_pred, title=titles[j+1]+f' N={n}',\
            ax=axes[j+1, Ns_prop.index(n)], model=model)

fig.suptitle('Comparación de clasificadores con mismos tamaños de muestras')
current_size = fig.get_size_inches()
fig.set_size_inches(current_size[0]*1.5, current_size[1]*1.5)
plt.savefig('./figures/same_covs_and_same_priors_classifiers.png')


## Datos con distintas medias y mismas covs (Clases desproporcionales)

fig, axes = plt.subplots(len(titles), len(Ns_nonprop))

for n in Ns_nonprop:
    X, y = generate_data(diff_means, same_covs, N=n)

    show_data(X, y, title=titles[0]+f' N={n}', ax=axes[0, Ns_nonprop.index(n)])

    for j, model in enumerate(models):
        model.fit(X, y)
        y_pred = model.predict(X)
        show_data(X, y_pred, title=titles[j+1]+f' N={n}',\
            ax=axes[j+1, Ns_nonprop.index(n)], model=model)

fig.suptitle('Comparación de clasificadores con diferentes tamaños de muestras')
current_size = fig.get_size_inches()
fig.set_size_inches(current_size[0]*1.5, current_size[1]*1.5)
plt.savefig('./figures/same_covs_and_distinct_priors_classifiers.png')


### Ahora variaremos k en knn (Datos proporcionales)

fig, axes = plt.subplots(nrows = len(ks)+1, ncols = len(Ns_prop))

for row, k in enumerate(ks):
    for col, n in enumerate(Ns_prop):
        X, y = generate_data(diff_means, same_covs, N=n)

        if row == 0:
            show_data(X, y, title=f'DataGen, N={n}', ax=axes[row, col])
        
        knn = KNeighborsClassifier(weights=knn_weights, n_neighbors=k)
        knn.fit(X, y)
        y_pred = knn.predict(X)
        show_data(X, y_pred, title=f'k={k}, N={n}', ax=axes[row+1, col],\
            model=knn)



fig.suptitle('Comparación de clasificadores con tamaños de muestras iguales')
current_size = fig.get_size_inches()
fig.set_size_inches(current_size[0]*1.5, current_size[1]*1.5)
plt.savefig('./figures/same_covs_and_same_priors_knn.png')


### Ahora variaremos k en knn (Datos no proporcionales)

fig, axes = plt.subplots(nrows = len(ks)+1, ncols = len(Ns_nonprop))

for row, k in enumerate(ks):
    for col, n in enumerate(Ns_nonprop):
        X, y = generate_data(diff_means, same_covs, N=n)

        if row == 0:
            show_data(X, y, title=f'DataGen, N={n}', ax=axes[row, col])
        
        knn = KNeighborsClassifier(weights=knn_weights, n_neighbors=k)
        knn.fit(X, y)
        y_pred = knn.predict(X)
        show_data(X, y_pred, title=f'k={k}, N={n}', ax=axes[row+1, col],\
            model=knn)

fig.suptitle('Comparación de clasificadores con diferentes tamaños de muestras')
current_size = fig.get_size_inches()
fig.set_size_inches(current_size[0]*1.5, current_size[1]*1.5)
plt.savefig('./figures/same_covs_and_distinct_priors_knn.png')
