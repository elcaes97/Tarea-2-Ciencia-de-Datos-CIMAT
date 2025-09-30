import numpy as np

points_size = 0.3

knn_weights = 'distance'

# medias
diff_means = [
    np.array([-1,0]),
    np.array([0,1])
]

same_means = [
    np.zeros(2),
    np.zeros(2)
]

separated_means = [
    np.array([-2,0]),
    np.array([2,0])
]


## Barridos de parametros
Ns_prop = [50, 100, 200, 500]    # Tamaños muestrales
ks = [1, 3, 5, 11, 21]      # Para k-NN

## Replicacion
Reps = 20   # Num de replicas independientes, promediar media \pm desv estandar de L

## Escenarios

### \Sigma_0 = \Sigma_1 (LDA optimo)
same_covs = [   # Multiplicar por constante para escalar
    np.array([
        [1,0],
        [0,1]
    ]),
    np.array([
        [1,0],
        [0,1]
    ])
]

low_covs = [
    np.array([
        [0.1,0],
        [0,0.1]
    ]),
    np.array([
        [0.1,0],
        [0,0.1]
    ])
]


### \Sigma_0 \neq \Sigma_1 (QDA optimo)
diff_covs = [
    np.array([
        [1,0],
        [0,2]
    ]),
    np.array([
        [2,0],
        [0,1]
    ])
]

### Desvalance de clases \pi_0=0.8 y \pi_1=0.2

Ns_nonprop = [[50, 200], [100, 400], [200, 800], [500, 2000]]    # Tamaños muestrales


### Correlaciones fuertes y/o malcondicionamiento en \Sigma_K
