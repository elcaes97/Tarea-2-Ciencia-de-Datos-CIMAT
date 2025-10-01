"""
Análisis de riesgo para clasificadores - Parte II del proyecto.
Calcula riesgos verdaderos y estimados para diferentes escenarios.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import os
import seaborn as sns

# Importar configuraciones y utilidades
from utils.config import *
from utils.distributions import generate_data
from utils.clasificadores import GaussianBayesClassifier

def calculate_bayes_risk(means, covs, priors, n_test=10000):
    """
    Calcula el riesgo verdadero de Bayes para distribuciones gaussianas.
    
    Args:
        means: Lista de medias de las clases
        covs: Lista de matrices de covarianza
        priors: Priors de las clases
        n_test: Número de puntos de test para Monte Carlo
        
    Returns:
        Riesgo de Bayes
    """
    # Generar datos de test
    X_test, y_test = generate_data(means, covs, [n_test // len(means)] * len(means))
    
    # Calcular probabilidades posteriores usando las distribuciones verdaderas
    posteriors = np.zeros((len(X_test), len(means)))
    
    for i in range(len(means)):
        # PDF multivariada para cada clase
        rv = multivariate_normal(means[i], covs[i])
        posteriors[:, i] = rv.pdf(X_test) * priors[i]
    
    # Normalizar posteriores
    posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
    
    # Predecir clase con mayor probabilidad posterior
    y_pred = np.argmax(posteriors, axis=1)
    
    # Calcular riesgo (tasa de error)
    risk = np.mean(y_pred != y_test)
    
    return risk

def riesgo_clasificador(model, X, y, cv=5, random_state=0):
    """
    Calcula el riesgo (tasa de error) de un clasificador usando validación cruzada estratificada.

    Parameters:
    -----------
    model : sklearn estimator
        Clasificador de scikit-learn.
    X : array-like
        Datos de entrenamiento.
    y : array-like
        Etiquetas objetivo.
    cv : int, default=5
        Número de folds en la validación cruzada.
    random_state : int, default=None
        Semilla para reproducibilidad.

    Returns:
    --------
    riesgo : tuple
        (riesgo promedio, desviación estándar del riesgo)
    """
    # Configurar validación cruzada estratificada
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Calcular precisión en cada fold
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    # El riesgo es 1 - precisión
    riesgos = 1 - scores

    return np.mean(riesgos), np.std(riesgos)

def run_single_risk_experiment(means, covs, sample_sizes, priors, scenario_name):
    """
    Ejecuta un experimento de riesgo completo para un escenario.
    
    Args:
        means: Medias de las distribuciones
        covs: Matrices de covarianza
        sample_sizes: Tamaños de muestra a evaluar
        priors: Priors de las clases
        scenario_name: Nombre del escenario para guardar resultados
        
    Returns:
        DataFrame con resultados
    """
    results = []
    classifiers = [
        ('Bayes', None),  # Riesgo verdadero de Bayes
        ('GNB', GaussianNB()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
        ('KNN-1', KNeighborsClassifier(n_neighbors=1)),
        ('KNN-5', KNeighborsClassifier(n_neighbors=5)),
        ('KNN-21', KNeighborsClassifier(n_neighbors=21)),
        ('GB', GaussianBayesClassifier(means, covs, priors))
    ]
    
    # Calcular riesgo de Bayes verdadero
    bayes_risk = calculate_bayes_risk(means, covs, priors)
    
    for n in sample_sizes:
        if isinstance(n, list):
            # Caso desbalanceado
            n_total = sum(n)
            n_str = f"{n[0]}+{n[1]}"
        else:
            # Caso balanceado
            n_total = n * len(means)
            n_str = str(n)
            
        for rep in range(REPS):
            # Generar datos
            X, y = generate_data(means, covs, n)
            
            for name, model in classifiers:
                if name == 'Bayes':
                    # Usar riesgo verdadero de Bayes
                    risk_mean, risk_std = bayes_risk, 0.0
                    risk_cv_mean, risk_cv_std = bayes_risk, 0.0
                else:
                    # Entrenar y calcular riesgo CV
                    risk_cv_mean, risk_cv_std = riesgo_clasificador(model, X, y)
                    
                    # Calcular riesgo en muestra de entrenamiento (como aproximación)
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    risk_mean = np.mean(y_pred != y)
                    risk_std = 0.0
                
                results.append({
                    'escenario': scenario_name,
                    'clasificador': name,
                    'n_muestra': n_str,
                    'n_total': n_total,
                    'replica': rep,
                    'riesgo_verdadero': risk_mean if name == 'Bayes' else np.nan,
                    'riesgo_cv_mean': risk_cv_mean,
                    'riesgo_cv_std': risk_cv_std,
                    'brecha_bayes': risk_cv_mean - bayes_risk
                })
    
    return pd.DataFrame(results)

def run_risk_experiments():
    """Ejecuta todos los experimentos de riesgo."""
    print("Ejecutando experimentos de análisis de riesgo...")
    
    # Crear directorios necesarios
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Escenario 1: Balanceado, misma covarianza
    print("  - Escenario 1: Balanceado, misma covarianza")
    df1 = run_single_risk_experiment(
        DIFF_MEANS, SAME_COVS, SAMPLE_SIZES_BALANCED, 
        [0.5, 0.5], "Balanceado_MismaCov"
    )
    
    # Escenario 2: Desbalanceado, misma covarianza
    print("  - Escenario 2: Desbalanceado, misma covarianza")
    df2 = run_single_risk_experiment(
        DIFF_MEANS, SAME_COVS, SAMPLE_SIZES_UNBALANCED,
        [0.2, 0.8], "Desbalanceado_MismaCov"
    )
    
    # Escenario 3: Balanceado, distinta covarianza
    print("  - Escenario 3: Balanceado, distinta covarianza")
    df3 = run_single_risk_experiment(
        DIFF_MEANS, DIFF_COVS, SAMPLE_SIZES_BALANCED,
        [0.5, 0.5], "Balanceado_DistintaCov"
    )
    
    # Escenario 4: Desbalanceado, distinta covarianza
    print("  - Escenario 4: Desbalanceado, distinta covarianza")
    df4 = run_single_risk_experiment(
        DIFF_MEANS, DIFF_COVS, SAMPLE_SIZES_UNBALANCED,
        [0.2, 0.8], "Desbalanceado_DistintaCov"
    )
    
    # Combinar todos los resultados
    all_results = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    # Guardar resultados
    all_results.to_csv('./results/risk_results.csv', index=False)
    
    # Generar gráficas
    create_risk_plots(all_results)
    
    print("  Experimentos de riesgo completados.")
    
    return all_results

def create_risk_plots(results_df):
    """Crea las gráficas de análisis de riesgo."""
    print("  Generando gráficas de riesgo...")
    
    # 1. L(g) vs n por método
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    scenarios = results_df['escenario'].unique()
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i//2, i%2]
        scenario_data = results_df[results_df['escenario'] == scenario]
        
        # Agrupar por clasificador y tamaño de muestra
        grouped = scenario_data.groupby(['clasificador', 'n_total'])['riesgo_cv_mean'].mean().reset_index()
        
        for classifier in grouped['clasificador'].unique():
            if classifier != 'Bayes':  # Bayes es constante
                class_data = grouped[grouped['clasificador'] == classifier]
                ax.plot(class_data['n_total'], class_data['riesgo_cv_mean'], 
                       'o-', label=classifier, markersize=6)
        
        # Línea de Bayes
        bayes_risk = scenario_data[scenario_data['clasificador'] == 'Bayes']['riesgo_verdadero'].iloc[0]
        ax.axhline(y=bayes_risk, color='red', linestyle='--', label='Bayes (óptimo)', linewidth=2)
        
        ax.set_xlabel('Tamaño de muestra total')
        ax.set_ylabel('Riesgo CV')
        ax.set_title(f'Riesgo vs Tamaño Muestral - {scenario}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/risk_vs_n.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. L(KNN) vs k
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i//2, i%2]
        scenario_data = results_df[results_df['escenario'] == scenario]
        
        # Filtrar solo KNN
        knn_data = scenario_data[scenario_data['clasificador'].str.startswith('KNN')]
        
        # Extraer valor de k
        knn_data = knn_data.copy()
        knn_data['k'] = knn_data['clasificador'].str.extract('KNN-(\d+)').astype(int)
        
        for n_size in knn_data['n_total'].unique():
            size_data = knn_data[knn_data['n_total'] == n_size]
            grouped = size_data.groupby('k')['riesgo_cv_mean'].mean()
            ax.plot(grouped.index, grouped.values, 'o-', label=f'n={n_size}', markersize=6)
        
        ax.set_xlabel('k (número de vecinos)')
        ax.set_ylabel('Riesgo CV')
        ax.set_title(f'KNN: Riesgo vs k - {scenario}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/knn_risk_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Brechas vs n
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i//2, i%2]
        scenario_data = results_df[results_df['escenario'] == scenario]
        
        # Excluir Bayes y agrupar
        non_bayes = scenario_data[scenario_data['clasificador'] != 'Bayes']
        grouped = non_bayes.groupby(['clasificador', 'n_total'])['brecha_bayes'].mean().reset_index()
        
        for classifier in grouped['clasificador'].unique():
            class_data = grouped[grouped['clasificador'] == classifier]
            ax.plot(class_data['n_total'], class_data['brecha_bayes'], 
                   'o-', label=classifier, markersize=6)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Tamaño de muestra total')
        ax.set_ylabel('Brecha vs Bayes (L(g) - L(Bayes))')
        ax.set_title(f'Brecha vs Bayes - {scenario}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/risk_gap_vs_n.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap de brechas
    create_risk_heatmaps(results_df)
    
    print("  Gráficas de riesgo generadas.")

def create_risk_heatmaps(results_df):
    """Crea heatmaps para análisis de brechas."""
    # Heatmap para brechas promedio por clasificador y escenario
    pivot_data = results_df[results_df['clasificador'] != 'Bayes'].groupby(
        ['clasificador', 'escenario']
    )['brecha_bayes'].mean().reset_index()
    
    heatmap_data = pivot_data.pivot(index='clasificador', columns='escenario', values='brecha_bayes')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
    plt.title('Brecha Promedio vs Bayes por Clasificador y Escenario')
    plt.tight_layout()
    plt.savefig('./figures/risk_gap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_risk_summary_tables():
    """Crea tablas resumen de los resultados de riesgo."""
    print("Generando tablas resumen de riesgo...")
    
    if not os.path.exists('./results/risk_results.csv'):
        print("No se encontraron resultados de riesgo. Ejecutando experimentos...")
        results_df = run_risk_experiments()
    else:
        results_df = pd.read_csv('./results/risk_results.csv')
    
    # Tabla resumen por escenario y clasificador
    summary = results_df.groupby(['escenario', 'clasificador', 'n_total']).agg({
        'riesgo_cv_mean': ['mean', 'std'],
        'brecha_bayes': ['mean', 'std']
    }).round(4)
    
    # Guardar tabla CSV
    summary.to_csv('./results/risk_summary.csv')
    
    # Crear tabla LaTeX
    latex_table = summary.to_latex()
    with open('./results/risk_summary.tex', 'w') as f:
        f.write(latex_table)
    
    # Tabla simplificada para reporte
    simple_summary = results_df.groupby(['escenario', 'clasificador']).agg({
        'riesgo_cv_mean': ['mean', 'std'],
        'brecha_bayes': ['mean', 'std']
    }).round(4)
    
    simple_summary.to_csv('./results/risk_simple_summary.csv')
    
    print("Tablas resumen guardadas en ./results/")
    
    return summary

def plot_validation_vs_true_risk():
    """
    Compara riesgo estimado por validación vs riesgo verdadero.
    Solo para escenarios donde podemos calcular riesgo verdadero fácilmente.
    """
    print("Generando comparación validación vs riesgo verdadero...")
    
    # Escenario simple: balanceado, misma covarianza
    means = DIFF_MEANS
    covs = SAME_COVS
    priors = [0.5, 0.5]
    
    # Calcular riesgo verdadero de Bayes
    true_bayes_risk = calculate_bayes_risk(means, covs, priors)
    
    results = []
    
    for n in SAMPLE_SIZES_BALANCED:
        for rep in range(REPS):
            X, y = generate_data(means, covs, n)
            
            # Clasificador Bayes con parámetros verdaderos
            gb_true = GaussianBayesClassifier(means, covs, priors)
            gb_true.fit(X, y)
            
            # Riesgo en training (aproximación)
            y_pred = gb_true.predict(X)
            train_risk = np.mean(y_pred != y)
            
            # Riesgo por validación cruzada
            cv_risk, cv_std = riesgo_clasificador(gb_true, X, y)
            
            results.append({
                'n': n * len(means),  # Total samples
                'replica': rep,
                'true_bayes_risk': true_bayes_risk,
                'train_risk': train_risk,
                'cv_risk': cv_risk,
                'cv_std': cv_std
            })
    
    results_df = pd.DataFrame(results)
    
    # Gráfica de comparación
    plt.figure(figsize=(10, 6))
    
    grouped = results_df.groupby('n').agg({
        'true_bayes_risk': 'mean',
        'train_risk': 'mean',
        'cv_risk': 'mean'
    }).reset_index()
    
    plt.plot(grouped['n'], grouped['true_bayes_risk'], 'ro-', label='Riesgo Verdadero Bayes', linewidth=2)
    plt.plot(grouped['n'], grouped['train_risk'], 'bs-', label='Riesgo Entrenamiento', linewidth=2)
    plt.plot(grouped['n'], grouped['cv_risk'], 'g^-', label='Riesgo CV', linewidth=2)
    
    plt.xlabel('Tamaño de Muestra Total')
    plt.ylabel('Riesgo')
    plt.title('Comparación: Riesgo Verdadero vs Estimado (Bayes con Parámetros Verdaderos)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./figures/validation_vs_true_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_df