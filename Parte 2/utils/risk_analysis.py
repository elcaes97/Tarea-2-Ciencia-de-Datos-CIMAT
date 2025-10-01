import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from utils.distributions import generate_data
from utils.clasificadores import GaussianBayesClassifier
from utils.config import DIFF_MEANS, SAME_COVS, SAMPLE_SIZES_BALANCED, KS


# Configurations
SAMPLE_SIZES = SAMPLE_SIZES_BALANCED
N_MONTE_CARLO = 50  # Número de simulaciones Monte Carlo
N_FOLDS = 20  # Número de folds para cross-validation


def compute_bayes_risk(priors, means, sigma):
    """Compute theoretical Bayes risk for equal covariance case (two classes)"""
    pi0, pi1 = priors
    mu1, mu0 = means

    inv = np.linalg.pinv(sigma) if np.linalg.det(sigma) == 0\
        else np.linalg.inv(sigma)

    delta = np.sqrt((mu1 - mu0).T @ inv @ (mu1 - mu0))
    
    if pi0 == pi1:  # Equal priors case
        return norm.cdf(-delta/2)
    else: # General case with different priors
        term1 = pi0 * norm.cdf(-delta/2 + np.log(pi1/pi0)/delta)
        term2 = pi1 * norm.cdf(-delta/2 - np.log(pi1/pi0)/delta)
        return term1 + term2

# Evaluation functions
def empirical_risk(clf, X_test, y_test):
    """Compute empirical risk (error rate)"""
    y_pred = clf.predict(X_test)
    return np.mean(y_pred != y_test)

def monte_carlo_estimation(clf_class, clf_params, distribution_params, n_samples, n_simulations=50):
    """Estimate risk using Monte Carlo simulation"""
    risks = []
    
    for i in range(n_simulations):
        # Generate new data for each simulation
        X_train, y_train = generate_data(
            means=distribution_params['means'], 
            covs=distribution_params['covs'], 
            n_samples=n_samples
        )
        X_test, y_test = generate_data(
            means=distribution_params['means'], 
            covs=distribution_params['covs'], 
            n_samples=1000  # Large test set
        )
        
        # Train and evaluate
        clf = clf_class(**clf_params) if clf_params else clf_class()
        clf.fit(X_train, y_train)
        risk = empirical_risk(clf, X_test, y_test)
        risks.append(risk)
    
    return np.mean(risks), np.std(risks)

def compare_validation_methods(clf_class, distribution_params, n_samples, n_simulations=20):
    """Compare cross-validation vs Monte Carlo estimates - CORREGIDO"""
    cv_risks, mc_risks = [], []
    
    for i in range(n_simulations):
        X, y = generate_data(
            means=distribution_params['means'], 
            covs=distribution_params['covs'], 
            n_samples=n_samples
        )
        
        # Cross-validation estimate
        cv_scores = cross_val_score(
            clf_class(), X, y, 
            cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
        )
        l_cv = 1 - np.mean(cv_scores)
        cv_risks.append(l_cv)
    
        
        # Monte Carlo estimate
        l_mc, _ = monte_carlo_estimation(
            clf_class, {}, distribution_params, n_samples, n_simulations=1
        )
        mc_risks.append(l_mc)
    
    return np.mean(cv_risks), np.std(cv_risks), np.mean(mc_risks), np.std(mc_risks)


# función para correr el analisis completo (llamar en el main.py)
def run_complete_analysis():
    """Ejecuta todo el análisis completo"""

    models = {
        'GaussianNB': GaussianNB,
        'LDA': LinearDiscriminantAnalysis,
        'QDA': QuadraticDiscriminantAnalysis,
        'KNN': KNeighborsClassifier
    }
    
    # Inicializar almacenamiento de resultados
    results = {
        'sample_sizes': SAMPLE_SIZES,
        'k_values': KS,
        'models': models.keys(),
        'risks': {model: [] for model in models.keys()},
        'risks_std': {model: [] for model in models.keys()},
        'knn_curves': {n: [] for n in SAMPLE_SIZES},
        'risk_gaps': {model: [] for model in models.keys()},
        'validation_comparison': {model: {} for model in models.keys()}
    }
    
    # Calcular riesgo de Bayes teórico
    sigma = SAME_COVS[0]
    print(f"Parámetros para Bayes risk:")
    print(f"mu0: {DIFF_MEANS[0]}")
    print(f"mu1: {DIFF_MEANS[1]}")
    print(f"sigma:\n{sigma}")
    bayes_risk = compute_bayes_risk([0.5, 0.5], DIFF_MEANS, sigma)
    print(f"Bayes risk calculado: {bayes_risk:.4f}")

    
    print("=" * 60)
    print("COMIENZO DEL ANÁLISIS DE RIESGO COMPLETO")
    print("=" * 60)
    
    # Definir parámetros de distribución
    dist_params = {'means': DIFF_MEANS,'covs': SAME_COVS}
    
    # L(g) vs n para cada modelo
    print("\n1. Calculando L(g) vs n para cada modelo...")
    
    for i, n in enumerate(SAMPLE_SIZES):
        print(f"\tProcesando n={n} ({i+1}/{len(SAMPLE_SIZES)})")
        
        for model_name in results['models']:
            risk_mean, risk_std = monte_carlo_estimation(
                models[model_name], {}, dist_params, n, N_MONTE_CARLO
            )
            
            results['risks'][model_name].append(risk_mean)
            results['risks_std'][model_name].append(risk_std)
            results['risk_gaps'][model_name].append(risk_mean - bayes_risk)
            
            print(f"\t\t{model_name}: {risk_mean:.4f} +/- {risk_std:.4f}")
          

    # L(k-NN) vs k curves para diferentes n
    print("\n2. Calculando curvas L(k-NN) vs k...")
    
    for i, n in enumerate(SAMPLE_SIZES):
        print(f"\tProcesando k-NN para n={n} ({i+1}/{len(SAMPLE_SIZES)})")
        knn_risks = []
        
        for k in KS:
            risk_mean, _ = monte_carlo_estimation(
                KNeighborsClassifier, {'n_neighbors': k}, dist_params, 
                n, n_simulations=10  # Menos simulaciones para mayor velocidad
            )
            knn_risks.append(risk_mean)
            print(f"\t\tk={k}: {risk_mean:.4f}")
        
        results['knn_curves'][n] = knn_risks
    
    
    # Comparación Validation vs Monte Carlo
    print("\n3. Comparando Validation vs Monte Carlo...")
    
    n_comparison = SAMPLE_SIZES[2]  # Usar un tamaño de muestra intermedio
    print(f"\tUsando n={n_comparison} para comparación")
    
    for model_name in results['models']:
        clf_class = models[model_name]
        
        cv_mean, cv_std, mc_mean, mc_std = compare_validation_methods(
            clf_class, dist_params, n_comparison, 5  # Reducido para pruebas
        )
        
        results['validation_comparison'][model_name] = {
            'cv_mean': cv_mean, 'cv_std': cv_std,
            'mc_mean': mc_mean, 'mc_std': mc_std
        }
        
        print(f"\t\t{model_name}: CV={cv_mean:.4f}, MC={mc_mean:.4f}")
    
    # generar graficas y mostrar tablas
    print("\n4. Generando gráficas y tablas...")
    generate_all_plots(results, bayes_risk)
    print_summary_table(results, bayes_risk)
    
    return results, bayes_risk


# funciones de visualizacion
def generate_all_plots(results, bayes_risk):
    """Genera todas las gráficas solicitadas"""
    
    # 4.1 L(g) vs n para cada modelo
    plt.figure(figsize=(12, 8))
    for model_name in results['models']:
        plt.plot(results['sample_sizes'], results['risks'][model_name], 
                label=model_name, linewidth=2, marker='o')
    
    plt.axhline(y=bayes_risk, color='red', linestyle='--', 
                label=f'Bayes Risk ({bayes_risk:.3f})', linewidth=2)
    plt.xlabel('Sample Size (n)') ; plt.ylabel('Theoretical Risk L(g)')
    plt.title('Model Risk vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./figures/risk_vs_samplesize.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4.2 L(k-NN) vs k curves
    plt.figure(figsize=(12, 8))
    for n in results['sample_sizes']:
        if n in results['knn_curves'] and len(results['knn_curves'][n]) > 0:
            plt.plot(results['k_values'], results['knn_curves'][n], 
                    label=f'n={n}', marker='s', linewidth=2)
    
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Risk L(k-NN)')
    plt.title('k-NN Risk vs k for Different Sample Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./figures/knn_risk_vs_k.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4.3 Risk gaps L(g)-L(Bayes) vs n
    plt.figure(figsize=(12, 8))
    for model_name in results['models']:
        plt.plot(results['sample_sizes'], results['risk_gaps'][model_name], 
                label=model_name, marker='^', linewidth=2)
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Risk Gap L(g) - L(Bayes)')
    plt.title('Risk Gap to Bayes Optimal vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./figures/risk_gaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4.4 Heatmap de brechas 
    plt.figure(figsize=(10, 6))
    gap_matrix = np.array([results['risk_gaps'][model]\
        for model in results['models']])
    sns.heatmap(gap_matrix, cmap='RdBu_r', annot=True, fmt=".3f", xticklabels\
        =results['sample_sizes'], yticklabels=results['models'])
    plt.xlabel('Sample Size (n)') ; plt.ylabel('Model')
    plt.title('Risk Gap Heatmap: L(g) - L(Bayes)')
    plt.savefig('./figures/risk_gap_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    # 4.5 Comparación Validation vs Monte Carlo
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax = ax.ravel()
    
    for idx, model_name in enumerate(results['models']):
        if model_name in results['validation_comparison']:
            comp = results['validation_comparison'][model_name]
            methods = ['CV', 'Monte Carlo']
            means = [comp['cv_mean'], comp['mc_mean']]
            stds = [comp['cv_std'], comp['mc_std']]
            
            bars = ax[idx].bar(methods, means, yerr=stds, capsize=5, 
                              alpha=0.7, color=['skyblue', 'lightcoral'])
            ax[idx].set_title(f'{model_name}: Validation vs Monte Carlo')
            ax[idx].set_ylabel('Risk')
            
            # Añadir valores en las barras
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{mean:.3f}', ha='center', va='bottom')
    
    plt.savefig('./figures/validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_table(results, bayes_risk):
    """Imprime la tabla resumen de resultados"""
    print("\n" + "="*80)
    print("TABLA RESUMEN: Riesgos Promedio y Desviaciones Estándar")
    print("="*80)
    print(f"{'Modelo':<12} {'Riesgo Promedio':<15} {'Desv Estándar':<15} {'Gap vs Bayes':<15}")
    print("-"*80)
    
    for model_name in results['models']:
        mean_risk = np.mean(results['risks'][model_name])
        std_risk = np.mean(results['risks_std'][model_name])
        mean_gap = np.mean(results['risk_gaps'][model_name])
        
        print(f"{model_name:<12} {mean_risk:<15.4f} {std_risk:<15.4f} {mean_gap:<15.4f}")
    
    print("-"*80)
    print(f"{'BAYES':<12} {bayes_risk:<15.4f} {'-':<15} {'0':<15}")
    print("="*80)
    
    # Tabla de comparación validation vs Monte Carlo
    print("\n" + "="*60)
    print("COMPARACIÓN: Cross-Validation vs Monte Carlo")
    print("="*60)
    print(f"{'Modelo':<12} {'CV Risk':<10} {'CV Std':<10} {'MC Risk':<10} {'MC Std':<10}")
    print("-"*60)
    
    for model_name in results['models']:
        if model_name in results['validation_comparison']:
            comp = results['validation_comparison'][model_name]
            print(f"{model_name:<12} {comp['cv_mean']:<10.4f} {comp['cv_std']:<10.4f} "
                  f"{comp['mc_mean']:<10.4f} {comp['mc_std']:<10.4f}")
    
    print("="*60)
