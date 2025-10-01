"""
Script principal para experimentos de comparación de clasificadores.
Genera comparaciones exhaustivas entre diferentes algoritmos de clasificación.
"""
from utils import *
from utils.risk_analysis import run_complete_analysis

def main():
    """Función principal que ejecuta todos los experimentos."""
    print("Iniciando experimentos de clasificación...")
    setup_environment()
    
    # Experimento 1: Clases balanceadas con misma covarianza
    print("Ejecutando experimento 1: Clases balanceadas, misma cov...")
    create_comparison_figure(
        DIFF_MEANS, SAME_COVS, 
        SAMPLE_SIZES_BALANCED,
        "Clases Balanceadas con Misma Covarianza",
        "balanced_same_cov"
    )
    
    # Experimento 2: Clases desbalanceadas con misma covarianza  
    print("Ejecutando experimento 2: Clases desbalanceadas, misma cov...")
    create_comparison_figure(
        DIFF_MEANS, SAME_COVS,
        SAMPLE_SIZES_UNBALANCED,
        "Clases Desbalanceadas con Misma Covarianza", 
        "unbalanced_same_cov"
    )
    
    # Experimento 3: Estudio de K en KNN (clases balanceadas)
    print("Ejecutando experimento 3: Estudio de K (balanceado)...")
    create_knn_study_figure(
        DIFF_MEANS, SAME_COVS,
        SAMPLE_SIZES_BALANCED,
        "Clases Balanceadas",
        "balanced"
    )
    
    # Experimento 4: Estudio de K en KNN (clases desbalanceadas)
    print("Ejecutando experimento 4: Estudio de K (desbalanceado)...")
    create_knn_study_figure(
        DIFF_MEANS, SAME_COVS, 
        SAMPLE_SIZES_UNBALANCED,
        "Clases Desbalanceadas",
        "unbalanced"
    )

    # Experimento 5: Clases balanceadas con distinta covarianza
    print("Ejecutando experimento 5: Clases balanceadas, distinta cov...")
    create_comparison_figure(
        DIFF_MEANS, DIFF_COVS, 
        SAMPLE_SIZES_BALANCED,
        "Clases Balanceadas con Misma Covarianza",
        "balanced_distinct_cov"
    )
    
    # Experimento 6: Clases desbalanceadas con distinta covarianza  
    print("Ejecutando experimento 6: Clases desbalanceadas, distinta cov...")
    create_comparison_figure(
        DIFF_MEANS, DIFF_COVS,
        SAMPLE_SIZES_UNBALANCED,
        "Clases Desbalanceadas con Misma Covarianza", 
        "unbalanced_distinct_cov"
    )
    
    print("Todos los experimentos completados. Figuras guardadas en ./figures/")


    results, bayes_risk = run_complete_analysis()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("Se han generado las siguientes gráficas:")
    print("1. risk_vs_samplesize.png - L(g) vs n")
    print("2. knn_risk_vs_k.png - L(k-NN) vs k") 
    print("3. risk_gaps.png - Brechas L(g)-L(Bayes)")
    print("4. risk_gap_heatmap.png - Heatmap de brechas")
    print("5. validation_comparison.png - Comparación métodos validación")
    print("\nLos resultados se han guardado en el objeto 'results'")

if __name__ == "__main__":
    main()