import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from scripts.utils.cross_validation import (
    cross_validate_gam_grid, 
    retrain_best_gam, 
    null_model_nll, 
    poisson_nll_per_sample, 
    pseudo_r2_mcfadden, 
    select_best_1se_gam
)

def train_and_evaluate_shapley_subsets(X, Y, folds, X_pool, Y_pool, X_held, Y_held, splines_grid, lambdas_grid, features=None):
    """
    Entrena y evalúa todos los subconjuntos posibles de variables indicadas en features.
    
    Args:
        X: matriz de diseño de (N, len(features) o más). Las columnas deben coincidir con la lógica 
           de cross_validation.py (columnas 0,1=pos, 2=hd, 3=view, 4=dist)
        Y: vector de espigas
        folds: folds de validación cruzada para X e Y
        X_pool, Y_pool: partición para reentrenamiento del modelo final
        X_held, Y_held: partición held out para calcular el R2 final
        splines_grid: opciones de cantidad de splines
        lambdas_grid: opciones de lambda (regularización)
        features: Lista de nombres de características a usar, ej: ['pos', 'hd', 'view', 'dist']
        
    Returns:
        Diccionario con el pseudo-R2 de cada submodelo.
    """
    if features is None:
        features = ['pos', 'hd', 'view', 'dist']
        
    # Generar todos los subconjuntos posibles (2^N - 1)
    model_types = []
    for i in range(1, len(features) + 1):
        for combo in itertools.combinations(features, i):
            model_types.append('shapley_' + '_'.join(combo))

    nll_nulo = null_model_nll(Y_pool, Y_held)
    
    results = {}
    results['null'] = 0.0 
    
    for mt in model_types:
        print(f"\n" + "="*40)
        print(f"Evaluando sub-modelo: {mt}")
        print("="*40)
        
        _, res_cv = cross_validate_gam_grid(X, Y, folds, splines_grid, lambdas_grid, model_type=mt)
        _, mejor_1se, _ = select_best_1se_gam(res_cv)
        best_sp, best_lam = int(mejor_1se[0]), mejor_1se[1]
        
        # Re-entrenamiento en train_pool usando los mejores hiperparámetros de la CV
        gam_final = retrain_best_gam(X_pool, Y_pool, best_sp, best_lam, model_type=mt)
        
        # Evaluar sobre el held-out real
        mu_gam = gam_final.predict(X_held)
        nll_gam = poisson_nll_per_sample(Y_held, mu_gam)
        r2 = pseudo_r2_mcfadden(nll_gam, nll_nulo)
        
        results[mt] = r2
        print(f"[*] Pseudo R² (McFadden) para {mt}: {r2:.6f}")
        
    return results

def calculate_shapley_values(r2_results, features=None):
    """
    Calcula la contribución de Shapley para las variables provistas de manera genérica.
    
    Args:
        r2_results: diccionario devuelto por train_and_evaluate_shapley_subsets
        features: lista de variables (ej: ['pos', 'hd', 'view', 'dist'])
        
    Returns:
        Diccionario con el Shapley Value (phi) absoluto de cada variable.
    """
    if features is None:
        features = ['pos', 'hd', 'view', 'dist']
        
    V = len(features)
    phi = {f: 0.0 for f in features}
    
    def weight(S_size):
        return (math.factorial(S_size) * math.factorial(V - S_size - 1)) / math.factorial(V)
        
    for f in features:
        other_features = [x for x in features if x != f]
        for S_size in range(V):
            for S in itertools.combinations(other_features, S_size):
                if S_size == 0:
                    model_S = 'null'
                else:
                    # Ordenar S para que coincida con el orden en features
                    S_sorted = [x for x in features if x in S]
                    model_S = 'shapley_' + '_'.join(S_sorted)
                    
                S_f = list(S) + [f]
                S_f_sorted = [x for x in features if x in S_f]
                model_S_f = 'shapley_' + '_'.join(S_f_sorted)
                
                r2_with = r2_results.get(model_S_f, 0.0)
                r2_without = r2_results.get(model_S, 0.0)
                
                phi[f] += weight(S_size) * (r2_with - r2_without)
                
    return phi

def plot_shapley_values(phi, total_r2=None):
    """
    Grafica el porcentaje de contribución de cada variable a la explicación
    del modelo predictivo (R2), adaptado para cantidad de variables.
    """
    labels = list(phi.keys())
    values = list(phi.values())
    
    total_phi = sum(values)
    if total_phi > 0:
        percentages = [v / total_phi * 100 for v in values]
    else:
        percentages = [0]*len(values)
        
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Paleta de colores más general
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, percentages, color=colors)
    
    ax.set_xlabel('Shapley Value (%)', fontsize=12)
    ax.set_title('Relative Contribution of Behavioral Variables', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, max(percentages) * 1.2 if max(percentages) > 0 else 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),  
                    textcoords="offset points",
                    ha='left', va='center', fontsize=11)
                    
    if total_r2 is not None:
        text_str = f'R² Full Model = {total_r2:.4f}\n(Efficiency: Shapley Sum = {total_phi:.4f})'
        ax.text(0.95, 0.95, text_str,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))
                
    plt.tight_layout()
    plt.show()
