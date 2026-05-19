import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from pygam import PoissonGAM, te
import sys
import os

# Asegurar que se puede importar de módulos hermanos (utils, etc.)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import preparar_datos_posicion
from play_cv import get_interleaved_folds_with_buffer

def evaluar_lambda(X, Y, folds, n_splines, lam):
    """
    Evalúa un modelo GAM con un lambda específico usando validación cruzada.
    Retorna el Negative Log-Likelihood (NLL) promedio en los folds de test.
    """
    errores_test_cv = []
    
    for train_idx, test_idx in folds:
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]
        
        # Entrenar modelo
        modelo = PoissonGAM(te(0, 1, n_splines=n_splines, lam=lam)).fit(X_train, Y_train)
        
        # Calcular NLL en el fold de test
        nll_val = -modelo.loglikelihood(X_test, Y_test) / len(Y_test)
        errores_test_cv.append(nll_val)
        
    return np.mean(errores_test_cv)

def optimizar_lambda_spline(X, Y, folds, n_splines, lambdas_iniciales):
    """
    Optimiza el parámetro lambda utilizando interpolación con CubicSpline
    sobre la escala log10 del lambda.
    """
    print(f"Evaluando {len(lambdas_iniciales)} valores iniciales de lambda...")
    
    log_lambdas = np.log10(lambdas_iniciales)
    errores = []
    
    for lam in lambdas_iniciales:
        error = evaluar_lambda(X, Y, folds, n_splines, lam)
        errores.append(error)
        print(f"  lambda = {lam:10.5f} (log10 = {np.log10(lam):6.3f}) -> NLL = {error:.6f}")
        
    # Ordenar por log_lambda para la interpolación (CubicSpline lo requiere estrictamente ordenado)
    sort_idx = np.argsort(log_lambdas)
    log_lambdas_sorted = log_lambdas[sort_idx]
    errores_sorted = np.array(errores)[sort_idx]
    
    # Ajustar un Cubic Spline (Interpolación)
    spline = CubicSpline(log_lambdas_sorted, errores_sorted)
    
    # Buscar el mínimo de la función interpolada en una grilla súper fina
    # Como evaluar el spline es computacionalmente "gratis", podemos usar 10,000 puntos
    x_fine = np.linspace(log_lambdas_sorted.min(), log_lambdas_sorted.max(), 10000)
    y_fine = spline(x_fine)
    
    min_idx = np.argmin(y_fine)
    best_log_lambda = x_fine[min_idx]
    min_error = y_fine[min_idx]
    best_lambda = 10 ** best_log_lambda
    
    print(f"\n¡Mínimo encontrado analíticamente en la curva interpolada!")
    print(f"  log10(lambda) ideal = {best_log_lambda:.5f}")
    print(f"  lambda ideal        = {best_lambda:.5f}")
    print(f"  NLL estimado        = {min_error:.6f}")
    
    # === GRAFICAR ===
    plt.figure(figsize=(10, 6))
    
    # Curva interpolada
    plt.plot(x_fine, y_fine, color='#1f77b4', alpha=0.8, linewidth=2.5, label='Cubic Spline (Interpolación)')
    
    # Puntos reales evaluados
    plt.scatter(log_lambdas_sorted, errores_sorted, color='#d62728', s=120, zorder=5, 
                edgecolor='white', linewidth=1.5, label='Evaluaciones Reales (Fuerza Bruta reducida)')
    
    # Punto mínimo
    plt.scatter([best_log_lambda], [min_error], color='#2ca02c', marker='*', s=400, zorder=6, 
                edgecolor='black', linewidth=1, label=rf'Mínimo Óptimo ($\lambda \approx {best_lambda:.2f}$)')
    
    plt.title(f'Optimización de Lambda mediante Interpolación (n_splines = {n_splines})', fontsize=14, fontweight='bold')
    plt.xlabel(r'$\log_{10}(\lambda)$', fontsize=13)
    plt.ylabel('Error de Validación Cruzada (NLL)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Añadir texto explicativo en el gráfico
    textstr = f'Lambda óptimo: {best_lambda:.3f}\nLog10: {best_log_lambda:.3f}\nNLL Estimado: {min_error:.5f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    plt.tight_layout()
    plt.show()
    
    return best_lambda, best_log_lambda, min_error

def main():
    print("=== Optimización Inteligente de Lambda usando Interpolación (Splines) ===")
    print("1. Cargando datos...")
    sesion, tetrodo, neurona = 2, 3, 3
    bin_size = 0.1
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size)

    print("\n2. Preparando validación cruzada (Interleaved K-Fold con Buffer de 2s)...")
    folds = get_interleaved_folds_with_buffer(
        n_muestras=len(X), 
        bin_size_sec=bin_size, 
        block_size_sec=60, 
        n_folds=5,
        buffer_sec=2
    )

    # Probemos con 5 splines para el modelo
    n_splines = 6
    
    # Solo 7 valores distribuidos en escala logarítmica (ahorramos MUCHISIMO tiempo comparado con una grilla de 100 puntos)
    # Rango desde 0.01 hasta 100.
    # log10: -2, -1.3, -0.6, 0, 0.7, 1.4, 2
    lambdas_iniciales = [0.0001,0.001,0.002,0.003,0.007,0.008,0.009, 3]
    
    print("\n3. Iniciando evaluación e interpolación...")
    optimizar_lambda_spline(X, Y, folds, n_splines, lambdas_iniciales)

if __name__ == "__main__":
    main()
