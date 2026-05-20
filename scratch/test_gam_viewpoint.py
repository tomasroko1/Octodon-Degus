import sys
import os
import numpy as np
import time

# Agregar el directorio 'scripts' al path para poder importar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import preparar_datos_mirada
from scripts.utils.cross_validation import (
    generate_all_splits,
    cross_validate_gam_grid,
    cross_validate_glm_grid,
    retrain_best_gam,
    retrain_best_glm,
    predict_glm_on_new_data,
    null_model_nll,
    poisson_nll_per_sample,
    pseudo_r2_mcfadden
)
from pygam import PoissonGAM, s

sesion = 2
tetrodo = 2
neurona = 3
bin_size = 0.1  # Usar 0.1s para concordancia con Sección 7

print("=== 1. CARGANDO DATOS DE MIRADA ===")
x_bins, y_bins, ang_bins_rad, spikes_gaze = preparar_datos_mirada(
    sesion, tetrodo, neurona, bin_size_sec=bin_size
)
print(f"Total de bines cargados: {len(x_bins)}")

# Usar la coordenada óptima encontrada en Sección 6 (59.7, 3.8)
x_best_p, y_best_p = 59.7, 3.8
print(f"Punto de perspectiva óptimo: ({x_best_p}, {y_best_p}) cm")

# Corregir la dirección de cabeza por perspectiva
dy = y_best_p - y_bins
dx = x_best_p - x_bins
ang_hacia_punto = np.arctan2(dy, dx)
theta_corrected = np.mod(ang_bins_rad - ang_hacia_punto, 2 * np.pi)

# Crear matriz de entrada con 3 columnas
X_cv = np.column_stack((x_bins, y_bins, theta_corrected))
Y_cv = spikes_gaze

print("=== 2. GENERANDO PARTICIONES DE CROSS-VALIDACIÓN ===")
folds, held_out_idx, train_pool_idx, roles = generate_all_splits(
    n_muestras=len(X_cv),
    bin_size_sec=bin_size,
    block_size_sec=60,
    n_folds=5,
    buffer_sec=2
)

X_pool, Y_pool = X_cv[train_pool_idx], Y_cv[train_pool_idx]
X_held, Y_held = X_cv[held_out_idx], Y_cv[held_out_idx]

print(f"Train pool: {len(X_pool)} muestras")
print(f"Held-out:   {len(X_held)} muestras")

# Para que el test sea rápido, probamos una grilla pequeña de hiperparámetros
print("\n=== 3. PROBANDO GRID SEARCH GAM (ESPACIO + VIEWPOINT) ===")
splines_a_probar = [4, 5]
lambdas_a_probar = [0.1, 1.0]

start_time = time.time()
error_matrix_gam, resultados_gam = cross_validate_gam_grid(
    X_cv, Y_cv, folds, splines_a_probar, lambdas_a_probar
)
print(f"Grid search GAM completado en {time.time() - start_time:.2f} segundos.")

mejor_gam_cv = sorted(resultados_gam, key=lambda x: x[3])[0]
best_sp, best_lam_gam = int(mejor_gam_cv[0]), mejor_gam_cv[1]
print(f"Mejor GAM de CV: splines={best_sp}, lam={best_lam_gam:.6f}")

print("\n=== 4. PROBANDO GRID SEARCH GLM (ESPACIAL) ===")
bines_a_probar = [4, 5]
alphas_a_probar = [0.01, 0.1]

start_time = time.time()
error_matrix_glm, resultados_glm = cross_validate_glm_grid(
    X_cv, Y_cv, folds, bines_a_probar, alphas_a_probar
)
print(f"Grid search GLM completado en {time.time() - start_time:.2f} segundos.")

mejor_glm_cv = sorted(resultados_glm, key=lambda x: x[2])[0]
best_bines, best_alpha = int(mejor_glm_cv[0]), mejor_glm_cv[1]
print(f"Mejor GLM de CV: bines={best_bines}x{best_bines}, alpha={best_alpha:.6f}")

print("\n=== 5. ENTRENANDO Y EVALUANDO MODELOS FINALES ===")
# 1. Re-entrenar GAM Posición + Viewpoint (3 columnas)
gam_joint_final = retrain_best_gam(X_pool, Y_pool, best_sp, best_lam_gam)
print(f"GAM Posición + Viewpoint final entrenado.")

# 2. Re-entrenar GAM Posición Puro (usando solo las primeras 2 columnas)
gam_spatial_final = retrain_best_gam(X_pool[:, :2], Y_pool, best_sp, best_lam_gam)
print(f"GAM Posición Puro final entrenado.")

# 3. Re-entrenar GAM Viewpoint Puro (usando solo la tercera columna, con spline circular s(0, basis='cp'))
gam_viewpoint_final = PoissonGAM(s(0, basis='cp', n_splines=8, lam=best_lam_gam, edge_knots=[0.0, 2*np.pi])).fit(X_pool[:, 2:3], Y_pool)
print(f"GAM Viewpoint Puro final entrenado.")

# 4. Re-entrenar GLM Posición (usando las primeras 2 columnas)
glm_final, cx, cy, sigma = retrain_best_glm(X_pool[:, :2], Y_pool, best_bines, best_alpha)
print(f"GLM Posición final entrenado.")

# Evaluación en held-out
nll_nulo = null_model_nll(Y_pool, Y_held)

# Predicciones y NLL
mu_gam_joint = gam_joint_final.predict(X_held)
nll_gam_joint = poisson_nll_per_sample(Y_held, mu_gam_joint)
r2_gam_joint  = pseudo_r2_mcfadden(nll_gam_joint, nll_nulo)

mu_gam_spatial = gam_spatial_final.predict(X_held[:, :2])
nll_gam_spatial = poisson_nll_per_sample(Y_held, mu_gam_spatial)
r2_gam_spatial  = pseudo_r2_mcfadden(nll_gam_spatial, nll_nulo)

mu_gam_viewpoint = gam_viewpoint_final.predict(X_held[:, 2:3])
nll_gam_viewpoint = poisson_nll_per_sample(Y_held, mu_gam_viewpoint)
r2_gam_viewpoint  = pseudo_r2_mcfadden(nll_gam_viewpoint, nll_nulo)

mu_glm = predict_glm_on_new_data(glm_final, X_held[:, :2], cx, cy, sigma)
nll_glm = poisson_nll_per_sample(Y_held, mu_glm)
r2_glm  = pseudo_r2_mcfadden(nll_glm, nll_nulo)

print(f"\n{'='*102}")
print(f"RESULTADOS FINALES EN HELD-OUT")
print(f"{'='*102}")
print(f"{'Métrica':<25} {'Modelo Nulo':>14} {'GAM Posición':>14} {'GAM Viewpoint':>14} {'GAM Pos+View':>14} {'GLM Posición':>14}")
print(f"{'-'*102}")
print(f"{'NLL (held-out)':<25} {nll_nulo:>14.8f} {nll_gam_spatial:>14.8f} {nll_gam_viewpoint:>14.8f} {nll_gam_joint:>14.8f} {nll_glm:>14.8f}")
print(f"{'Pseudo R² (McFadden)':<25} {'---':>14} {r2_gam_spatial:>14.6f} {r2_gam_viewpoint:>14.6f} {r2_gam_joint:>14.6f} {r2_glm:>14.6f}")
print(f"{'-'*102}")
print(f"Pseudo R² GAM Posición:  {r2_gam_spatial:.4f} ({r2_gam_spatial*100:.2f}%)")
print(f"Pseudo R² GAM Viewpoint: {r2_gam_viewpoint:.4f} ({r2_gam_viewpoint*100:.2f}%)")
print(f"Pseudo R² GAM Pos+View:  {r2_gam_joint:.4f} ({r2_gam_joint*100:.2f}%)")
print(f"Pseudo R² GLM Posición:  {r2_glm:.4f} ({r2_glm*100:.2f}%)")
print("="*102)
