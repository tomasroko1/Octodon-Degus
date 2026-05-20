import json
notebook_path = 'playground.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modificar Celda 23 (CV de los modelos con grillas independientes)
nb['cells'][23]['source'] = [
    '# hiperparametros GAM. Aprox. ~2-3 min en optimizar todos los modelos independientemente\n',
    'splines_a_probar = [4, 5, 6, 7]\n',
    'lambdas_a_probar = np.logspace(-4, 2, 11)\n',
    '\n',
    '# Grilla específica para el Viewpoint Puro (1D circular spline)\n',
    '# Ampliamos lambdas hasta 10000 (10^4) para evitar el efecto frontera y permitir curvas ultra-suaves\n',
    'splines_a_probar_view = [4, 5, 6, 7, 8, 9, 10]\n',
    'lambdas_a_probar_view = np.logspace(-2, 4, 13)\n',
    '\n',
    'print("=== 1. VALIDACIÓN CRUZADA: GAM POSICIÓN PURO ===")\n',
    'error_matrix_spatial, resultados_spatial = cross_validate_gam_grid(\n',
    '    X_cv[:, :2], Y_cv, folds, splines_a_probar, lambdas_a_probar\n',
    ')\n',
    'mejor_spatial = sorted(resultados_spatial, key=lambda x: x[3])[0]\n',
    'best_sp_spatial, best_lam_spatial = int(mejor_spatial[0]), mejor_spatial[1]\n',
    'print(f"Mejor GAM Posición Puro: splines={best_sp_spatial}, lambda={best_lam_spatial:.6f} | NLL CV={mejor_spatial[3]:.8f}")\n',
    '\n',
    'print("\\n=== 2. VALIDACIÓN CRUZADA: GAM VIEWPOINT PURO ===")\n',
    'error_matrix_view, resultados_view = cross_validate_gam_grid(\n',
    '    X_cv[:, 2:3], Y_cv, folds, splines_a_probar_view, lambdas_a_probar_view\n',
    ')\n',
    'mejor_view = sorted(resultados_view, key=lambda x: x[3])[0]\n',
    'best_sp_view, best_lam_view = int(mejor_view[0]), mejor_view[1]\n',
    'print(f"Mejor GAM Viewpoint Puro: splines={best_sp_view}, lambda={best_lam_view:.6f} | NLL CV={mejor_view[3]:.8f}")\n',
    '\n',
    'print("\\n=== 3. VALIDACIÓN CRUZADA: GAM POSICIÓN + VIEWPOINT CONJUNTO ===")\n',
    'error_matrix_joint, resultados_joint = cross_validate_gam_grid(\n',
    '    X_cv, Y_cv, folds, splines_a_probar, lambdas_a_probar\n',
    ')\n',
    'mejor_joint = sorted(resultados_joint, key=lambda x: x[3])[0]\n',
    'best_sp_joint, best_lam_joint = int(mejor_joint[0]), mejor_joint[1]\n',
    'print(f"Mejor GAM Posición + Viewpoint: splines={best_sp_joint}, lambda={best_lam_joint:.6f} | NLL CV={mejor_joint[3]:.8f}")\n'
]

# Modificar Celda 24 (Heatmap)
nb['cells'][24]['source'] = [
    '# Azul es mejor - Mostramos el mapa de la combinación conjunta\n',
    'plot_cv_heatmap(\n',
    '    error_matrix_joint, lambdas_a_probar, splines_a_probar,\n',
    '    title="GAM Pos+View Joint – Negative Log Likelihood", xlabel="lambda", ylabel="n_splines"\n',
    ')\n'
]

# Modificar Celda 29 (Retrain & Evaluation)
nb['cells'][29]['source'] = [
    'from pygam import PoissonGAM, te, s\n',
    '\n',
    '# 1. Re-entrenar GAM Posición + Viewpoint (3 columnas, usando sus propios hiperparámetros óptimos)\n',
    'gam_joint_final = retrain_best_gam(X_pool, Y_pool, best_sp_joint, best_lam_joint)\n',
    'print(f"GAM Posición + Viewpoint final entrenado (splines={best_sp_joint}, lambda={best_lam_joint:.6f})")\n',
    '\n',
    '# 2. Re-entrenar GAM Posición Puro (usando sus propios hiperparámetros óptimos)\n',
    'gam_spatial_final = retrain_best_gam(X_pool[:, :2], Y_pool, best_sp_spatial, best_lam_spatial)\n',
    'print(f"GAM Posición Puro final entrenado (splines={best_sp_spatial}, lambda={best_lam_spatial:.6f})")\n',
    '\n',
    '# 3. Re-entrenar GAM Viewpoint Puro (usando sus propios hiperparámetros óptimos y sin hardcodear)\n',
    'gam_viewpoint_final = retrain_best_gam(X_pool[:, 2:3], Y_pool, best_sp_view, best_lam_view)\n',
    'print(f"GAM Viewpoint Puro final entrenado (splines={best_sp_view}, lambda={best_lam_view:.6f})")\n',
    '\n',
    '# 4. Re-entrenar GLM Posición\n',
    'glm_final, cx, cy, sigma = retrain_best_glm(X_pool[:, :2], Y_pool, best_bines, best_alpha)\n',
    'print(f"GLM Posición final entrenado (bines={best_bines}, alpha={best_alpha:.6f})")\n',
    '\n',
    '# Evaluación en held-out\n',
    'nll_nulo = null_model_nll(Y_pool, Y_held)\n',
    '\n',
    '# Predicciones y NLL\n',
    'mu_gam_joint = gam_joint_final.predict(X_held)\n',
    'nll_gam_joint = poisson_nll_per_sample(Y_held, mu_gam_joint)\n',
    'r2_gam_joint  = pseudo_r2_mcfadden(nll_gam_joint, nll_nulo)\n',
    '\n',
    'mu_gam_spatial = gam_spatial_final.predict(X_held[:, :2])\n',
    'nll_gam_spatial = poisson_nll_per_sample(Y_held, mu_gam_spatial)\n',
    'r2_gam_spatial  = pseudo_r2_mcfadden(nll_gam_spatial, nll_nulo)\n',
    '\n',
    'mu_gam_viewpoint = gam_viewpoint_final.predict(X_held[:, 2:3])\n',
    'nll_gam_viewpoint = poisson_nll_per_sample(Y_held, mu_gam_viewpoint)\n',
    'r2_gam_viewpoint  = pseudo_r2_mcfadden(nll_gam_viewpoint, nll_nulo)\n',
    '\n',
    'mu_glm = predict_glm_on_new_data(glm_final, X_held[:, :2], cx, cy, sigma)\n',
    'nll_glm = poisson_nll_per_sample(Y_held, mu_glm)\n',
    'r2_glm  = pseudo_r2_mcfadden(nll_glm, nll_nulo)\n',
    '\n',
    'print(f"{"="*102}")\n',
    'print(f"RESULTADOS FINALES EN HELD-OUT")\n',
    'print(f"{"="*102}")\n',
    'print(f"{"Métrica":<25} {"Modelo Nulo":>14} {"GAM Posición":>14} {"GAM Viewpoint":>14} {"GAM Pos+View":>14} {"GLM Posición":>14}")\n',
    'print(f"{"-"*102}")\n',
    'print(f"{"NLL (held-out)":<25} {nll_nulo:>14.8f} {nll_gam_spatial:>14.8f} {nll_gam_viewpoint:>14.8f} {nll_gam_joint:>14.8f} {nll_glm:>14.8f}")\n',
    'print(f"{"Pseudo R² (McFadden)":<25} {"---":>14} {r2_gam_spatial:>14.6f} {r2_gam_viewpoint:>14.6f} {r2_gam_joint:>14.6f} {r2_glm:>14.6f}")\n',
    'print(f"{"-"*102}")\n',
    'print(f"Pseudo R² GAM Posición:  {r2_gam_spatial:.4f} ({r2_gam_spatial*100:.2f}%)")\n',
    'print(f"Pseudo R² GAM Viewpoint: {r2_gam_viewpoint:.4f} ({r2_gam_viewpoint*100:.2f}%)")\n',
    'print(f"Pseudo R² GAM Pos+View:  {r2_gam_joint:.4f} ({r2_gam_joint*100:.2f}%)")\n',
    'print(f"Pseudo R² GLM Posición:  {r2_glm:.4f} ({r2_glm*100:.2f}%)")\n',
    'print(f"{"="*102}")\n'
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Modificaciones en el notebook guardadas con éxito.')
