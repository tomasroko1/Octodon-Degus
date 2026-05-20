import json

with open("playground.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 21 corresponds to Section 7.1 (loading and preparing data)
cell_21 = nb["cells"][21]
cell_21["source"] = [
    "# Cargar datos de posición y dirección de cabeza con filtro de velocidad (> 2 cm/s)\n",
    "x_bins, y_bins, ang_bins_rad, spikes_gaze = preparar_datos_mirada(\n",
    "    sesion, tetrodo, neurona, bin_size_sec=bin_size\n",
    ")\n",
    "\n",
    "# Obtener la coordenada óptima de perspectiva (ancla de mirada) de forma dinámica o usar fallback\n",
    "try:\n",
    "    x_best_p, y_best_p = res_persp['best_coord']\n",
    "except NameError:\n",
    "    x_best_p, y_best_p = 59.7, 3.8 # Fallback obtenido en Sección 6\n",
    "\n",
    "print(f\"Coordenada Óptima de Perspectiva utilizada: ({x_best_p:.1f}, {y_best_p:.1f}) cm\")\n",
    "\n",
    "# Calcular la Head Direction corregida por perspectiva\n",
    "dy = y_best_p - y_bins\n",
    "dx = x_best_p - x_bins\n",
    "ang_hacia_punto = np.arctan2(dy, dx)\n",
    "theta_corrected = np.mod(ang_bins_rad - ang_hacia_punto, 2 * np.pi)\n",
    "\n",
    "# X_cv tendrá 3 columnas: [Posición X, Posición Y, Viewpoint Corregido (rad)]\n",
    "# Esto alimentará al GAM con la fórmula: te(0, 1) + s(2, bs='cc')\n",
    "X_cv = np.column_stack((x_bins, y_bins, theta_corrected))\n",
    "Y_cv = spikes_gaze\n",
    "print(f\"Total de muestras (con velocidad > 2 cm/s): {len(X_cv)}\")\n",
    "\n",
    "folds, held_out_idx, train_pool_idx, roles = generate_all_splits(\n",
    "    n_muestras=len(X_cv),\n",
    "    bin_size_sec=bin_size,\n",
    "    block_size_sec=60,\n",
    "    n_folds=5,\n",
    "    buffer_sec=2\n",
    ")\n",
    "\n",
    "X_pool, Y_pool = X_cv[train_pool_idx], Y_cv[train_pool_idx]\n",
    "X_held, Y_held = X_cv[held_out_idx], Y_cv[held_out_idx]\n",
    "\n",
    "n_held = np.sum(roles == -1)\n",
    "n_cv   = np.sum(roles >= 0)\n",
    "print(f\"Bloques: {len(roles)} | Held-out: {n_held} | CV: {n_cv}\")\n",
    "print(f\"Train pool: {len(X_pool)} ({100*len(X_pool)/len(X_cv):.1f}%)\")\n",
    "print(f\"Held-out:   {len(X_held)} ({100*len(X_held)/len(X_cv):.1f}%)\")\n",
    "for k, (tr, te) in enumerate(folds):\n",
    "    print(f\"  Fold {k}: train={len(tr)} | test={len(te)}\")\n"
]

# Cell 29 corresponds to Section 7.4 (retraining and evaluation)
cell_29 = nb["cells"][29]
cell_29["source"] = [
    "# 1. Re-entrenar GAM Posición + Viewpoint (3 columnas)\n",
    "gam_joint_final = retrain_best_gam(X_pool, Y_pool, best_sp, best_lam_gam)\n",
    "print(f\"GAM Posición + Viewpoint final entrenado (splines={best_sp}, lambda={best_lam_gam:.6f})\")\n",
    "\n",
    "# 2. Re-entrenar GAM Posición Puro (usando solo las primeras 2 columnas)\n",
    "gam_spatial_final = retrain_best_gam(X_pool[:, :2], Y_pool, best_sp, best_lam_gam)\n",
    "print(f\"GAM Posición Puro final entrenado (splines={best_sp}, lambda={best_lam_gam:.6f})\")\n",
    "\n",
    "# 3. Re-entrenar GLM Posición (usando las primeras 2 columnas)\n",
    "glm_final, cx, cy, sigma = retrain_best_glm(X_pool[:, :2], Y_pool, best_bines, best_alpha)\n",
    "print(f\"GLM Posición final entrenado (bines={best_bines}, alpha={best_alpha:.6f})\")\n",
    "\n",
    "# Evaluación en held-out\n",
    "nll_nulo = null_model_nll(Y_pool, Y_held)\n",
    "\n",
    "# Predicciones y NLL\n",
    "mu_gam_joint = gam_joint_final.predict(X_held)\n",
    "nll_gam_joint = poisson_nll_per_sample(Y_held, mu_gam_joint)\n",
    "r2_gam_joint  = pseudo_r2_mcfadden(nll_gam_joint, nll_nulo)\n",
    "\n",
    "mu_gam_spatial = gam_spatial_final.predict(X_held[:, :2])\n",
    "nll_gam_spatial = poisson_nll_per_sample(Y_held, mu_gam_spatial)\n",
    "r2_gam_spatial  = pseudo_r2_mcfadden(nll_gam_spatial, nll_nulo)\n",
    "\n",
    "mu_glm = predict_glm_on_new_data(glm_final, X_held[:, :2], cx, cy, sigma)\n",
    "nll_glm = poisson_nll_per_sample(Y_held, mu_glm)\n",
    "r2_glm  = pseudo_r2_mcfadden(nll_glm, nll_nulo)\n",
    "\n",
    "print(f\"{'='*85}\")\n",
    "print(f\"RESULTADOS FINALES EN HELD-OUT\")\n",
    "print(f\"{'='*85}\")\n",
    "print(f\"{'Métrica':<25} {'Modelo Nulo':>14} {'GAM Posición':>14} {'GAM Pos+View':>14} {'GLM Posición':>14}\")\n",
    "print(f\"{'-'*85}\")\n",
    "print(f\"{'NLL (held-out)':<25} {nll_nulo:>14.8f} {nll_gam_spatial:>14.8f} {nll_gam_joint:>14.8f} {nll_glm:>14.8f}\")\n",
    "print(f\"{'Pseudo R² (McFadden)':<25} {'---':>14} {r2_gam_spatial:>14.6f} {r2_gam_joint:>14.6f} {r2_glm:>14.6f}\")\n",
    "print(f\"{'-'*85}\")\n",
    "print(f\"Pseudo R² GAM Posición: {r2_gam_spatial:.4f} ({r2_gam_spatial*100:.2f}%)\")\n",
    "print(f\"Pseudo R² GAM Pos+View: {r2_gam_joint:.4f} ({r2_gam_joint*100:.2f}%)\")\n",
    "print(f\"Pseudo R² GLM Posición: {r2_glm:.4f} ({r2_glm*100:.2f}%)\")\n"
]

with open("playground.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("¡Notebook actualizado con éxito!")
