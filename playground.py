#!/usr/bin/env python
# coding: utf-8

# ## 0. Imports y Setup

# In[1]:


import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath('__file__')))

from scripts.utils import (
    _cargar_datos_neurona,
    preparar_datos_posicion,
    preparar_datos_head_direction,
    preparar_datos_mirada,
    firing_map,
    rate_map,
    get_gam_posicion,
    graficar_gam_posicion,
    glm_position,
)

from scripts.utils.angulo_preferido import (
    calcular_angulo_preferido_coseno,
    analizar_mirada_hacia_punto,
    encontrar_punto_preferido_mirada,
    evaluar_confianza_punto_preferido,
    evaluar_confianza_angulo_preferido,
    analizar_sintonizacion_perspectiva_continua,
    evaluar_confianza_sintonizacion_perspectiva,
)

from scripts.utils.cross_validation import (
    generate_all_splits,
    cross_validate_gam_grid,
    cross_validate_glm_grid,
    retrain_best_gam,
    retrain_best_glm,
    predict_glm_on_new_data,
    poisson_nll_per_sample,
    null_model_nll,
    pseudo_r2_mcfadden,
    plot_cv_heatmap,
)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Parámetros Globales
# Configuración de la neurona y el bin temporal.

# In[3]:


# ---- Parámetros de la neurona ----
sesion  = 2
tetrodo = 3
neurona = 3

# ---- Parámetros temporales ----
bin_size = 0.1  # segundos por bin


# ## 2. Firing Map
# Trayectoria completa del animal (gris) con los puntos de disparo de la neurona superpuestos (rojo).

# In[3]:


firing_map(sesion, tetrodo, neurona)


# ## 3. GLM de Posición
# GLM Poisson. Resolución controlada por `n_bines`, regularización ridge por `alpha`.

# In[27]:


glm_position(sesion, tetrodo, neurona, n_bines=6, alpha=0.01)


# ## 4. GAM de Posición (pyGAM)
# Entrena o carga GAM desde caché `.pkl`.

# In[6]:


splines = 9
lam = 0.5

modelo_gam, X, Y = get_gam_posicion(
    sesion, tetrodo, neurona, 
    splines=splines, lam=lam, 
    bin_size_sec=bin_size,
    force_retrain=False
)


# In[7]:


graficar_gam_posicion(modelo_gam, X, Y, sesion, tetrodo, neurona, splines, bin_size_sec=bin_size)


# ## 5. Head Direction
# Curva de sintonización de la dirección de la cabeza. Se calcula el ángulo preferido mediante ajuste de coseno.

# In[5]:


# ---- Parámetros de la neurona ----
sesion  = 2
tetrodo = 3
neurona = 5

# Neuronas encontradas de VIEWPOINT
# 2 3 4
# 2 2 9
# 2 2 8
# 2 2 3 

# Neurona encontrada HEAD-DIRECTION
# 2 3 5

# ---- Parámetros temporales ----
bin_size = 0.1  # segundos por bin


# In[6]:


ang_bins_deg, conteo_spikes = preparar_datos_head_direction(
    sesion, tetrodo, neurona, 
    bin_size_sec=bin_size
)

print(f"Bines temporales: {len(ang_bins_deg)}")
print(f"Spikes totales: {np.sum(conteo_spikes)}")


# In[7]:


# Ajuste matemático para encontrar el ángulo preferido
angulos_candidatos, delta_R, amplitud, angulo_pref, baseline, fn_ajuste = calcular_angulo_preferido_coseno(
    ang_bins_deg, conteo_spikes, tiempos_bin=bin_size, tolerancia_deg=30
)

print(f"Ángulo preferido (HD): {angulo_pref:.1f}°")
print(f"Amplitud de selectividad: {amplitud:.3f}")
print(f"Baseline ajustado: {baseline:.3f}")

# Graficar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Tuning curve empírica
n_bins_hist = 36
bin_edges = np.linspace(0, 360, n_bins_hist + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

spike_sum = np.zeros(n_bins_hist)
occ_count = np.zeros(n_bins_hist)
for i in range(n_bins_hist):
    mask = (ang_bins_deg >= bin_edges[i]) & (ang_bins_deg < bin_edges[i+1])
    spike_sum[i] = np.sum(conteo_spikes[mask])
    occ_count[i] = np.sum(mask) * bin_size

rate_hd = np.divide(spike_sum, occ_count, where=occ_count > 0, out=np.zeros_like(spike_sum))

ax1.bar(bin_centers, rate_hd, width=360/n_bins_hist, alpha=0.7, color='steelblue', edgecolor='white')
ax1.axvline(angulo_pref, color='red', linestyle='--', label=f'Preferido: {angulo_pref:.1f}°')
ax1.set_xlabel('Dirección de Cabeza (°)')
ax1.set_ylabel('Tasa de Disparo (Hz)')
ax1.set_title('Tuning Curve de Head Direction')
ax1.legend()

# Panel 2: Delta R + modelo matemático
x_fit = np.linspace(0, 360, 500)
# ¡MIRA QUÉ LIMPIO QUEDA! La función ya "recuerda" sus propios parámetros
y_fit = fn_ajuste(x_fit) 

ax2.scatter(angulos_candidatos, delta_R, color='steelblue', s=50, zorder=5, label='ΔR empírico')
ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label='Ajuste del Modelo')
ax2.axvline(angulo_pref, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Ángulo candidato (°)')
ax2.set_ylabel('ΔR (Hz)')
ax2.set_title('Curva del Modelo Analítico sobre ΔR')
ax2.legend()

plt.tight_layout()
plt.show()


# ## 6. Mirada
# Busca el punto en la caja 2D que maximiza la selectividad de mirada de la neurona.
# Evalúa `ΔR = Tasa(mirando al punto) - Tasa(mirando a otro lado)` sobre una grilla.

# In[8]:


# Cargar datos con posición + ángulo de cabeza
x_bins, y_bins, ang_bins_rad, spikes_gaze = preparar_datos_mirada(
    sesion, tetrodo, neurona, bin_size_sec=bin_size
)
print(f"Bines: {len(x_bins)} | Spikes totales: {np.sum(spikes_gaze)}")


# In[127]:


# Buscar el punto preferido de mirada en la caja
XX, YY, delta_R_matrix, coord_best, best_delta_R = encontrar_punto_preferido_mirada(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, 
    tiempos_bin=bin_size, grid_res=25, tolerancia_deg=30
)

x_best, y_best = coord_best
print(f"Punto preferido de mirada: ({x_best:.1f}, {y_best:.1f}) cm")
print(f"Selectividad ΔR: {best_delta_R:.2f} Hz")

# Evaluar métricas en esa coordenada
rate_mira, rate_no_mira, mira_al_punto = analizar_mirada_hacia_punto(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, bin_size, x_best, y_best, tolerancia_deg=30
)
print(f"Tasa mirando al punto: {rate_mira:.2f} Hz")
print(f"Tasa mirando fuera:    {rate_no_mira:.2f} Hz")


# In[128]:


# Visualización del análisis de mirada
fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

# Panel A: Heatmap de selectividad
ax = axes[0]
mesh = ax.pcolormesh(XX, YY, delta_R_matrix, cmap='jet', shading='nearest')
fig.colorbar(mesh, ax=ax, label='ΔR (Hz)')
ax.scatter(x_best, y_best, color='gold', marker='*', s=250, edgecolor='black', zorder=10)
ax.set_title('A: Mapa 2D de Selectividad de Mirada')
ax.set_xlabel('X (cm)'); ax.set_ylabel('Y (cm)')
ax.axis('equal')

# Panel B: Barras comparativas
ax = axes[1]
bars = ax.bar(['Mirando al\nPunto (ON)', 'Mirando\nFuera (OFF)'], 
              [rate_mira, rate_no_mira], color=['#2196F3', '#FF5722'], alpha=0.85, edgecolor='black')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f'{h:.2f} Hz', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('Tasa de Disparo (Hz)')
ax.set_title('B: Comparación Gaze ON vs OFF')

plt.suptitle(f'Análisis de Mirada | s={sesion} t={tetrodo} c={neurona} | Punto: ({x_best:.1f}, {y_best:.1f})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()


# ### 6.1 Test de Control de Significancia Estadística (Spike Shifting)
# Para validar si la sintonización del ángulo preferido (Head Direction) y de la mirada (Viewpoint) son significativos, realizamos un análisis de control mediante el **desplazamiento circular de los spikes** (*spike shifting*).
# 
# Este test:
# 1. Desplaza circularmente el vector de spikes por una cantidad de tiempo aleatoria (mínimo de 10 segundos de cada lado para evitar solapamientos inmediatos).
# 2. Rompe la correlación temporal exacta entre los eventos de disparo de la neurona y el comportamiento en tiempo real del animal (dirección de cabeza y posición en la grilla).
# 3. Preserva de manera idéntica todas las propiedades intrínsecas de disparo de la célula (bursting, tasa de disparo promedio, autocorrelación temporal).
# 
# Repitiendo este procedimiento $N$ veces (por ejemplo, 100 veces), construimos una **distribución nula** para evaluar si nuestra selectividad/amplitud experimental es significativamente mayor de lo esperado por mero azar.

# In[130]:


from scripts.utils.angulo_preferido import evaluar_confianza_angulo_preferido
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. EVALUAR CONFIANZA DE HEAD DIRECTION (HD)
# ==========================================
print("Ejecutando test de control para Head Direction (100 desplazamientos)")
confianza_hd = evaluar_confianza_angulo_preferido(
    ang_bins_deg, conteo_spikes, tiempos_bin=bin_size, tolerancia_deg=30, n_shifts=100, min_shift_sec=10.0
)

print("\n=== RESULTADOS CONFIANZA HEAD DIRECTION ===")
print(f"Amplitud Original: {confianza_hd['original_amplitude']:.4f} Hz")
print(f"Media Distribución Nula: {confianza_hd['mean_shuffled_amplitude']:.4f} Hz ± {confianza_hd['std_shuffled_amplitude']:.4f} Hz")
print(f"Z-Score: {confianza_hd['z_score']:.2f}")
print(f"P-Valor: {confianza_hd['p_value']:.4f}")
print(f"¿Es significativo?: {'SÍ (p < 0.05)' if confianza_hd['confianza_significativa'] else 'NO (p >= 0.05)'}")

# ==========================================
# 2. EVALUAR CONFIANZA DE MIRADA (VIEWPOINT)
# ==========================================
print("\nEjecutando test de control para Viewpoint de Mirada (100 desplazamientos)")
confianza_gaze = evaluar_confianza_punto_preferido(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, 
    grid_res=25, tolerancia_deg=30, n_shifts=100, min_shift_sec=10.0
)

print("\n=== RESULTADOS CONFIANZA VIEWPOINT (MIRADA) ===")
print(f"Coordenada Preferida: {confianza_gaze['original_coord']}")
print(f"Selectividad ΔR Original: {confianza_gaze['original_delta_R']:.4f} Hz")
print(f"Media Distribución Nula: {confianza_gaze['mean_shuffled_delta_R']:.4f} Hz ± {confianza_gaze['std_shuffled_delta_R']:.4f} Hz")
print(f"Z-Score: {confianza_gaze['z_score']:.2f}")
print(f"P-Valor: {confianza_gaze['p_value']:.4f}")
print(f"¿Es significativo?: {'SÍ (p < 0.05)' if confianza_gaze['confianza_significativa'] else 'NO (p >= 0.05)'}")

# ==========================================
# 3. VISUALIZACIÓN DE AMBAS DISTRIBUCIONES NULAS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel izquierda: HD Amplitudes
sns.histplot(confianza_hd['shuffled_amplitudes'], ax=ax1, color='gray', alpha=0.6, kde=True, label='Shuffled (Nula)')
ax1.axvline(confianza_hd['original_amplitude'], color='red', linestyle='--', linewidth=2.5, 
            label=f'Original: {confianza_hd["original_amplitude"]:.3f} Hz\np={confianza_hd["p_value"]:.3f} (Z={confianza_hd["z_score"]:.1f})')
ax1.set_title('Test de Control: Amplitud de Head Direction', fontweight='bold')
ax1.set_xlabel('Amplitud del Ajuste Analítico (Hz)')
ax1.set_ylabel('Frecuencia')
ax1.legend()

# Panel derecha: Viewpoint Delta R
sns.histplot(confianza_gaze['shuffled_delta_Rs'], ax=ax2, color='gray', alpha=0.6, kde=True, label='Shuffled (Nula)')
ax2.axvline(confianza_gaze['original_delta_R'], color='red', linestyle='--', linewidth=2.5, 
            label=f'Original: {confianza_gaze["original_delta_R"]:.3f} Hz\np={confianza_gaze["p_value"]:.3f} (Z={confianza_gaze["z_score"]:.1f})')
ax2.set_title('Test de Control: Selectividad de Mirada (ΔR)', fontweight='bold')
ax2.set_xlabel('Selectividad Máxima ΔR (Hz)')
ax2.set_ylabel('Frecuencia')
ax2.legend()

plt.suptitle(f'Validación de Preferencias mediante Spike Shifting (100 shuffles, shift > 10s)', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()


# ### 6.2 Sintonización de Perspectiva Circular Continua (Enfoque MATLAB)
# Evaluación del modelo de proyección de coseno continuo sobre la grilla 2D para identificar el punto óptimo de perspectiva y validar su significancia estadística.

# In[ ]:


# ---- Parámetros de la grilla y análisis ----
grid_res = 25

print("Ejecutando Sintonización de Perspectiva Circular Continua (MATLAB style)...")
res_persp = analizar_sintonizacion_perspectiva_continua(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, grid_res=grid_res
)

x_best_p, y_best_p = res_persp['best_coord']
print(f"Coordenada Óptima de Perspectiva: ({x_best_p:.1f}, {y_best_p:.1f}) cm")
print(f"Amplitud de Proyección Máxima: {res_persp['persp_indx']:.4f} Hz")

# ---- Evaluación de confianza mediante Spike Shifting (100 desplazamientos) ----
print("\nEjecutando test de control de significancia mediante Spike Shifting (100 shuffles)...")
confianza_persp = evaluar_confianza_sintonizacion_perspectiva(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, grid_res=grid_res, n_shifts=100, min_shift_sec=10.0
)

print("\n=== RESULTADOS CONFIANZA PERSPECTIVA CONTINUA ===")
print(f"Amplitud Original: {confianza_persp['original_persp_indx']:.4f} Hz")
print(f"Media Distribución Nula: {confianza_persp['mean_shuffled_persp_indx']:.4f} Hz ± {confianza_persp['std_shuffled_persp_indx']:.4f} Hz")
print(f"Z-Score: {confianza_persp['z_score']:.2f}")
print(f"P-Valor: {confianza_persp['p_value']:.4f}")
print(f"¿Es significativo?: {'SÍ (p < 0.05)' if confianza_persp['confianza_significativa'] else 'NO (p >= 0.05)'}")

# ---- Generar visualización elegante y moderna de 3 paneles ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Heatmap de Perspectividad
ax1 = axes[0]
XX_p = res_persp['XX']
YY_p = res_persp['YY']
map_p_matrix = res_persp['map_p_matrix']
mesh1 = ax1.pcolormesh(XX_p, YY_p, map_p_matrix, cmap='jet', shading='nearest')
fig.colorbar(mesh1, ax=ax1, label='Amplitud de Coseno (Hz)')
ax1.scatter(x_best_p, y_best_p, color='gold', marker='*', s=300, edgecolor='black', zorder=10, label=f'Óptimo ({x_best_p:.1f}, {y_best_p:.1f})')
ax1.set_title('A: Mapa de Perspectividad 2D', fontweight='bold')
ax1.set_xlabel('X (cm)')
ax1.set_ylabel('Y (cm)')
ax1.axis('equal')
ax1.legend()

# Panel 2: Tuning Curve Corregida y Modelo de Coseno Ajustado
ax2 = axes[1]
hdbins_deg = np.degrees(res_persp['hdbins'])
tuning_curve_best = res_persp['tuning_curve_best']
ax2.bar(hdbins_deg, tuning_curve_best, width=360/36, alpha=0.6, color='steelblue', edgecolor='white', label='Tuning Curve Empírica')

# Graficar modelo ajustado de coseno
x_fit = np.linspace(0, 360, 500)
rate_mean = np.mean(spikes_gaze) / bin_size
y_fit = rate_mean + res_persp['persp_indx'] * np.cos(np.radians(x_fit) - res_persp['theta0'])
y_fit = np.clip(y_fit, 0, None)
ax2.plot(x_fit, y_fit, color='crimson', linewidth=2.5, label='Ajuste Coseno Proyectado')
ax2.axvline(np.degrees(res_persp['theta0']), color='gold', linestyle='--', linewidth=2, label=f'Fase: {np.degrees(res_persp["theta0"]):.1f}°')
ax2.set_xlabel('Head Direction Corregida (°)')
ax2.set_ylabel('Tasa de Disparo (Hz)')
ax2.set_title('B: Tuning Curve en Punto de Perspectiva', fontweight='bold')
ax2.legend()

# Panel 3: Distribución Nula mediante Spike Shifting
ax3 = axes[2]
sns.histplot(confianza_persp['shuffled_persp_indxs'], ax=ax3, color='gray', alpha=0.6, kde=True, label='Shuffled (Nula)')
ax3.axvline(confianza_persp['original_persp_indx'], color='red', linestyle='--', linewidth=2.5,
            label=f'Original: {confianza_persp["original_persp_indx"]:.4f} Hz\\np={confianza_persp["p_value"]:.3f} (Z={confianza_persp["z_score"]:.1f})')
ax3.set_title('C: Test de Control Spike Shifting', fontweight='bold')
ax3.set_xlabel('Amplitud de Proyección Máxima (Hz)')
ax3.set_ylabel('Frecuencia')
ax3.legend()

plt.suptitle(f'Sintonización de Perspectiva Circular Continua | s={sesion} t={tetrodo} c={neurona}', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()


# ## 7. Cross-Validation con Held-Out y Pseudo R²
# Pipeline completo:
# 1. Partición unificada (held-out intercalado + folds de CV con buffers temporales 2s)
# 2. CV para selección de hiperparámetros (GAM y GLM)
# 3. Re-entrenamiento del mejor modelo en el train pool
# 4. Evaluación final en held-out con Pseudo R² de McFadden

# ### 7.1 Cargar datos y generar particiones

# In[21]:


# Bloque:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
# Rol:    HO   0   1   2   3   4  HO   0   1   2   3   4  HO   0   1   2   3   4  HO


X_cv, Y_cv = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size)
print(f"Total de muestras: {len(X_cv)}")

folds, held_out_idx, train_pool_idx, roles = generate_all_splits(
    n_muestras=len(X_cv),
    bin_size_sec=bin_size,
    block_size_sec=60,
    n_folds=5,
    buffer_sec=2
)

X_pool, Y_pool = X_cv[train_pool_idx], Y_cv[train_pool_idx]
X_held, Y_held = X_cv[held_out_idx], Y_cv[held_out_idx]

n_held = np.sum(roles == -1)
n_cv   = np.sum(roles >= 0)
print(f"Bloques: {len(roles)} | Held-out: {n_held} | CV: {n_cv}")
print(f"Train pool: {len(X_pool)} ({100*len(X_pool)/len(X_cv):.1f}%)")
print(f"Held-out:   {len(X_held)} ({100*len(X_held)/len(X_cv):.1f}%)")
for k, (tr, te) in enumerate(folds):
    print(f"  Fold {k}: train={len(tr)} | test={len(te)}")


# ### 7.2 CV — Selección de hiperparámetros GAM

# In[22]:


# hiperparametros GAM. Aprox. ~2min 10s en entrenar los 44 GAMS
splines_a_probar = [4, 5, 6, 7]
lambdas_a_probar = np.logspace(-4, 2, 11)

error_matrix_gam, resultados_gam = cross_validate_gam_grid(
    X_cv, Y_cv, folds, splines_a_probar, lambdas_a_probar
)

mejor_gam_cv = sorted(resultados_gam, key=lambda x: x[3])[0]
best_sp, best_lam_gam = int(mejor_gam_cv[0]), mejor_gam_cv[1]

print(f"\n Mejor GAM: splines={best_sp}, lambda={best_lam_gam:.6f} | NLL CV={mejor_gam_cv[3]:.8f}")


# In[23]:


# Azul es mejor
plot_cv_heatmap(
    error_matrix_gam, lambdas_a_probar, splines_a_probar,
    title='GAM — Negative Log Likelihood', xlabel='lambda', ylabel='n_splines'
)


# ### 7.3 CV — Selección de hiperparámetros GLM

# In[24]:


# hiperparametros GLM. Aprox. ~37s los 44 GLMs
bines_a_probar  = [4, 5, 6, 7]
alphas_a_probar = np.logspace(-6, 0, 11)

error_matrix_glm, resultados_glm = cross_validate_glm_grid(
    X_cv, Y_cv, folds, bines_a_probar, alphas_a_probar
)

mejor_glm_cv = sorted(resultados_glm, key=lambda x: x[2])[0]
best_bines, best_alpha = int(mejor_glm_cv[0]), mejor_glm_cv[1]
print(f"\nMejor GLM: bines={best_bines}x{best_bines}, alpha={best_alpha:.6f} | NLL CV={mejor_glm_cv[2]:.8f}")


# In[25]:


plot_cv_heatmap(
    error_matrix_glm, alphas_a_probar, bines_a_probar,
    title='GLM — Negative Log Likelihood', xlabel='alpha', ylabel='n_bases'
)


# ### 7.4 Re-entrenar mejores modelos y evaluar en Held-Out

# In[26]:


# Re-entrenar en todo el train pool
gam_final = retrain_best_gam(X_pool, Y_pool, best_sp, best_lam_gam)
print(f"GAM final (splines={best_sp}, lambda={best_lam_gam:.6f})")

glm_final, cx, cy, sigma = retrain_best_glm(X_pool, Y_pool, best_bines, best_alpha)
print(f"GLM final (bines={best_bines}, alpha={best_alpha:.6f})")

# Evaluación en held-out
nll_nulo = null_model_nll(Y_pool, Y_held)

mu_gam = gam_final.predict(X_held)
nll_gam = poisson_nll_per_sample(Y_held, mu_gam)
r2_gam  = pseudo_r2_mcfadden(nll_gam, nll_nulo)

mu_glm = predict_glm_on_new_data(glm_final, X_held, cx, cy, sigma)
nll_glm = poisson_nll_per_sample(Y_held, mu_glm)
r2_glm  = pseudo_r2_mcfadden(nll_glm, nll_nulo)

print(f"{'='*70}")
print(f"RESULTADOS FINALES EN HELD-OUT")
print(f"{'='*70}")
print(f"{'Métrica':<25} {'Nulo':>14} {'GAM':>14} {'GLM':>14}")
print(f"{'-'*70}")
print(f"{'NLL':<25} {nll_nulo:>14.8f} {nll_gam:>14.8f} {nll_glm:>14.8f}")
print(f"{'Pseudo R2 (McFadden)':<25} {'---':>14} {r2_gam:>14.6f} {r2_glm:>14.6f}")
print(f"{'-'*70}")
print(f"Pseudo R² GAM: {r2_gam:.4f} ({r2_gam*100:.2f}%)")
print(f"Pseudo R² GLM: {r2_glm:.4f} ({r2_glm*100:.2f}%)")

