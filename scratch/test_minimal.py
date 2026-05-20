import numpy as np
import time
from scripts.utils import preparar_datos_mirada
from scripts.utils.angulo_preferido import (
    analizar_sintonizacion_perspectiva_continua,
    evaluar_confianza_sintonizacion_perspectiva
)

sesion = 2
tetrodo = 2
neurona = 3
bin_size = 0.02

print("Cargando datos...")
x_bins, y_bins, ang_bins_rad, spikes_gaze = preparar_datos_mirada(
    sesion, tetrodo, neurona, bin_size_sec=bin_size
)

print(f"Datos cargados: {len(x_bins)} bines. Iniciando análisis continuo...")
start_time = time.time()
res = analizar_sintonizacion_perspectiva_continua(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, grid_res=20
)
duration = time.time() - start_time
print(f"¡Análisis continuo completado en {duration:.4f} segundos!")
print(f"Coordenada Óptima de Perspectiva: {res['best_coord']}")
print(f"Amplitud de Proyección Máxima: {res['persp_indx']:.4f} Hz")

print("\nIniciando test de confianza (Spike Shifting, 20 repeticiones)...")
start_time = time.time()
conf = evaluar_confianza_sintonizacion_perspectiva(
    x_bins, y_bins, ang_bins_rad, spikes_gaze, tiempos_bin=bin_size, grid_res=20, n_shifts=20, min_shift_sec=10.0
)
duration = time.time() - start_time
print(f"¡Test de confianza completado en {duration:.4f} segundos!")
print(f"Z-Score: {conf['z_score']:.2f}")
print(f"P-Valor: {conf['p_value']:.4f}")
print(f"¿Es significativo?: {conf['confianza_significativa']}")
