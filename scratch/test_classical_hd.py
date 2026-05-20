import numpy as np
import time
from scripts.utils import preparar_datos_head_direction
from scripts.utils.angulo_preferido import (
    calcular_angulo_preferido_coseno,
    evaluar_confianza_angulo_preferido
)

sesion = 2
tetrodo = 2
neurona = 3
bin_size = 0.02

print("Cargando datos clásicos de Head Direction...")
ang_bins_deg, conteo_spikes = preparar_datos_head_direction(
    sesion, tetrodo, neurona
)

print(f"Datos cargados: {len(ang_bins_deg)} registros. Calculando sintonización clásica...")
start = time.time()
ang_cand, delta_R, amp, ang_pref, base, fn_fit = calcular_angulo_preferido_coseno(
    ang_bins_deg, conteo_spikes, tiempos_bin=bin_size, tolerancia_deg=30
)
duration = time.time() - start
print(f"¡Cálculo clásico completado en {duration:.4f} segundos!")
print(f"Ángulo Preferido (HD): {ang_pref:.1f}°")
print(f"Amplitud de Selectividad: {amp:.4f} Hz")
print(f"Baseline ajustado: {base:.4f} Hz")

print("\nIniciando test de confianza clásica (Spike Shifting, 20 repeticiones)...")
start = time.time()
conf = evaluar_confianza_angulo_preferido(
    ang_bins_deg, conteo_spikes, tiempos_bin=bin_size, tolerancia_deg=30, n_shifts=20, min_shift_sec=10.0
)
duration = time.time() - start
print(f"¡Test de confianza clásica completado en {duration:.4f} segundos!")
print(f"Z-Score clásico: {conf['z_score']:.2f}")
print(f"P-Valor clásico: {conf['p_value']:.4f}")
print(f"¿Es significativo el HD clásico?: {conf['confianza_significativa']}")
