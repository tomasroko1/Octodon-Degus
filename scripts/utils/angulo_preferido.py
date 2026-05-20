"""
métodos:
- calcular_angulo_preferido_coseno: calcula el ángulo preferido y la amplitud mediante ajuste de coseno (Head Direction clásico).
- analizar_mirada_hacia_punto: mide la selectividad del disparo al mirar hacia una coordenada 2D específica.
- encontrar_punto_preferido_mirada: barre una grilla espacial en 2D para identificar el punto que maximiza la selectividad de la mirada.
- evaluar_confianza_punto_preferido: realiza un test de significancia estadística para el punto preferido usando desplazamientos circulares de spikes.
- evaluar_confianza_angulo_preferido: realiza un test de significancia estadística para la sintonización de dirección de cabeza (HD) usando desplazamientos circulares de spikes.
"""
from IPython.terminal import prompts
import numpy as np
from scipy.optimize import curve_fit
import sys
import os
import matplotlib.pyplot as plt

def calcular_angulo_preferido_coseno(angulos_reales, spikes, tiempos_bin, tolerancia_deg=30):
    angulos_candidatos = np.arange(0, 360, 10)
    delta_R = []
    
    for theta in angulos_candidatos:
        diff_angular = np.abs((angulos_reales - theta + 180) % 360 - 180)
        
        mira_hacia = diff_angular <= tolerancia_deg
        no_mira_hacia = ~mira_hacia
        
        if np.sum(mira_hacia) > 0 and np.sum(no_mira_hacia) > 0:
            rate_mira = np.sum(spikes[mira_hacia]) / (np.sum(mira_hacia) * tiempos_bin)
            rate_no_mira = np.sum(spikes[no_mira_hacia]) / (np.sum(no_mira_hacia) * tiempos_bin)
            delta_R.append(rate_mira - rate_no_mira)
        else:
            delta_R.append(0.0)
            
    delta_R = np.array(delta_R)
    
    def funcion_von_mises(x, A, kappa, theta_pref_rad, B):
        return A * np.exp(kappa * np.cos(np.radians(x) - theta_pref_rad)) + B
    
    # Estimación inicial
    amp_emp = np.max(delta_R) - np.min(delta_R)
    kappa_emp = 1.0 
    theta_emp = np.radians(angulos_candidatos[np.argmax(delta_R)])
    base_emp = np.min(delta_R)
    p0 = [amp_emp, kappa_emp, theta_emp, base_emp]
    
    # === LA SOLUCIÓN AQUÍ ===
    # Forzamos A >= 0 y kappa >= 0. 
    # Dejamos a theta y B libres (-inf a inf)
    bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    
    try:
        popt, _ = curve_fit(funcion_von_mises, angulos_candidatos, delta_R, p0=p0, bounds=bounds)
    except Exception:
        popt = p0
        
    amplitud_optima = popt[0] * np.exp(popt[1]) 
    angulo_preferido_deg = np.degrees(popt[2]) % 360
    baseline = popt[3]
    
    funcion_lista_para_graficar = lambda x: funcion_von_mises(x, *popt)
    
    return angulos_candidatos, delta_R, amplitud_optima, angulo_preferido_deg, baseline, funcion_lista_para_graficar   


def analizar_mirada_hacia_punto(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, x_target, y_target, tolerancia_deg=30):
    """
    Calcula la tasa de disparo de la neurona cuando el animal mira hacia un punto específico 
    (x_target, y_target) en el mapa 2D, en comparación con cuando mira hacia otros lados.
    """
    dx = x_target - x_bins
    dy = y_target - y_bins
    
    # Calcular el ángulo desde la posición del animal hasta el punto de interés
    ang_hacia_punto = np.arctan2(dy, dx)
    
    # Diferencia angular circular mínima entre la cabeza del animal y el punto en el mapa
    diff_ang_rad = np.arctan2(np.sin(ang_bins_rad - ang_hacia_punto), np.cos(ang_bins_rad - ang_hacia_punto))
    diff_ang_deg = np.abs(np.degrees(diff_ang_rad))
    
    # Máscaras de mirar al punto vs mirar a otro lado
    mira_al_punto = diff_ang_deg <= tolerancia_deg
    mira_otro_lado = ~mira_al_punto
    
    rate_mira = 0.0
    rate_no_mira = 0.0
    
    if np.sum(mira_al_punto) > 0:
        rate_mira = np.sum(spikes[mira_al_punto]) / (np.sum(mira_al_punto) * tiempos_bin)
    if np.sum(mira_otro_lado) > 0:
        rate_no_mira = np.sum(spikes[mira_otro_lado]) / (np.sum(mira_otro_lado) * tiempos_bin)
        
    return rate_mira, rate_no_mira, mira_al_punto


def encontrar_punto_preferido_mirada(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, grid_res=20, tolerancia_deg=30, min_dist=5.0):
    """
    Recorre una grilla de puntos 2D en el mapa y evalúa cuál de todas las coordenadas
    maximiza la selectividad de la mirada (Tasa_Mirando - Tasa_Mirando_Otro_Lado).
    """
    x_min, x_max = np.min(x_bins), np.max(x_bins)
    y_min, y_max = np.min(y_bins), np.max(y_bins)
    
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    
    XX, YY = np.meshgrid(x_grid, y_grid)
    delta_R_matrix = np.zeros_like(XX)
    
    best_delta_R = -np.inf
    best_coord = (0.0, 0.0)
    
    for i in range(grid_res):
        for j in range(grid_res):
            x_target = XX[i, j]
            y_target = YY[i, j]
            
            # Filtro: Omitimos evaluar bines cuando el animal está encima del punto analizado
            dist_al_punto = np.sqrt((x_bins - x_target)**2 + (y_bins - y_target)**2)
            validos = dist_al_punto > min_dist
            
            if np.sum(validos) > 20:
                rate_mira, rate_no_mira, _ = analizar_mirada_hacia_punto(
                    x_bins[validos], y_bins[validos], ang_bins_rad[validos], spikes[validos], tiempos_bin, x_target, y_target, tolerancia_deg
                )
                delta_R = rate_mira - rate_no_mira
            else:
                delta_R = 0.0
                
            delta_R_matrix[i, j] = delta_R
            
            if delta_R > best_delta_R:
                best_delta_R = delta_R
                best_coord = (x_target, y_target)
                
    return XX, YY, delta_R_matrix, best_coord, best_delta_R


def evaluar_confianza_punto_preferido(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, 
                                      grid_res=20, tolerancia_deg=30, n_shifts=100, 
                                      min_shift_sec=10.0, min_dist=5.0):
    """
    Evalúa la confianza estadística del punto de mirada preferido mediante un test de
    control basado en el desplazamiento circular de los spikes ('spike shifting').
    """
    # 1. Encontrar el punto preferido y selectividad original
    _, _, _, original_coord, original_delta_R = encontrar_punto_preferido_mirada(
        x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, grid_res, tolerancia_deg, min_dist
    )
    
    # 2. Precalcular máscaras de mirada en la grilla para acelerar el shuffle test
    x_min, x_max = np.min(x_bins), np.max(x_bins)
    y_min, y_max = np.min(y_bins), np.max(y_bins)
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    XX_mesh, YY_mesh = np.meshgrid(x_grid, y_grid)
    
    grid_masks = []
    for i in range(grid_res):
        for j in range(grid_res):
            x_target = XX_mesh[i, j]
            y_target = YY_mesh[i, j]
            
            dist = np.sqrt((x_bins - x_target)**2 + (y_bins - y_target)**2)
            validos = dist > min_dist
            
            if np.sum(validos) > 20:
                dx = x_target - x_bins[validos]
                dy = y_target - y_bins[validos]
                ang_h = np.arctan2(dy, dx)
                diff_rad = np.arctan2(np.sin(ang_bins_rad[validos] - ang_h), np.cos(ang_bins_rad[validos] - ang_h))
                diff_deg = np.abs(np.degrees(diff_rad))
                
                mira_al_punto = diff_deg <= tolerancia_deg
                
                mira_mask = np.zeros(len(x_bins), dtype=bool)
                valid_idx = np.where(validos)[0]
                mira_mask[valid_idx[mira_al_punto]] = True
                
                no_mira_mask = np.zeros(len(x_bins), dtype=bool)
                no_mira_mask[valid_idx[~mira_al_punto]] = True
                
                grid_masks.append((mira_mask, no_mira_mask, np.sum(mira_mask), np.sum(no_mira_mask)))
            else:
                grid_masks.append(None)
                
    # 3. Definir rangos válidos de shift circular en bines
    min_shift_bins = int(np.ceil(min_shift_sec / tiempos_bin))
    L = len(spikes)
    if L < 2 * min_shift_bins:
        raise ValueError(f"La duración de la sesión ({L} bines) es menor que el doble del shift mínimo ({2 * min_shift_bins} bines).")
        
    shuffled_delta_Rs = []
    
    # Generador aleatorio con seed fija para reproducibilidad
    rng = np.random.default_rng(42)
    valid_shifts = rng.integers(min_shift_bins, L - min_shift_bins, size=n_shifts)
    
    # Ejecutar test de control
    for s in valid_shifts:
        spikes_shuffled = np.roll(spikes, s)
        best_shuffled_val = -np.inf
        
        for mask_info in grid_masks:
            if mask_info is None:
                continue
            mira_mask, no_mira_mask, sum_mira, sum_no_mira = mask_info
            
            rate_mira = np.sum(spikes_shuffled[mira_mask]) / (sum_mira * tiempos_bin)
            rate_no_mira = np.sum(spikes_shuffled[no_mira_mask]) / (sum_no_mira * tiempos_bin)
            delta_R = rate_mira - rate_no_mira
            
            if delta_R > best_shuffled_val:
                best_shuffled_val = delta_R
                
        shuffled_delta_Rs.append(best_shuffled_val)
              
    shuffled_delta_Rs = np.array(shuffled_delta_Rs)
    
    # 4. Calcular métricas estadísticas de confianza
    p_value = np.sum(shuffled_delta_Rs >= original_delta_R) / n_shifts
    mean_shuff = np.mean(shuffled_delta_Rs)
    std_shuff = np.std(shuffled_delta_Rs) if np.std(shuffled_delta_Rs) > 0 else 1e-6
    z_score = (original_delta_R - mean_shuff) / std_shuff
    confianza_significativa = (p_value < 0.05) or (z_score > 1.96)
    
    return {
        'original_coord': original_coord,
        'original_delta_R': original_delta_R,
        'mean_shuffled_delta_R': mean_shuff,
        'std_shuffled_delta_R': std_shuff,
        'shuffled_delta_Rs': shuffled_delta_Rs,
        'p_value': p_value,
        'z_score': z_score,
        'confianza_significativa': confianza_significativa
    }


def evaluar_confianza_angulo_preferido(angulos_reales, spikes, tiempos_bin, tolerancia_deg=30, 
                                       n_shifts=100, min_shift_sec=10.0):
    ang_cand, delta_R, original_amp, original_angle, baseline, _ = calcular_angulo_preferido_coseno(
        angulos_reales, spikes, tiempos_bin, tolerancia_deg
    )
    
    # 2. Precalcular máscaras de dirección candidato para acelerar el test
    angulos_candidatos = np.arange(0, 360, 10)
    angle_masks = []
    for theta in angulos_candidatos:
        diff_angular = np.abs((angulos_reales - theta + 180) % 360 - 180)
        mira_hacia = diff_angular <= tolerancia_deg
        sum_mira = np.sum(mira_hacia)
        sum_no_mira = np.sum(~mira_hacia)
        angle_masks.append((mira_hacia, ~mira_hacia, sum_mira, sum_no_mira))
        
    # 3. Desplazamientos circulares
    min_shift_bins = int(np.ceil(min_shift_sec / tiempos_bin))
    L = len(spikes)
    if L < 2 * min_shift_bins:
        raise ValueError("Sesión muy corta para el shift mínimo.")
        
    shuffled_amplitudes = []
    
    # ¡CRÍTICO! Usar la misma matemática que en la función principal (Von Mises)
    def funcion_von_mises(x, A, kappa, theta_pref_rad, B):
        return A * np.exp(kappa * np.cos(np.radians(x) - theta_pref_rad)) + B
        
    rng = np.random.default_rng(42)
    valid_shifts = rng.integers(min_shift_bins, L - min_shift_bins, size=n_shifts)
    
    for s in valid_shifts:
        spikes_shuffled = np.roll(spikes, s)
        delta_R_shuff = []
        
        for mira_hacia, no_mira_hacia, sum_mira, sum_no_mira in angle_masks:
            if sum_mira > 0 and sum_no_mira > 0:
                rate_mira = np.sum(spikes_shuffled[mira_hacia]) / (sum_mira * tiempos_bin)
                rate_no_mira = np.sum(spikes_shuffled[no_mira_hacia]) / (sum_no_mira * tiempos_bin)
                delta_R_shuff.append(rate_mira - rate_no_mira)
            else:
                delta_R_shuff.append(0.0)
                
        delta_R_shuff = np.array(delta_R_shuff)
        
        # Ajuste libre para la distribución nula
        amp_emp = np.max(delta_R_shuff) - np.min(delta_R_shuff)
        p0 = [amp_emp, 1.0, np.radians(angulos_candidatos[np.argmax(delta_R_shuff)]), np.min(delta_R_shuff)]
        
        bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        
        try:
            popt, _ = curve_fit(funcion_von_mises, angulos_candidatos, delta_R_shuff, p0=p0, bounds=bounds)
            amp_shuff = popt[0] * np.exp(popt[1]) # Amplitud real del pico
        except Exception:
            amp_shuff = amp_emp * np.exp(1.0) # Fallback empírico
            
        shuffled_amplitudes.append(amp_shuff)
        
    shuffled_amplitudes = np.array(shuffled_amplitudes)
    
    # 4. Calcular métricas estadísticas
    p_value = np.sum(shuffled_amplitudes >= original_amp) / n_shifts
    mean_shuff = np.mean(shuffled_amplitudes)
    std_shuff = np.std(shuffled_amplitudes) if np.std(shuffled_amplitudes) > 0 else 1e-6
    z_score = (original_amp - mean_shuff) / std_shuff
    confianza_significativa = (p_value < 0.05) or (z_score > 1.96)
    
    return {
        'original_amplitude': original_amp,
        'original_angle': original_angle,
        'mean_shuffled_amplitude': mean_shuff,
        'std_shuffled_amplitude': std_shuff,
        'shuffled_amplitudes': shuffled_amplitudes,
        'p_value': p_value,
        'z_score': z_score,
        'confianza_significativa': confianza_significativa
    }