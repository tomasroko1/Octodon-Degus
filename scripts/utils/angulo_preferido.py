import numpy as np
from scipy.optimize import curve_fit
import sys
import os
import matplotlib.pyplot as plt

def calcular_angulo_preferido_coseno(angulos_reales, spikes, tiempos_bin, tolerancia_deg=30):
    """
    Calcula el ángulo preferido y la amplitud mediante ajuste de coseno (Head Direction clásico).
    Eje X: Ángulo absoluto de la cabeza con respecto a la habitación (0-360°).
    
    Args:
        angulos_reales: array de la dirección de la mirada en cada instante (en grados, 0-360).
        spikes: array con el conteo de spikes en cada instante.
        tiempos_bin: duración de cada bin en segundos.
        tolerancia_deg: ventana de grados para considerar que mira hacia esa dirección.
    """
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
            delta_R.append(0)
            
    delta_R = np.array(delta_R)
    
    def funcion_coseno(x, A, theta_pref_rad, B):
        return A * np.cos(np.radians(x) - theta_pref_rad) + B
    
    p0 = [ (np.max(delta_R) - np.min(delta_R))/2, np.radians(angulos_candidatos[np.argmax(delta_R)]), np.mean(delta_R) ]
    popt, _ = curve_fit(funcion_coseno, angulos_candidatos, delta_R, p0=p0)
    
    amplitud_optima = popt[0]
    angulo_preferido_deg = np.degrees(popt[1]) % 360
    baseline = popt[2]
    
    return angulos_candidatos, delta_R, amplitud_optima, angulo_preferido_deg, funcion_coseno


def analizar_mirada_hacia_punto(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, x_target, y_target, tolerancia_deg=30):
    """
    Calcula la tasa de disparo de la neurona cuando el animal mira hacia un punto específico 
    (x_target, y_target) en el mapa 2D, en comparación con cuando mira hacia otros lados.
    
    Args:
        x_bins, y_bins: coordenadas 2D del animal en cada bin.
        ang_bins_rad: dirección de la cabeza (head direction) en radianes en cada bin.
        spikes: conteo de spikes en cada bin.
        tiempos_bin: duración del bin en segundos (ej. 0.1).
        x_target, y_target: coordenadas del punto objetivo en la caja (ej. una esquina o el centro).
        tolerancia_deg: tolerancia angular en grados.
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


def encontrar_punto_preferido_mirada(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, grid_res=20, tolerancia_deg=30):
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
            
            # Filtro: Omitimos evaluar bines cuando el animal está encima del punto analizado (<5 cm)
            # porque el ángulo de mirada se vuelve inestable matemáticamente
            dist_al_punto = np.sqrt((x_bins - x_target)**2 + (y_bins - y_target)**2)
            validos = dist_al_punto > 5.0
            
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