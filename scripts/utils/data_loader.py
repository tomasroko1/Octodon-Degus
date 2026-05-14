"""
carga y preparación de datos
======================================
métodos:
- _cargar_datos_neurona: lee hdf5, extrae spikes, suaviza trayectorias y calcula velocidad.
- preparar_datos_posicion: genera bins temporales de posición (x, y) y conteo de spikes.
- preparar_datos_viewpoint_1d: mapea la dirección de mirada a los bordes de la caja (perímetro 1d).
"""
import os
import h5py
import scipy.io as sio
import numpy as np
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

db_merged = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V_merged.db')
db_clnew = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V.db_clnew')

all_clust = sio.loadmat(db_clnew)['all_clust']

def _cargar_datos_neurona(sesion, tetrodo, neurona):
    """
    funcion base de carga. lee hdf5, extrae spikes.
    devuelve datos crudos sin centrar ni filtrar por velocidad.
    """
    file = h5py.File(db_merged, 'r')
    
    # 1. trayectoria del degus
    nombre_pos = f'pos_{sesion}'
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    
    dt_video = np.mean(np.diff(pos_t)) # segundos por frame

    print('DEBUG TOMI', dt_video)
    ## calculo de velocidad
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.maximum(np.diff(pos_t), 1e-6) # evitamos div x0
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)
    
    # 2. sacamos los spikes de esta neurona
    nombre_spk = f'spk_ts_{sesion}_{tetrodo}'
    spikes = np.array(file[nombre_spk][0]).flatten()
    
    cluster = all_clust[sesion-1][tetrodo-1]
    indices_celula = cluster[0][neurona-1].flatten().astype(int) - 1
    tiempos_celula = spikes[indices_celula]

    file.close()
    
    return pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula

def _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video):
    """
    aplica filtro gaussiano a la trayectoria para eliminar el ruido del tracking 
    y recalcula la velocidad con las posiciones suavizadas.
    """
    from scipy.ndimage import gaussian_filter1d
    sigma_frames = 1
    pos_x = gaussian_filter1d(pos_x, sigma=sigma_frames)
    pos_y = gaussian_filter1d(pos_y, sigma=sigma_frames)
    
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.maximum(np.diff(pos_t), 1e-6)
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)
    
    return pos_x, pos_y, vel

def preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec):
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    pos_x, pos_y, vel = _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video)
    
    # 3. binning temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    # y: spikes por ventanita
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    # x: posición por ventanita
    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    
    X = np.column_stack((x_bins, y_bins))
    Y = conteo_spikes
    
    return X, Y

def preparar_datos_viewpoint_1d(sesion, tetrodo, neurona, bin_size_sec=0.1):
    """
    calcula el punto de intersección de la mirada y "desenrolla" la caja 
    en un perímetro 1d continuo (0 a perímetro total).
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    pos_x, pos_y, vel = _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video)
    
    x_min, x_max = np.min(pos_x), np.max(pos_x)
    y_min, y_max = np.min(pos_y), np.max(pos_y)
    W = x_max - x_min
    H = y_max - y_min
    
    # angulo de la mirada
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    
    angulo = np.append(np.arctan2(dy, dx), 0)
    angulo = np.unwrap(angulo)
    
    # 4. binning temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    ang_bins = np.interp(centros_bins, pos_t, angulo)
    vel_bins = np.interp(centros_bins, pos_t, vel)
    
    # 5. ray casting (intersección con la pared 2d)
    vx = np.cos(ang_bins)
    vy = np.sin(ang_bins)
    vx[vx == 0] = 1e-10
    vy[vy == 0] = 1e-10
    
    tx = np.where(vx > 0, (x_max - x_bins) / vx, (x_min - x_bins) / vx)
    ty = np.where(vy > 0, (y_max - y_bins) / vy, (y_min - y_bins) / vy)
    
    t_hit = np.minimum(tx, ty)
    mirada_x = x_bins + t_hit * vx
    mirada_y = y_bins + t_hit * vy
    
    # 6. desenrollar la caja a 1d
    # perímetro p va de 0 a 2w + 2h
    p_mirada = np.zeros_like(mirada_x)
    tol = 1e-3 # tolerancia flotante
    
    # abajo (y_min) [0 a w]
    mask_bottom = np.abs(mirada_y - y_min) < tol
    p_mirada[mask_bottom] = mirada_x[mask_bottom] - x_min
    
    # derecha (x_max) [w a w+h]
    mask_right = np.abs(mirada_x - x_max) < tol
    p_mirada[mask_right] = W + (mirada_y[mask_right] - y_min)
    
    # arriba (y_max) [w+h a 2w+h] -> derecha a izquierda
    mask_top = np.abs(mirada_y - y_max) < tol
    p_mirada[mask_top] = W + H + (x_max - mirada_x[mask_top])
    
    # izquierda (x_min) [2w+h a 2w+2h] -> arriba a abajo
    mask_left = np.abs(mirada_x - x_min) < tol
    p_mirada[mask_left] = 2*W + H + (y_max - mirada_y[mask_left])
    
    mascara_vel = vel_bins > 2.0
    
    X = p_mirada[mascara_vel].reshape(-1, 1)
    Y = conteo_spikes[mascara_vel]
    
    return X, Y, W, H
