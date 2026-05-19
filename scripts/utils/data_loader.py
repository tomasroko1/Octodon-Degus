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
    
    dt_video = np.mean(np.diff(pos_t)) # segundos por frame ~0.02

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

def preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec):
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    # pos_x, pos_y, vel = _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video)
    
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

def preparar_datos_head_direction(sesion, tetrodo, neurona, bin_size_sec=0.1, min_vel=2.0):
    """
    carga y prepara la dirección de la cabeza (head direction, hd) y los spikes correspondientes.
    
    args:
        sesion, tetrodo, neurona: identificadores de la neurona.
        bin_size_sec: tamaño del bin temporal (por defecto 0.1s).
        min_vel: velocidad mínima para filtrar la actividad estática (por defecto 2.0 cm/s).
        
    returns:
        ang_bins_deg: array de la dirección de la cabeza en grados (0 a 360).
        conteo_spikes: array con el conteo de spikes en cada bin de tiempo.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    # cargar head direction (hd) real desde la base de datos
    file = h5py.File(db_merged, 'r')
    angulo = np.array(file[f'pos_{sesion}']['hd']).flatten()
    file.close()
    
    # el tracking puede tener ángulos con wrap en -pi/pi, los unwrap-eamos para interpolar bien
    angulo = np.unwrap(angulo)
    
    # binning temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    # interpolamos los ángulos y velocidades a los centros de los bins
    ang_bins = np.interp(centros_bins, pos_t, angulo)
    vel_bins = np.interp(centros_bins, pos_t, vel)
    
    # convertimos a grados y envolvemos en [0, 360)
    ang_bins_deg = np.degrees(ang_bins) % 360
    
    # filtramos por velocidad mínima si se especifica
    if min_vel > 0:
        mascara_vel = vel_bins > min_vel
        ang_bins_deg = ang_bins_deg[mascara_vel]
        conteo_spikes = conteo_spikes[mascara_vel]
        
    return ang_bins_deg, conteo_spikes

def preparar_datos_mirada(sesion, tetrodo, neurona, bin_size_sec=0.1, min_vel=2.0):
    """
    carga y prepara la trayectoria 2D del animal, su dirección de cabeza (HD) en radianes
    y los spikes correspondientes.
    
    args:
        sesion, tetrodo, neurona: identificadores de la neurona.
        bin_size_sec: tamaño del bin temporal.
        min_vel: velocidad mínima para filtrar la actividad estática.
        
    returns:
        x_bins: array de la posición X del animal por bin.
        y_bins: array de la posición Y del animal por bin.
        ang_bins: array de la dirección de la cabeza en radianes por bin.
        conteo_spikes: array con el conteo de spikes en cada bin.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    # cargar head direction (hd) real desde la base de datos
    file = h5py.File(db_merged, 'r')
    angulo = np.array(file[f'pos_{sesion}']['hd']).flatten()
    file.close()
    
    # unwrap para interpolación correcta
    angulo = np.unwrap(angulo)
    
    # binning temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    # interpolamos los ángulos, posiciones y velocidades a los centros de los bins
    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    ang_bins = np.interp(centros_bins, pos_t, angulo)
    vel_bins = np.interp(centros_bins, pos_t, vel)
    
    # filtramos por velocidad mínima si se especifica
    if min_vel > 0:
        mascara_vel = vel_bins > min_vel
        x_bins = x_bins[mascara_vel]
        y_bins = y_bins[mascara_vel]
        ang_bins = ang_bins[mascara_vel]
        conteo_spikes = conteo_spikes[mascara_vel]
        
    return x_bins, y_bins, ang_bins, conteo_spikes
