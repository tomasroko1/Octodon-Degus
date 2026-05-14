"""
métodos:
- firing_map: grafica el recorrido del animal (gris) con los puntos de disparo superpuestos (rojo).
- rate_map: genera un mapa de calor suavizado espacialmente (hz) enmascarando zonas no visitadas.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
try:
    from .data_loader import _cargar_datos_neurona, _suavizar_posicion_y_velocidad
except ImportError:
    from data_loader import _cargar_datos_neurona, _suavizar_posicion_y_velocidad

def firing_map(sesion, tetrodo, neurona):
    """
    Devuelve el mapa con el recorrido y los lugares de disparo
    de una neurona, en un tetrodo, en una sesión.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)

    # filtro gaussiano preguntar.. -> no se si estos x,y ya estan procesados,
    #                                 o son datos crudos. no se si puede haber ruido, etc.
    pos_x, pos_y, vel = _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video)
    
    print(f"-> graficando s={sesion}, t={tetrodo}, c={neurona}. disparos: {len(tiempos_celula)}")
    
    ## filtro de velocidad (speed filter) preguntar..
    # interpolamos la velocidad exacta en el instante de cada spike
    vel_celula = np.interp(tiempos_celula, pos_t, vel)
    
    # descartamos spikes de "modo quieto" preguntar..
    umbral_velocidad = 2
    mask_movimiento = vel_celula > umbral_velocidad
    tiempos_celula = tiempos_celula[mask_movimiento]
    print(f"-> disparos tras filtro de velocidad (> {umbral_velocidad} cm/s): {len(tiempos_celula)}")

    # 3. interpolamos para saber la coordenada x,y aproximada en el microsegundo del disparo
    pos_x_celula = np.interp(tiempos_celula, pos_t, pos_x)
    pos_y_celula = np.interp(tiempos_celula, pos_t, pos_y)
    
    # 4. graficamos

    plt.figure(figsize=(9, 9))
    # camino gris de fondo
    plt.plot(pos_x, pos_y, color='lightgray', linewidth=1)
    # puntos rojos por spike
    plt.scatter(pos_x_celula, pos_y_celula, color='red', s=10, zorder=5)
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    plt.title(f'firing map | sesion {sesion} | tetrodo {tetrodo} | celula {neurona}')
        
    plt.show()

def rate_map(sesion, tetrodo, neurona, bin_size_cm=2.5, sigma_cm=4, min_tiempo_seg=0.1):
    """    
    1. carga y suaviza datos crudos.
    2. aplica filtro de velocidad (>2 cm/s) a trayectoria y spikes.
    3. construye histogramas 2d para ocupación y spikes.
    4. suaviza cada mapa independientemente y los divide para obtener la tasa (hz).
    """
    
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    pos_x, pos_y, vel = _suavizar_posicion_y_velocidad(pos_x, pos_y, pos_t, dt_video)
    
    # interpolar posiciones y velocidades de los spikes
    spk_x = np.interp(tiempos_celula, pos_t, pos_x)
    spk_y = np.interp(tiempos_celula, pos_t, pos_y)
    vel_spk = np.interp(tiempos_celula, pos_t, vel)
    
    # guardar trayectoria completa antes del filtro de velocidad
    pos_x_all, pos_y_all = pos_x.copy(), pos_y.copy()
    
    # filtro de velocidad (descartar puntos quietos)
    mask_mov = vel > 2
    pos_x, pos_y = pos_x[mask_mov], pos_y[mask_mov]
    
    mask_spk = vel_spk > 2
    spk_x, spk_y = spk_x[mask_spk], spk_y[mask_spk]
    
    # tamaño del campo
    inicio_x = np.nanmin(pos_x_all)
    inicio_y = np.nanmin(pos_y_all)
    bordes_x = np.arange(inicio_x, inicio_x + 90 + bin_size_cm, bin_size_cm)
    bordes_y = np.arange(inicio_y, inicio_y + 90 + bin_size_cm, bin_size_cm)
    
    # tiempo exacto transcurrido en cada frame
    dt_exacto = np.append(np.diff(pos_t), dt_video)
    
    # filtramos los dt_exacto segun la mask_mov
    dt_mov = dt_exacto[mask_mov]
    
    # histogramas crudos (ocupación pesada por el tiempo exacto)
    ocup, _, _ = np.histogram2d(pos_x, pos_y, bins=[bordes_x, bordes_y], weights=dt_mov)
    spk, _, _ = np.histogram2d(spk_x, spk_y, bins=[bordes_x, bordes_y])
    
    # zona visitada (toda la trayectoria, sin filtro de vel)
    visitado, _, _ = np.histogram2d(pos_x_all, pos_y_all, bins=[bordes_x, bordes_y])
    zona_visitada = visitado > 0
    
    # suavizado gaussiano (sigma = 1.6 bins)
    sigma_bins = sigma_cm / bin_size_cm
    ocup_s = gaussian_filter(ocup, sigma=sigma_bins, mode='constant', cval=0)
    spk_s = gaussian_filter(spk, sigma=sigma_bins, mode='constant', cval=0)
    
    # tasa de disparo (Hz)
    rate = np.zeros_like(ocup_s)
    valido = ocup_s > 0.0001
    rate[valido] = spk_s[valido] / ocup_s[valido]
    
    # blanco donde el animal nunca pasó
    rate[~zona_visitada] = np.nan
    max_rate = np.nanmax(rate)
    
    # grafico
    rango_x = bordes_x[-1] - bordes_x[0]
    rango_y = bordes_y[-1] - bordes_y[0]
    escala = 6 / max(rango_x, rango_y)
    fig, ax = plt.subplots(figsize=(rango_x * escala + 1, rango_y * escala))
    ax.set_facecolor('white')
    mesh = ax.pcolormesh(bordes_x, bordes_y, rate.T, cmap='jet', vmin=0, vmax=max_rate, shading='flat')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Rate Map | Max. {max_rate:.1f} Hz', fontsize=12, pad=10)
    
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.04, ticks=[0, max_rate])
    cbar.ax.set_yticklabels(['0', f'{max_rate:.1f} Hz'])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.show()
