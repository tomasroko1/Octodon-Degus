import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


import os

# Configuración de rutas (apuntando a la carpeta data en la raíz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

db_merged = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V_merged.db')
db_clnew = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V.db_clnew')

all_clust = sio.loadmat(db_clnew)['all_clust']
## lista anidada tipo 'cell' de matlab. se accede: all_clust[sesion][tetrodo][0][celula] -> devuelve un array de índices.
## esos indices son la posición en la lista de spikes de los disparon asociados a esa celula.

def firing_map(sesion, tetrodo, neurona):
    """
    Devuelve el mapa con el recorrido y los lugares de disparo
    de una neurona, en un tetrodo, en una sesión.
    """
    
    file = h5py.File(db_merged, 'r')
    
    # 1. trayectoria del degus
    nombre_pos = 'pos_' + str(sesion)
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    
    ## 1.5 filtro Gaussiano (suavizado de trayectoria)

    # Ventana biológica estándar: ~300 ms (0.3 segundos) ?? REVISAR SIGNIFICADO
    
    dt_video = np.mean(np.diff(pos_t)) # Segundos por frame
    sigma_frames = 0.3 / dt_video
    
    pos_x = gaussian_filter1d(pos_x, sigma=sigma_frames)
    pos_y = gaussian_filter1d(pos_y, sigma=sigma_frames)

    ## CALCULO DE VELOCIDAD
    dx = np.diff(pos_x) # [x2-x1, x3-x2, ........, x_n-1 - x_n]
    dy = np.diff(pos_y) # [y2-y1, y3-y2, ........, y_n-1 - y_n]
    dt = np.maximum(np.diff(pos_t), 1e-6) # evitamos div x0
    # calculamos velocidad y le agregamos un 0 al final para mantener el tamaño del array pos_t
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)
    
    # 2. sacamos todos los spikes grabados por este tetrodo
    
    ## spike timestamps -> TODOS los spikes. después toca separar que spikes son de c/cel. para eso usamos el 2do file despues.
    nombre_spk = 'spk_ts_' + str(sesion) + '_' + str(tetrodo)
    spikes = np.array(file[nombre_spk][0]).flatten()
    
    idx_sesion = sesion - 1
    idx_tetrodo = tetrodo - 1
    idx_celula = neurona - 1
    
    cluster = all_clust[idx_sesion][idx_tetrodo]
    indices_celula = cluster[0][idx_celula].flatten().astype(int) - 1
    
    # nos quedamos solo con los tiempos de disparo de nuestra celula -> 'S2020_MarkIX-OF-V.db_clnew'
    tiempos_celula = spikes[indices_celula]
    print(f"-> graficando s={sesion}, t={tetrodo}, c={neurona}. disparos: {len(tiempos_celula)}")
    
    ## 2.5 FILTRO DE VELOCIDAD (Speed Filter)
    # Interpolamos la velocidad exacta en el instante de cada spike
    vel_celula = np.interp(tiempos_celula, pos_t, vel)
    
    # Descartamos spikes de "Modo Replay/Quieto" (e.g. < 2 cm/s)
    umbral_velocidad = 2.0 
    mask_movimiento = vel_celula > umbral_velocidad
    tiempos_celula = tiempos_celula[mask_movimiento]
    print(f"-> disparos tras filtro de velocidad (> {umbral_velocidad} u/s): {len(tiempos_celula)}")

    # 3. interpolamos para saber la coordenada x,y exacta en el microsegundo del disparo
    pos_x_celula = np.interp(tiempos_celula, pos_t, pos_x)
    pos_y_celula = np.interp(tiempos_celula, pos_t, pos_y)
    
    # 4. graficamos

    plt.figure(figsize=(9, 9))
    # caminito gris de fondo
    plt.plot(pos_x, pos_y, color='lightgray', linewidth=1)
    # puntitos rojos donde hubo spike
    plt.scatter(pos_x_celula, pos_y_celula, color='red', s=10, zorder=5)
    
    plt.title(f'firing map | sesion {sesion} | tetrodo {tetrodo} | celula {neurona}')
    plt.axis('equal') # que quede cuadradito
    plt.show()
    
    file.close()

# --- ejemplos ---

firing_map(sesion=2, tetrodo=3, neurona=3)

# firing_map(sesion=2, tetrodo=2, neurona=1)

# firing_map(sesion=2, tetrodo=2, neurona=7)