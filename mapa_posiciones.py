import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


db_merged = 'S2020_MarkIX-OF-V_merged.db'
db_clnew = 'S2020_MarkIX-OF-V.db_clnew'

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


# TODO: Este analisis solo nos sirve para el Open Field porque proyecta directo contra la pared
# podemos omitir esta función
def viewpoint_map(sesion, tetrodo, neurona):
    """
    Analiza el Punto de Vista (Head Direction / Viewpoint Analysis).
    Asume que el degú mira hacia la dirección de su movimiento.
    Traza un vector desde el degú hasta chocar con los bordes de la caja.
    """
    file = h5py.File(db_merged, 'r')
    
    # 1. Trayectoria y Suavizado
    nombre_pos = 'pos_' + str(sesion)
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    
    dt_video = np.mean(np.diff(pos_t))
    sigma_frames = 0.3 / dt_video
    pos_x = gaussian_filter1d(pos_x, sigma=sigma_frames)
    pos_y = gaussian_filter1d(pos_y, sigma=sigma_frames)
    
    # 2. Velocidad y Dirección (Ángulo)
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.maximum(np.diff(pos_t), 1e-6)
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)
    
    # Ángulo de movimiento (Arcotangente). Repetimos el último para alinear tamaños.
    angulo = np.append(np.arctan2(dy, dx), 0)
    # Desenvolvemos el ángulo (evitar salto brusco de 360 a 0) para no arruinar la interpolación
    angulo = np.unwrap(angulo)
    
    # 3. Spikes y Filtro de Velocidad
    nombre_spk = 'spk_ts_' + str(sesion) + '_' + str(tetrodo)
    spikes = np.array(file[nombre_spk][0]).flatten()
    
    cluster = all_clust[sesion-1][tetrodo-1]
    indices_celula = cluster[0][neurona-1].flatten().astype(int) - 1
    tiempos_celula = spikes[indices_celula]
    
    vel_celula = np.interp(tiempos_celula, pos_t, vel)
    tiempos_celula = tiempos_celula[vel_celula > 2.0] # Speed filter
    
    # 4. Interpolación
    pos_x_celula = np.interp(tiempos_celula, pos_t, pos_x)
    pos_y_celula = np.interp(tiempos_celula, pos_t, pos_y)
    angulo_celula = np.interp(tiempos_celula, pos_t, angulo)
    
    # 5. RAY CASTING VECTORIZADO (Cálculo de Colisión Geométrico)
    # Encontramos los bordes de la caja según el recorrido máximo del degú
    x_min, x_max = np.min(pos_x), np.max(pos_x)
    y_min, y_max = np.min(pos_y), np.max(pos_y)
    
    # Vectores direccionales de la mirada
    vx = np.cos(angulo_celula)
    vy = np.sin(angulo_celula)
    
    # Evitamos dividir por cero absoluto
    vx[vx == 0] = 1e-10
    vy[vy == 0] = 1e-10
    
    # Matemática de colisión: Cuánto tarda el rayo en chocar contra cada pared
    tx = np.where(vx > 0, (x_max - pos_x_celula) / vx, (x_min - pos_x_celula) / vx)
    ty = np.where(vy > 0, (y_max - pos_y_celula) / vy, (y_min - pos_y_celula) / vy)
    
    # El choque real es el que ocurre primero (el tiempo mínimo)
    t_hit = np.minimum(tx, ty)
    
    # Coordenadas exactas del impacto visual en la pared de la caja
    mirada_x = pos_x_celula + t_hit * vx
    mirada_y = pos_y_celula + t_hit * vy
    
    # 6. GRAFICAMOS
    fig = plt.figure(figsize=(14, 6))
    
    # Subgráfico 1: El mapa con los impactos visuales
    ax1 = fig.add_subplot(121)
    ax1.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='black', lw=2)
    ax1.plot(pos_x, pos_y, color='lightgray', linewidth=0.5, alpha=0.5)
    
    # Dibujamos al degú (azul) y dónde chocó su mirada (rojo)
    ax1.scatter(pos_x_celula, pos_y_celula, color='blue', s=5, alpha=0.5, label='Posición Degú')
    ax1.scatter(mirada_x, mirada_y, color='red', s=20, label='Punto de Vista (Pared)')
    
    # Trazamos algunos "rayos láser" visuales transparentes para entenderlo
    step = max(1, len(pos_x_celula) // 50) 
    for i in range(0, len(pos_x_celula), step):
        ax1.plot([pos_x_celula[i], mirada_x[i]], [pos_y_celula[i], mirada_y[i]], color='red', alpha=0.1, linewidth=1)
        
    ax1.set_title(f'Viewpoint Map | s={sesion} t={tetrodo} c={neurona}')
    ax1.axis('equal')
    ax1.legend()
    
    # Subgráfico 2: Histograma Polar (Rosa de los vientos)
    ax2 = fig.add_subplot(122, polar=True)
    angulos_rad = angulo_celula % (2 * np.pi)  # Volvemos a restringir a 0-360 grados
    bins = np.linspace(0, 2*np.pi, 36) # 36 sectores (cada 10 grados)
    counts, _ = np.histogram(angulos_rad, bins=bins)
    
    ax2.bar(bins[:-1], counts, width=bins[1]-bins[0], color='red', alpha=0.6, edgecolor='black')
    ax2.set_title('Sintonía de Dirección (Head Direction)')
    
    plt.tight_layout()
    plt.show()
    file.close()

# --- ejemplos ---

firing_map(sesion=2, tetrodo=3, neurona=3)

# firing_map(sesion=3, tetrodo=4, neurona=1)

# firing_map(sesion=2, tetrodo=2, neurona=7)