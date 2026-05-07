import h5py
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


import os
import statsmodels.api as sm

try:
    from pygam import PoissonGAM, te, s
except ImportError:
    pass

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
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    print(f"-> graficando s={sesion}, t={tetrodo}, c={neurona}. disparos: {len(tiempos_celula)}")
    
    ## 2.5 FILTRO DE VELOCIDAD (Speed Filter)
    # Interpolamos la velocidad exacta en el instante de cada spike
    vel_celula = np.interp(tiempos_celula, pos_t, vel)
    
    # Descartamos spikes de "Modo Replay/uieto"
    umbral_velocidad = 2.0 
    mask_movimiento = vel_celula > umbral_velocidad
    tiempos_celula = tiempos_celula[mask_movimiento]
    print(f"-> disparos tras filtro de velocidad (> {umbral_velocidad} cm/s): {len(tiempos_celula)}")

    # 3. interpolamos para saber la coordenada x,y aproximada en el microsegundo del disparo
    pos_x_celula = np.interp(tiempos_celula, pos_t, pos_x)
    pos_y_celula = np.interp(tiempos_celula, pos_t, pos_y)
    
    # centramos las coordenadas para que el campo vaya de 0 a 90 cm
    mid_x = (np.nanmax(pos_x) + np.nanmin(pos_x)) / 2
    mid_y = (np.nanmax(pos_y) + np.nanmin(pos_y)) / 2
    shift_x = 45 - mid_x
    shift_y = 45 - mid_y
    
    pos_x_plot = pos_x + shift_x
    pos_y_plot = pos_y + shift_y
    pos_x_celula_plot = pos_x_celula + shift_x
    pos_y_celula_plot = pos_y_celula + shift_y
    
    # 4. graficamos

    plt.figure(figsize=(9, 9))
    # caminito gris de fondo
    plt.plot(pos_x_plot, pos_y_plot, color='lightgray', linewidth=1)
    # puntitos rojos donde hubo spike
    plt.scatter(pos_x_celula_plot, pos_y_celula_plot, color='red', s=10, zorder=5)
    
    plt.title(f'firing map | sesion {sesion} | tetrodo {tetrodo} | celula {neurona}')
    
    plt.xlim(0, 90)
    plt.ylim(0, 90)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# --- ejemplos ---

# firing_map(sesion=2, tetrodo=3, neurona=3)

# firing_map(sesion=2, tetrodo=2, neurona=1)

# firing_map(sesion=2, tetrodo=2, neurona=7)

def preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec):
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    # 3. Binning Temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    # Y: Spikes por ventanita
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    # X: Posición por ventanita
    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    
    X = np.column_stack((x_bins, y_bins))
    Y = conteo_spikes
    
    return X, Y

def glm_posicion_manual(sesion, tetrodo, neurona, n_bines=45):
    print("\n--- INICIANDO GLM MANUAL DE POSICIÓN ---")
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec=0.1)
    
    pos_x = X[:, 0]
    pos_y = X[:, 1]
    
    print(f"Construyendo Basis Functions (Grilla de {n_bines}x{n_bines} Campanas Gauss)...")
    n_bases_x = n_bines
    n_bases_y = n_bines
    centros_x = np.linspace(np.min(pos_x), np.max(pos_x), n_bases_x)
    centros_y = np.linspace(np.min(pos_y), np.max(pos_y), n_bases_y)
    sigma_pos = (np.max(pos_x) - np.min(pos_x)) / n_bases_x
    
    # Matriz donde cada columna es una de las campanas
    X_bases_pos = np.zeros((len(pos_x), n_bases_x * n_bases_y))
    columna = 0
    for cx in centros_x:
        for cy in centros_y:
            dist_cuadrada = (pos_x - cx)**2 + (pos_y - cy)**2
            X_bases_pos[:, columna] = np.exp(- dist_cuadrada / (2 * sigma_pos**2))
            columna += 1
            
    # Agregamos constante y entrenamos
    X_glm_pos = sm.add_constant(X_bases_pos)
    
    # Solución: "Regularización" (Ridge)
    modelo = sm.GLM(Y, X_glm_pos, family=sm.families.Poisson()).fit_regularized(alpha=0.1, L1_wt=0.0)
    print("¡Modelo entrenado (y regularizado)!")

    # --- GRAFICAR ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    # Generamos los datos evaluando la matemática EXACTAMENTE en los centros de los bins
    x_grid = np.linspace(np.min(pos_x), np.max(pos_x), n_bines)
    y_grid = np.linspace(np.min(pos_y), np.max(pos_y), n_bines)
    XX, YY = np.meshgrid(x_grid, y_grid)
    
    X_test_pos = np.zeros((n_bines * n_bines, n_bases_x * n_bases_y))
    columna = 0
    for cx in centros_x:
        for cy in centros_y:
            dist_sq = (XX.flatten() - cx)**2 + (YY.flatten() - cy)**2
            X_test_pos[:, columna] = np.exp(- dist_sq / (2 * sigma_pos**2))
            columna += 1
            
    X_test_pos_const = sm.add_constant(X_test_pos, has_constant='add')
    prediccion_pos = modelo.predict(X_test_pos_const).reshape(n_bines, n_bines)
    
    # Graficamos usando pcolormesh para pintar los "cuadrados" rígidos sin suavizar
    # Le agregamos bordes negros para que los bines sean 100% distinguibles
    mesh = ax.pcolormesh(x_grid, y_grid, prediccion_pos, cmap='jet', shading='nearest', linewidth=0.5)
    fig.colorbar(mesh, ax=ax, label='Tasa de Disparo (Spikes/Bin)')
    ax.set_title(f'GLM Manual (Píxeles): {n_bines}x{n_bines} | s={sesion} t={tetrodo} c={neurona}')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.tight_layout()
    plt.show()

import pickle

def get_gam_posicion(sesion, tetrodo, neurona, splines=5, bin_size_sec=0.1, force_retrain=False):
    archivo_modelo = f'modelo_gam_pos_s{sesion}_t{tetrodo}_n{neurona}_sp{splines}.pkl'
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec)
    
    if os.path.exists(archivo_modelo) and not force_retrain:
        print(f"[+] Cargando GAM Posición guardado desde {archivo_modelo}...")
        with open(archivo_modelo, 'rb') as f:
            modelo_gam = pickle.load(f)
    else:
        print(f"[-] Entrenando GAM Posición ({splines}x{splines} splines)...")
        modelo_gam = PoissonGAM(te(0, 1, n_splines=splines)).gridsearch(X, Y, progress=False)
        with open(archivo_modelo, 'wb') as f:
            pickle.dump(modelo_gam, f)
            
    print("\n=== RESUMEN GAM POSICIÓN ===")
    modelo_gam.summary()
    return modelo_gam, X, Y

def graficar_gam_posicion(modelo_gam, X, Y, sesion, tetrodo, neurona, splines, bin_size_sec=0.1):
    print("\n--- GRAFICANDO GAM ---")
    XX_pos = modelo_gam.generate_X_grid(term=0, n=36)
    Z_pos = modelo_gam.partial_dependence(term=0, X=XX_pos)
    
    x_grid = XX_pos[:, 0].reshape(36, 36)
    y_grid = XX_pos[:, 1].reshape(36, 36)
    z_grid = Z_pos.reshape(36, 36)
    
    # 1. Crear una máscara de ocupancia basada en las posiciones reales (X)
    from scipy.spatial import cKDTree
    # Construimos un árbol KD con las posiciones por las que pasó el animal
    tree = cKDTree(X)
    # Buscamos la distancia desde cada punto del grid al punto real más cercano
    distancias, _ = tree.query(XX_pos)
    distancias = distancias.reshape(36, 36)
    
    # Si un punto del grid está a más de 5 cm de una pisada real, lo consideramos "no visitado"
    # y lo volvemos NaN para que matplotlib lo dibuje blanco.
    z_grid[distancias > 5.0] = np.nan
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    # 2.
    ax.set_facecolor('white')
    mesh = ax.pcolormesh(x_grid, y_grid, z_grid, cmap='jet', shading='nearest')
    fig.colorbar(mesh, ax=ax, label='Tasa de Disparo (Spikes/Bin)')
    ax.set_title(f'GAM Model | s={sesion} t={tetrodo} c={neurona}')
    ax.set_aspect('equal')
    ax.axis('off')
    
    prediccion_tiempo = modelo_gam.predict(X)
    
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    
    limite = min(10000, len(Y)) 
    tiempo_eje = np.arange(limite) * bin_size_sec
    
    ax2.bar(tiempo_eje, Y[:limite], width=bin_size_sec, color='black', alpha=0.6, label='Spikes Reales')
    ax2.plot(tiempo_eje, prediccion_tiempo[:limite], color='red', linewidth=2, label='Predicción Continua')
    
    ax2.set_xlabel('Tiempo (segundos)')
    ax2.set_ylabel('Cantidad de Spikes')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


###


def preparar_datos_viewpoint_1d(sesion, tetrodo, neurona, bin_size_sec=0.1):
    """
    Calcula el punto de intersección de la mirada y "desenrolla" la caja 
    en un perímetro 1D continuo (0 a Perímetro Total).
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    x_min, x_max = np.min(pos_x), np.max(pos_x)
    y_min, y_max = np.min(pos_y), np.max(pos_y)
    W = x_max - x_min
    H = y_max - y_min
    
    # angulo de la mirada
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    
    angulo = np.append(np.arctan2(dy, dx), 0)
    angulo = np.unwrap(angulo)
    
    # 4. Binning Temporal
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)
    
    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)
    
    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    ang_bins = np.interp(centros_bins, pos_t, angulo)
    vel_bins = np.interp(centros_bins, pos_t, vel)
    
    # 5. RAY CASTING (Intersección con la pared 2D)
    vx = np.cos(ang_bins)
    vy = np.sin(ang_bins)
    vx[vx == 0] = 1e-10
    vy[vy == 0] = 1e-10
    
    tx = np.where(vx > 0, (x_max - x_bins) / vx, (x_min - x_bins) / vx)
    ty = np.where(vy > 0, (y_max - y_bins) / vy, (y_min - y_bins) / vy)
    
    t_hit = np.minimum(tx, ty)
    mirada_x = x_bins + t_hit * vx
    mirada_y = y_bins + t_hit * vy
    
    # 6. DESENROLLAR LA CAJA A 1D
    # Perímetro P va de 0 a 2W + 2H
    p_mirada = np.zeros_like(mirada_x)
    tol = 1e-3 # Tolerancia flotante
    
    # Abajo (y_min) [0 a W]
    mask_bottom = np.abs(mirada_y - y_min) < tol
    p_mirada[mask_bottom] = mirada_x[mask_bottom] - x_min
    
    # Derecha (x_max) [W a W+H]
    mask_right = np.abs(mirada_x - x_max) < tol
    p_mirada[mask_right] = W + (mirada_y[mask_right] - y_min)
    
    # Arriba (y_max) [W+H a 2W+H] -> Derecha a izquierda
    mask_top = np.abs(mirada_y - y_max) < tol
    p_mirada[mask_top] = W + H + (x_max - mirada_x[mask_top])
    
    # Izquierda (x_min) [2W+H a 2W+2H] -> Arriba a abajo
    mask_left = np.abs(mirada_x - x_min) < tol
    p_mirada[mask_left] = 2*W + H + (y_max - mirada_y[mask_left])
    
    mascara_vel = vel_bins > 2.0
    
    X = p_mirada[mascara_vel].reshape(-1, 1)
    Y = conteo_spikes[mascara_vel]
    
    return X, Y, W, H


def get_gam_viewpoint_1d(sesion, tetrodo, neurona, splines=20, bin_size_sec=0.1, force_retrain=False):
    archivo_modelo = f'modelo_gam_vp1d_s{sesion}_t{tetrodo}_n{neurona}_sp{splines}.pkl'
    X, Y, W, H = preparar_datos_viewpoint_1d(sesion, tetrodo, neurona, bin_size_sec)
    
    if os.path.exists(archivo_modelo) and not force_retrain:
        print(f"[+] Cargando GAM Viewpoint 1D desde {archivo_modelo}...")
        with open(archivo_modelo, 'rb') as f:
            modelo_gam = pickle.load(f)
    else:
        print(f"[-] Entrenando GAM Cíclico Viewpoint 1D ({splines} splines)...")
        modelo_gam = PoissonGAM(s(0, basis='cp', n_splines=splines)).gridsearch(X, Y, progress=False)
        with open(archivo_modelo, 'wb') as f:
            pickle.dump(modelo_gam, f)
            
    print("\n=== RESUMEN GAM VIEWPOINT 1D ===")
    modelo_gam.summary()
    return modelo_gam, X, Y, W, H


def graficar_gam_viewpoint_1d(modelo_gam, X, Y, W, H, sesion, tetrodo, neurona):
    """
    Grafica el "mapa de pared" desenrollado.
    """
    perimetro_total = 2*W + 2*H
    
    XX_pred = np.linspace(0, perimetro_total, 200).reshape(-1, 1)
    YY_pred = modelo_gam.predict(XX_pred)
    intervalos = modelo_gam.confidence_intervals(XX_pred, width=.95)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Tasa Empírica (Barras grises de fondo)
    n_bins = 50
    bins_perimetro = np.linspace(0, perimetro_total, n_bins+1)
    spikes_sect, _ = np.histogram(X.flatten(), bins=bins_perimetro, weights=Y)
    tiempo_sect, _ = np.histogram(X.flatten(), bins=bins_perimetro)
    
    tiempo_sect = np.maximum(tiempo_sect, 1)
    tasa_cruda = spikes_sect / (tiempo_sect * 0.1)
    
    ax.bar(bins_perimetro[:-1], tasa_cruda, width=bins_perimetro[1]-bins_perimetro[0], 
            color='gray', alpha=0.3, align='edge')
    
    # Predicción GAM (Línea roja)
    ax.fill_between(XX_pred.flatten(), intervalos[:, 0], intervalos[:, 1], color='red', alpha=0.2)
    ax.plot(XX_pred.flatten(), YY_pred, color='red', linewidth=2)
    
    # Decoración para entender las paredes
    esquinas = [0, W, W+H, 2*W+H, perimetro_total]
    nombres_paredes = ['Pared Sur\n(Abajo)', 'Pared Este\n(Derecha)', 'Pared Norte\n(Arriba)', 'Pared Oeste\n(Izquierda)']
    
    for e in esquinas:
        ax.axvline(e, color='black', linestyle='--', linewidth=2)
        
    for i in range(4):
        midpoint = (esquinas[i] + esquinas[i+1]) / 2
        ax.text(midpoint, ax.get_ylim()[1]*0.9, nombres_paredes[i], 
                ha='center', va='top', fontsize=10, fontweight='bold', color='black')
        
    ax.set_xlim(0, perimetro_total)
    ax.set_title(f"Viewpoint Mapeado al Perímetro de la Caja (1D GAM Cíclico)\nS={sesion} T={tetrodo} C={neurona} | EDoF: {modelo_gam.statistics_['edof']:.1f}")
    ax.set_xlabel("Distancia a lo largo del perímetro (cm o unidades)")
    ax.set_ylabel("Frecuencia de Disparo Predicha")
    
    plt.tight_layout()
    plt.show()

def _cargar_datos_neurona(sesion, tetrodo, neurona):
    """
    funcion base de carga. lee hdf5, suaviza trayectoria, extrae spikes.
    devuelve datos crudos sin centrar ni filtrar por velocidad.
    """
    file = h5py.File(db_merged, 'r')
    
    # 1. trayectoria del degus
    nombre_pos = f'pos_{sesion}'
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    
    ## 1.5 filtro Gaussiano (suavizado de trayectoria)
    # Ventana biológica estándar: ~300 ms (0.3 segundos)
    # Suaviza el ruido del tracking de la cámara sin borrar movimientos reales del animal.
    dt_video = np.mean(np.diff(pos_t)) # segundos por frame
    sigma_frames = 0.3 / dt_video
    
    pos_x = gaussian_filter1d(pos_x, sigma=sigma_frames)
    pos_y = gaussian_filter1d(pos_y, sigma=sigma_frames)

    ## CALCULO DE VELOCIDAD
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

def rate_map(sesion, tetrodo, neurona, n_bins=36, smooth_sigma=1.5, min_tiempo_seg=0.1):
    """    
    1. Divide el espacio en una grilla de n_bins x n_bins.
    2. Cuenta spikes por bin y tiempo de ocupación por bin.
    3. Calcula tasa = spikes / tiempo (Hz).
    4. Aplica suavizado gaussiano espacial.
    5. Enmascara en blanco las zonas que el animal no visitó.
    
    Parámetros:
        n_bins:          Cantidad de cuadraditos por lado (default 36 -> ~2.5 cm/bin en caja de 90cm).
        smooth_sigma:    Sigma del filtro gaussiano 2D aplicado al rate map (en bins).
        min_tiempo_seg:  Tiempo mínimo de ocupación para considerar un bin como "visitado".
    """
    from scipy.ndimage import gaussian_filter
    
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(sesion, tetrodo, neurona)
    
    # posiciones interpoladas de cada spike
    spk_x = np.interp(tiempos_celula, pos_t, pos_x)
    spk_y = np.interp(tiempos_celula, pos_t, pos_y)
    
    # speed filter
    vel_spk = np.interp(tiempos_celula, pos_t, vel)
    mask_mov = vel_spk > 2.0
    spk_x = spk_x[mask_mov]
    spk_y = spk_y[mask_mov]
    
    # centrar a 0-90 cm
    mid_x = (np.nanmax(pos_x) + np.nanmin(pos_x)) / 2
    mid_y = (np.nanmax(pos_y) + np.nanmin(pos_y)) / 2
    shift_x = 45 - mid_x
    shift_y = 45 - mid_y
    pos_x = pos_x + shift_x
    pos_y = pos_y + shift_y
    spk_x = spk_x + shift_x
    spk_y = spk_y + shift_y
    
    # definimos los bordes de la grilla — más ancha que la trayectoria para
    # que aparezcan bins vacíos (blancos) en los bordes como en el paper
    pad = 10  # cm extra a cada lado
    bin_size = 90.0 / n_bins  # tamaño de cada cuadradito en cm
    bordes = np.arange(-pad, 90 + pad + bin_size, bin_size)
    
    # 1. mapa de ocupación: cuánto tiempo pasó el animal en cada bin
    ocup_counts, _, _ = np.histogram2d(pos_x, pos_y, bins=[bordes, bordes])
    ocup_tiempo = ocup_counts * dt_video  # convertir frames a segundos
    
    # 2. mapa de spikes: cuántos spikes cayeron en cada bin
    spk_counts, _, _ = np.histogram2d(spk_x, spk_y, bins=[bordes, bordes])
    
    # 3. suavizar ambos mapas antes de dividir
    ocup_suave = gaussian_filter(ocup_tiempo.astype(float), sigma=smooth_sigma)
    spk_suave  = gaussian_filter(spk_counts.astype(float),  sigma=smooth_sigma)
    
    # 4. calcular tasa de disparo (Hz) = spikes / tiempo
    rate = np.zeros_like(ocup_suave)
    mask_visitado = ocup_suave > min_tiempo_seg
    rate[mask_visitado] = spk_suave[mask_visitado] / ocup_suave[mask_visitado]
    
    # 5. NaN se dibujan blancos
    rate[~mask_visitado] = np.nan
    
    max_rate = np.nanmax(rate)
    
    # 6. graficar
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # firing map
    ax1 = axes[0]
    ax1.plot(pos_x, pos_y, color='black', linewidth=0.3, alpha=0.6)
    ax1.scatter(spk_x, spk_y, color='red', s=8, zorder=5, linewidths=0)
    ax1.set_xlim(bordes[0], bordes[-1]); ax1.set_ylim(bordes[0], bordes[-1])
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('90 cm', fontsize=12, pad=8)
    
    # rate map
    ax2 = axes[1]
    ax2.set_facecolor('white')
    mesh = ax2.pcolormesh(bordes, bordes, rate.T, cmap='jet', vmin=0, vmax=max_rate, shading='flat')
    ax2.set_xlim(bordes[0], bordes[-1]); ax2.set_ylim(bordes[0], bordes[-1])
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Max. {max_rate:.1f} Hz', fontsize=12, pad=8)
    
    # colorbar
    cbar = fig.colorbar(mesh, ax=ax2, fraction=0.03, pad=0.02, ticks=[0, max_rate])
    cbar.ax.set_yticklabels(['0', f'{max_rate:.1f}'])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    plt.show()
