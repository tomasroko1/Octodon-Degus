import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
try:
    from pygam import PoissonGAM, te
except ImportError:
    pass

import os

# Configuración de rutas (apuntando a la carpeta data en la raíz)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

db_merged = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V_merged.db')
db_clnew = os.path.join(DATA_DIR, 'S2020_MarkIX-OF-V.db_clnew')
all_clust = sio.loadmat(db_clnew)['all_clust']

def preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec=0.1):
    file = h5py.File(db_merged, 'r')
    
    # 1. Trayectoria suavizada
    nombre_pos = f'pos_{sesion}'
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    
    dt_video = np.mean(np.diff(pos_t))
    sigma_frames = 0.3 / dt_video
    pos_x = gaussian_filter1d(pos_x, sigma=sigma_frames)
    pos_y = gaussian_filter1d(pos_y, sigma=sigma_frames)
    
    # 2. Spikes
    nombre_spk = f'spk_ts_{sesion}_{tetrodo}'
    spikes = np.array(file[nombre_spk][0]).flatten()
    cluster = all_clust[sesion-1][tetrodo-1]
    indices_celula = cluster[0][neurona-1].flatten().astype(int) - 1
    tiempos_celula = spikes[indices_celula]
    
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
    
    file.close()
    return X, Y

def glm_posicion_manual(sesion, tetrodo, neurona, n_bines=8):
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
    
    # ¡AQUÍ ESTABA EL PROBLEMA!
    # Si el degú nunca pisó la esquina inferior derecha, esa "campana" siempre vale 0 en los datos.
    # Un GLM puro se vuelve loco y le asigna un peso de 1,000,000 a esa esquina vacía.
    # Solución: "Regularización" (Ridge). Le decimos al modelo: "Si no tienes datos, mantén el peso cerca de 0".
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

def gam_posicion_automatico(sesion, tetrodo, neurona, splines):
    print("\n--- INICIANDO GAM AUTOMÁTICO DE POSICIÓN ---")
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec=0.1)
    
    print("Entrenando GAM con Splines Tensoriales...")
    print("El modelo buscará automáticamente la suavidad perfecta (Cross-Validation).")
    
    # te(0, 1, n_splines=5) significa Tensor Splines para X e Y, forzando usar 5x5 "nudos" (knots).
    # ¡Exactamente igual que los knots=5 del código en MATLAB de Emilio!
    modelo_gam = PoissonGAM(te(0, 1, n_splines=splines)).gridsearch(X, Y, progress=False)
    print("¡GAM entrenado con éxito!")
    
    # Le pedimos al modelo que evalúe su matemática en una grilla fina (50x50) para dibujar
    XX_pos = modelo_gam.generate_X_grid(term=0, n=50)
    Z_pos = modelo_gam.partial_dependence(term=0, X=XX_pos)
    
    x_grid = XX_pos[:, 0].reshape(50, 50)
    y_grid = XX_pos[:, 1].reshape(50, 50)
    z_grid = Z_pos.reshape(50, 50)
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    # Para el GAM usamos contourf porque los Splines son matemáticamente continuos y suaves
    contour = ax.contourf(x_grid, y_grid, z_grid, levels=30, cmap='jet')
    fig.colorbar(contour, ax=ax, label='Tasa de Disparo (Spikes/Bin)')
    ax.set_title(f'GAM (Splines 5x5 como MATLAB) | s={sesion} t={tetrodo} c={neurona}')
    ax.axis('equal')
    
    # === GRÁFICO 2: SERIE TEMPORAL (Predicción vs Realidad) ===
    # Calculamos la predicción continua sobre el recorrido EXACTO que hizo el degú en el tiempo
    prediccion_tiempo = modelo_gam.predict(X)
    
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    
    # Vamos a hacer zoom en los primeros 10000 bines (1000 segundos) para que se vea bien
    limite = 10000 
    tiempo_eje = np.arange(limite) * 0.1 # Multiplicamos por 0.1s para pasarlo a segundos
    
    # 1. Graficamos los spikes reales (discretos) como barras negras
    ax2.bar(tiempo_eje, Y[:limite], width=0.1, color='black', alpha=0.6, label='Spikes Reales (Discretos)')
    
    # 2. Graficamos la predicción suave del modelo como línea roja superpuesta
    ax2.plot(tiempo_eje, prediccion_tiempo[:limite], color='red', linewidth=2, label='Predicción del GAM (Continua)')
    
    ax2.set_xlabel('Tiempo del Experimento (segundos)')
    ax2.set_ylabel('Cantidad de Spikes')
    ax2.set_title(f'¿El modelo adivina cuándo dispara? (Zoom primeros 100s)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# --- Ejecución ---
# 1. Mira el resultado de tu modelo manual usando 5x5 bines como Emilio
# glm_posicion_manual(sesion=2, tetrodo=3, neurona=3, n_bines=5)

# 2. Compara con el GAM profesional, también forzado a 5x5 splines
gam_posicion_automatico(sesion=2, tetrodo=3, neurona=3, splines=5)
