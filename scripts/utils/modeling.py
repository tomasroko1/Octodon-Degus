"""
métodos:
- glm_position: implementa un glm manual desde cero usando campanas de gauss.
- get_gam_posicion: entrena o carga un modelo gam (pygam) poisson para posición 2d.
- graficar_gam_posicion: visualiza los resultados espaciales y temporales del gam de posición.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pygam import PoissonGAM, te, s

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELOS_DIR = os.path.join(BASE_DIR, 'models')

def glm_position(X, Y, n_bines=36, alpha=0.01, title=''):
    print("\n--- INICIANDO GLM ---")
    
    pos_x = X[:, 0]
    pos_y = X[:, 1]
    
    n_bases_x = n_bines
    n_bases_y = n_bines
    centros_x = np.linspace(np.min(pos_x), np.max(pos_x), n_bases_x)
    centros_y = np.linspace(np.min(pos_y), np.max(pos_y), n_bases_y)
    sigma_pos = (np.max(pos_x) - np.min(pos_x)) / n_bases_x
    
    # matriz donde cada columna es una de las campanas
    X_bases_pos = np.zeros((len(pos_x), n_bases_x * n_bases_y))
    columna = 0
    for cx in centros_x:
        for cy in centros_y:
            dist_cuadrada = (pos_x - cx)**2 + (pos_y - cy)**2
            X_bases_pos[:, columna] = np.exp(- dist_cuadrada / (2 * sigma_pos**2))
            columna += 1
            
    # agregamos constante y entrenamos
    X_glm_pos = sm.add_constant(X_bases_pos)
    
    # solución: "regularización" (ridge)
    modelo = sm.GLM(Y, X_glm_pos, family=sm.families.Poisson()).fit_regularized(alpha=alpha, L1_wt=0.0)
    print("Modelo entrenado")

    # --- graficar ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    
    # generamos los datos evaluando la matemática exactamente en los centros de los bins
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
    
    # graficamos usando pcolormesh para pintar los "cuadrados" rígidos sin suavizar
    # le agregamos bordes negros para que los bines sean 100% distinguibles
    mesh = ax.pcolormesh(x_grid, y_grid, prediccion_pos, cmap='jet', shading='nearest', linewidth=0.5)
    fig.colorbar(mesh, ax=ax, label='Tasa de Disparo (Spikes/Bin)')
    ax.set_title(title or f'GLM: {n_bines}x{n_bines}')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()

    return modelo

def get_gam_posicion(X, Y, cell_id, splines, lam, force_retrain=False):
    os.makedirs(MODELOS_DIR, exist_ok=True)
    archivo_modelo = os.path.join(MODELOS_DIR, f'modelo_gam_pos_{cell_id}_sp{splines}.pkl')
    
    if os.path.exists(archivo_modelo) and not force_retrain:
        print(f"[+] Cargando GAM Posición guardado desde {archivo_modelo}...")
        with open(archivo_modelo, 'rb') as f:
            modelo_gam = pickle.load(f)
    else:
        print(f"[-] Entrenando GAM Posición ({splines}x{splines} splines)...")
        modelo_gam = PoissonGAM(te(0, 1, n_splines=splines, lam=lam)).fit(X, Y)

        with open(archivo_modelo, 'wb') as f:
            pickle.dump(modelo_gam, f)
            
    print("\n=== RESUMEN GAM POSICIÓN ===")
    modelo_gam.summary()
    return modelo_gam

def graficar_gam_posicion(modelo_gam, X, Y, title='', bin_size_sec=0.1):
    print("\n--- GRAFICANDO GAM ---")
    
    # 1. Definimos una resolución alta (n=100) para un renderizado muy suave
    n_res = 100 
    XX_pos = modelo_gam.generate_X_grid(term=0, n=n_res)
    Z_pos = modelo_gam.predict(XX_pos)

    # 2. Obligatorio para contourf: Convertir las listas planas en matrices 2D (100x100)
    x_grid = XX_pos[:, 0].reshape(n_res, n_res)
    y_grid = XX_pos[:, 1].reshape(n_res, n_res)
    z_grid = Z_pos.reshape(n_res, n_res)
    
    # 3. Crear una máscara de ocupancia basada en las posiciones reales (X)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    distancias, _ = tree.query(XX_pos)
    
    # También debemos hacer reshape a las distancias para que coincidan con la grilla
    distancias = distancias.reshape(n_res, n_res)
    
    # Ocultar las zonas no visitadas (> 5 cm)
    z_grid[distancias > 5.0] = np.nan
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    mesh = ax.pcolormesh(x_grid, y_grid, z_grid, cmap='jet', shading='nearest')
    fig.colorbar(mesh, ax=ax, label='Tasa de Disparo (Spikes/Bin)')
    ax.set_title(title or 'GAM Model')
    ax.set_aspect('equal')
    ax.axis('off')
    
    ## 2do plot
    prediccion_tiempo = modelo_gam.predict(X)
    
    fig2 = plt.figure(figsize=(12, 4))
    ax2 = fig2.add_subplot(111)
    
    limite = len(Y) 
    tiempo_eje = np.arange(limite) * bin_size_sec
    
    ax2.bar(tiempo_eje, Y[:limite], width=bin_size_sec, color='black', alpha=0.6, label='spikes')
    ax2.plot(tiempo_eje, prediccion_tiempo[:limite], color='red', linewidth=2, label='spikes prediction')
    
    ax2.set_xlabel('time (seconds)')
    ax2.set_ylabel('spike count')
    
    conteos = np.bincount(Y[:limite].astype(int))
    umbral = max(1, int(limite * 0.001))
    valores_comunes = np.where(conteos > umbral)[0]
    max_visible = np.max(valores_comunes) if len(valores_comunes) > 0 else np.max(Y[:limite])
    
    ax2.set_ylim(0, max_visible + 1)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()