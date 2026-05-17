"""
métodos:
- glm_posicion_manual: implementa un glm manual desde cero usando campanas de gauss.
- get_gam_posicion: entrena o carga un modelo gam (pygam) poisson para posición 2d.
- graficar_gam_posicion: visualiza los resultados espaciales y temporales del gam de posición.
- get_gam_viewpoint_1d: entrena o carga un modelo gam cíclico para la mirada en el perímetro 1d.
- graficar_gam_viewpoint_1d: visualiza los resultados predichos del gam sobre las 4 paredes.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pygam import PoissonGAM, te, s
try:
    from .data_loader import preparar_datos_posicion, preparar_datos_viewpoint_1d
except ImportError:
    from data_loader import preparar_datos_posicion, preparar_datos_viewpoint_1d

def glm_posicion_manual(sesion, tetrodo, neurona, n_bines=36):
    print("\n--- INICIANDO GLM ---")
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec=0.1)
    
    pos_x = X[:, 0]
    pos_y = X[:, 1]
    
    print(f"Construyendo Basis Functions (Grilla de {n_bines}x{n_bines} Campanas Gauss)...")
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
    modelo = sm.GLM(Y, X_glm_pos, family=sm.families.Poisson()).fit_regularized(alpha=0.1, L1_wt=0.0)
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
    ax.set_title(f'GLM (Píxeles): {n_bines}x{n_bines} | s={sesion} t={tetrodo} c={neurona}')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()

def get_gam_posicion(sesion, tetrodo, neurona, splines, lam, bin_size_sec=0.1, force_retrain=False):
    archivo_modelo = f'modelo_gam_pos_s{sesion}_t{tetrodo}_n{neurona}_sp{splines}.pkl'
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size_sec)
    
    if os.path.exists(archivo_modelo) and not force_retrain:
        print(f"[+] Cargando GAM Posición guardado desde {archivo_modelo}...")
        with open(archivo_modelo, 'rb') as f:
            modelo_gam = pickle.load(f)
    else:
        print(f"[-] Entrenando GAM Posición ({splines}x{splines} splines)...")
        
        #modelo_gam = PoissonGAM(te(0, 1, n_splines=splines)).gridsearch(X, Y, progress=False)

        # Evitamos usar .gridsearch() que usa GCV (Generalized Cross Validation)
        # para s 2 3 3 obtuvo los mismos valores de lambda que los que fueron
        # obtenidos cross-validando
        # Ya que tenemos el lambda cross-validado -> evitamos el GCV innecesario 
        
        modelo_gam = PoissonGAM(te(0, 1, n_splines=splines, lam=lam)).fit(X, Y)


        ## queremos calcular el error del gam. por ejemplo compararlo con el glm, la prediccion
        ## de spikes (media, depende del tiempo. no fija) contra los spikes reales (realizacion)


        ## agregar grafico gam2 tambien para glm y comparar

        ## queremos asegurarnos de que esto este crossvalidando bien -> croosvalidar por segmento en la linea temporal

        with open(archivo_modelo, 'wb') as f:
            pickle.dump(modelo_gam, f)
            
    print("\n=== RESUMEN GAM POSICIÓN ===")
    modelo_gam.summary()
    return modelo_gam, X, Y

def graficar_gam_posicion(modelo_gam, X, Y, sesion, tetrodo, neurona, splines, bin_size_sec=0.1):
    print("\n--- GRAFICANDO GAM ---")
    
    # 1. Definimos una resolución alta (n=100) para un renderizado muy suave
    n_res = 100 
    XX_pos = modelo_gam.generate_X_grid(term=0, n=n_res)
    #Z_pos = np.exp(modelo_gam.partial_dependence(term=0, X=XX_pos))
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
    ax.set_title(f'GAM Model | s={sesion} t={tetrodo} c={neurona}')
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

def get_gam_viewpoint_1d(sesion, tetrodo, neurona, splines=20, bin_size_sec=0.1, force_retrain=False):
    archivo_modelo = f'modelo_gam_vp1d_s{sesion}_t{tetrodo}_n{neurona}_sp{splines}.pkl'
    X, Y, W, H = preparar_datos_viewpoint_1d(sesion, tetrodo, neurona, bin_size_sec)
    
    if os.path.exists(archivo_modelo) and not force_retrain:
        print(f"[+] Cargando GAM Viewpoint 1D desde {archivo_modelo}...")
        with open(archivo_modelo, 'rb') as f:
            modelo_gam = pickle.load(f)
    else:
        print(f"[-] Entrenando GAM Viewpoint 1D ({splines} splines)...")
        modelo_gam = PoissonGAM(s(0, basis='cp', n_splines=splines)).gridsearch(X, Y, progress=False)
        with open(archivo_modelo, 'wb') as f:
            pickle.dump(modelo_gam, f)
            
    print("\n=== RESUMEN GAM 1D ===")
    modelo_gam.summary()
    return modelo_gam, X, Y, W, H

def graficar_gam_viewpoint_1d(modelo_gam, X, Y, W, H, sesion, tetrodo, neurona):
    """
    grafica el "mapa de pared" desenrollado.
    """
    perimetro_total = 2*W + 2*H
    
    XX_pred = np.linspace(0, perimetro_total, 200).reshape(-1, 1)
    YY_pred = modelo_gam.predict(XX_pred)
    intervalos = modelo_gam.confidence_intervals(XX_pred, width=.95)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # tasa empírica (barras grises de fondo)
    n_bins = 50
    bins_perimetro = np.linspace(0, perimetro_total, n_bins+1)
    spikes_sect, _ = np.histogram(X.flatten(), bins=bins_perimetro, weights=Y)
    tiempo_sect, _ = np.histogram(X.flatten(), bins=bins_perimetro)
    
    tiempo_sect = np.maximum(tiempo_sect, 1)
    tasa_cruda = spikes_sect / (tiempo_sect * 0.1)
    
    ax.bar(bins_perimetro[:-1], tasa_cruda, width=bins_perimetro[1]-bins_perimetro[0], 
            color='gray', alpha=0.3, align='edge')
    
    # predicción gam (línea roja)
    ax.fill_between(XX_pred.flatten(), intervalos[:, 0], intervalos[:, 1], color='red', alpha=0.2)
    ax.plot(XX_pred.flatten(), YY_pred, color='red', linewidth=2)
    
    # paredes
    esquinas = [0, W, W+H, 2*W+H, perimetro_total]
    nombres_paredes = ['Pared Abajo', 'Pared Derecha', 'Pared Arriba', 'Pared Izquierda']
    
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
