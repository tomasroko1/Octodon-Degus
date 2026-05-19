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
    Calcula la tasa de disparo de la neurona cuando el animal mira hacia un punto específico (x_target, y_target)
    en el mapa 2D, en comparación con cuando mira hacia otros lados.
    
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
    
    Esto nos permite descubrir de forma totalmente objetiva cuál es el "punto preferido de la caja" 
    al que esta neurona le presta atención.
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


def main():
    # Asegurar que se puede importar de módulos hermanos (data_loader, etc.)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from data_loader import preparar_datos_mirada
    except ImportError:
        from .data_loader import preparar_datos_mirada

    # 1. Cargar los datos reales usando el data_loader
    sesion, tetrodo, neurona = 2, 3, 1
    bin_size = 0.1
    print(f"Cargando datos espaciales y Head Direction para s={sesion}, t={tetrodo}, c={neurona}...")
    
    try:
        x_bins, y_bins, ang_bins_rad, spikes = preparar_datos_mirada(sesion, tetrodo, neurona, bin_size_sec=bin_size, min_vel=2.0)
        es_sintetico = False
    except Exception as e:
        print(f"Error al cargar datos reales: {e}")
        print("Generando datos sintéticos de una 'Gaze Cell' (célula de mirada) para demostración visual...")
        es_sintetico = True
        # Trayectoria sintética tipo caminata aleatoria suave en arena de 80x80 cm
        np.random.seed(42)
        n_steps = 8000
        from scipy.ndimage import gaussian_filter1d
        x_bins = gaussian_filter1d(np.cumsum(np.random.normal(0, 1.8, n_steps)), sigma=6)
        y_bins = gaussian_filter1d(np.cumsum(np.random.normal(0, 1.8, n_steps)), sigma=6)
        
        # Escalar a límites realistas (-40 a 40)
        x_bins = (x_bins - np.min(x_bins)) / (np.max(x_bins) - np.min(x_bins)) * 80 - 40
        y_bins = (y_bins - np.min(y_bins)) / (np.max(y_bins) - np.min(y_bins)) * 80 - 40
        
        # Dirección de la cabeza con algo de ruido sobre la dirección de movimiento
        dx = np.diff(x_bins)
        dy = np.diff(y_bins)
        ang_bins_rad = np.arctan2(dy, dx)
        ang_bins_rad = np.append(ang_bins_rad, ang_bins_rad[-1]) + np.random.normal(0, 0.4, n_steps)
        
        # Definimos un objeto/punto preferido en la coordenada (-20, 15)
        x_pref_real, y_pref_real = -20.0, 15.0
        
        # El animal mira hacia ese punto
        dx_target = x_pref_real - x_bins
        dy_target = y_pref_real - y_bins
        ang_hacia_punto = np.arctan2(dy_target, dx_target)
        diff_ang = np.arctan2(np.sin(ang_bins_rad - ang_hacia_punto), np.cos(ang_bins_rad - ang_hacia_punto))
        diff_ang_deg = np.abs(np.degrees(diff_ang))
        
        # Probabilidad de disparo alta solo cuando mira al punto y está a más de 10 cm de él
        dist_punto = np.sqrt(dx_target**2 + dy_target**2)
        prob_disparo = 0.02 + 0.35 * np.exp(- (diff_ang_deg)**2 / (2 * 18**2))
        prob_disparo[dist_punto < 10.0] = 0.01
        
        spikes = np.random.poisson(prob_disparo)
        
    print(f"Datos cargados con éxito: {len(x_bins)} bines temporales.")
    print("Buscando de forma óptima el punto preferido de mirada en la caja 2D...")
    
    # 2. Correr la optimización sobre una grilla 2D
    XX, YY, delta_R_matrix, coord_best, best_delta_R = encontrar_punto_preferido_mirada(
        x_bins, y_bins, ang_bins_rad, spikes, bin_size, grid_res=25, tolerancia_deg=30
    )
    
    x_best, y_best = coord_best
    print(f"\n=== ANÁLISIS DE MIRADA COMPLETADO ===")
    print(f"Coordenada Preferida Detectada en el Mapa: X = {x_best:.1f} cm, Y = {y_best:.1f} cm")
    
    # Evaluar métricas en esa coordenada óptima
    rate_mira, rate_no_mira, mira_al_punto = analizar_mirada_hacia_punto(
        x_bins, y_bins, ang_bins_rad, spikes, bin_size, x_best, y_best, tolerancia_deg=30
    )
    
    print(f"Tasa de disparo mirando al Punto Preferido: {rate_mira:.2f} Hz")
    print(f"Tasa de disparo mirando hacia otros lados: {rate_no_mira:.2f} Hz")
    print(f"Aumento de respuesta neta (Selectividad Delta R): {best_delta_R:.2f} Hz")
    
    # --- Graficación con Estética Científica Premium ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 5.8), facecolor='#111115')
    
    # Panel 1: Mapa de Trayectoria y Spikes con el Punto Preferido
    ax_map = fig.add_subplot(131, facecolor='#16161D')
    # Recorrido del degu en gris sutil
    ax_map.plot(x_bins, y_bins, color='#666670', alpha=0.35, linewidth=0.8, label='Trayectoria del animal')
    
    # Momentos donde la neurona disparó (spikes en puntos rojos)
    idx_spk = spikes > 0
    ax_map.scatter(x_bins[idx_spk], y_bins[idx_spk], color='#FF4B4B', s=25, alpha=0.8, 
                   edgecolors='white', linewidths=0.3, zorder=5, label='Spikes de la Neurona')
    
    # Dibujar el Punto Preferido de Mirada como una estrella dorada brillante
    ax_map.scatter(x_best, y_best, color='#FFEA00', marker='*', s=350, edgecolor='black', linewidths=1.2, 
                   zorder=10, label='Punto Preferido')
    
    # Añadir algunas líneas de mirada ilustrativas de color cian y vectores de cabeza en amarillo
    # para entender EXACTAMENTE hacia dónde apuntaba el animal al disparar
    mira_y_dispara = np.where(mira_al_punto & (spikes > 0))[0]
    if len(mira_y_dispara) > 0:
        dibujados = 0
        puntos_elegidos = np.linspace(0, len(mira_y_dispara)-1, min(6, len(mira_y_dispara)), dtype=int)
        for idx_idx in puntos_elegidos:
            idx = mira_y_dispara[idx_idx]
            xs, ys = x_bins[idx], y_bins[idx]
            # Línea de mirada al punto preferido
            ax_map.plot([xs, x_best], [ys, y_best], color='#00E5FF', linestyle=':', alpha=0.5, linewidth=1.1)
            # Flecha física de la dirección de la cabeza
            v_len = 5.0
            vx = v_len * np.cos(ang_bins_rad[idx])
            vy = v_len * np.sin(ang_bins_rad[idx])
            ax_map.arrow(xs, ys, vx, vy, head_width=1.8, head_length=2.2, fc='#FFEA00', ec='#FFEA00', alpha=0.85, zorder=6)
            dibujados += 1
            
    ax_map.set_title('A: Trayectoria y Spikes en la Caja', fontsize=12, fontweight='bold', color='white', pad=15)
    ax_map.set_xlabel('Posición X (cm)', fontsize=10, color='#DDDDDD')
    ax_map.set_ylabel('Posición Y (cm)', fontsize=10, color='#DDDDDD')
    ax_map.axis('equal')
    ax_map.grid(True, linestyle=':', alpha=0.15, color='#FFFFFF')
    ax_map.legend(frameon=True, facecolor='#16161D', edgecolor='#44444C', loc='upper right', fontsize=9)
    
    # Panel 2: Mapa de calor 2D de selectividad de mirada
    ax_heat = fig.add_subplot(132, facecolor='#16161D')
    # Graficar la grilla de selectividad
    mesh = ax_heat.pcolormesh(XX, YY, delta_R_matrix, cmap='jet', shading='nearest')
    cbar = fig.colorbar(mesh, ax=ax_heat)
    cbar.set_label('Selectividad ΔR (Hz) [Mirar vs No Mirar]', fontsize=10, color='#DDDDDD')
    cbar.ax.tick_params(colors='#AAAAAB')
    
    # Marcar el punto óptimo aquí también
    ax_heat.scatter(x_best, y_best, color='#FFEA00', marker='*', s=250, edgecolor='black', linewidths=1.0, zorder=10)
    
    ax_heat.set_title('B: Mapa 2D de Sintonización de Mirada', fontsize=12, fontweight='bold', color='white', pad=15)
    ax_heat.set_xlabel('Posición X (cm)', fontsize=10, color='#DDDDDD')
    ax_heat.set_ylabel('Posición Y (cm)', fontsize=10, color='#DDDDDD')
    ax_heat.axis('equal')
    ax_heat.grid(True, linestyle=':', alpha=0.15, color='#FFFFFF')
    
    # Panel 3: Gráfico de Barra Sencillo e Intuitivo (Gaze ON vs OFF)
    ax_bar = fig.add_subplot(133, facecolor='#16161D')
    barras = ax_bar.bar(['Mirando al Punto\n(Gaze ON)', 'Mirando Fuera\n(Gaze OFF)'], 
                       [rate_mira, rate_no_mira], 
                       color=['#00E5FF', '#FF6B6B'], alpha=0.85, edgecolor='white', width=0.5, zorder=3)
    
    # Añadir valores exactos sobre las barras
    for bar in barras:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f} Hz',
                    ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')
        
    ax_bar.set_ylabel('Tasa de Disparo Promedio (Hz)', fontsize=10, color='#DDDDDD')
    ax_bar.set_title('C: Tasa según Condición de Mirada', fontsize=12, fontweight='bold', color='white', pad=15)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#44444C')
    ax_bar.spines['bottom'].set_color('#44444C')
    ax_bar.tick_params(colors='#AAAAAB')
    ax_bar.grid(True, axis='y', linestyle=':', alpha=0.15, color='#FFFFFF', zorder=0)
    
    # Título Principal
    tipo_data = "DATOS REALES" if not es_sintetico else "DATOS DEMOSTRATIVOS (GAZE CELL)"
    plt.suptitle(f'CÉLULA DE MIRADA ESPACIAL (Viewpoint Gaze Cell) | {tipo_data}\ns={sesion} | t={tetrodo} | c={neurona} | Punto Óptimo: ({x_best:.1f}, {y_best:.1f}) cm', 
                 fontsize=14, fontweight='bold', color='white', y=0.97)
    
    plt.tight_layout()
    output_png = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sintonizacion_gaze_2d.png')
    plt.savefig(output_png, dpi=300, facecolor='#111115')
    print(f"\n¡Gráfico intuitivo de mirada 2D guardado con éxito en: {output_png}!")
    plt.show()

if __name__ == "__main__":
    main()
