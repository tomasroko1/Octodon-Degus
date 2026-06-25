import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    mat_file = 'viewpoint_shuffling_results.mat'
    
    if not os.path.exists(mat_file):
        print(f"Error: No se encontró el archivo {mat_file} en este directorio.")
        return

    print("Cargando resultados de MATLAB...")
¿    mat = sio.loadmat(mat_file, squeeze_me=True)

    # Extraer variables
    best_x = mat['best_x']
    best_y = mat['best_y']
    persp_indx = mat['persp_indx_all']
    z_score = mat['z_score_all']
    p_value = mat['p_value_all']
    
    # Manejar el caso donde la nueva variable pueda no existir en mat viejos
    stability_dist = mat.get('stability_dist_all', [None]*len(cell_ids))
    
    is_sig = mat['is_sig'].astype(bool)
    cell_ids = mat['cell_ids']
    status = mat['status']

    # 1. Crear y exportar la tabla para que puedas verla
    df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'Status': status,
        'Significant': is_sig,
        'Best_X': best_x,
        'Best_Y': best_y,
        'Stability_Dist': stability_dist,
        'Persp_Index': persp_indx,
        'Z_Score': z_score,
        'P_Value': p_value
    })

    csv_path = 'viewpoint_results_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nLa tabla fue exportada a: {csv_path}")

    # 2. Generar el Scatter Plot
    # Filtramos las que se procesaron bien
    df_valid = df[df['Status'] == 'success']
    
    sig_cells = df_valid[df_valid['Significant'] == True]
    nonsig_cells = df_valid[df_valid['Significant'] == False]

    plt.figure(figsize=(10, 8))
    
    # Graficar primero las no significativas (gris de fondo)
    plt.scatter(nonsig_cells['Best_X'], nonsig_cells['Best_Y'], 
                color='gray', alpha=0.4, label=f'No Sig. (n={len(nonsig_cells)})')
    
    # Graficar las significativas arriba (rojo y más grandes)
    plt.scatter(sig_cells['Best_X'], sig_cells['Best_Y'], 
                color='red', alpha=0.8, s=60, edgecolor='black', 
                label=f'Significativa (n={len(sig_cells)})')

    # Dibujar los límites aproximados de la caja (0 a 90 cm) como referencia
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=90, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=90, color='k', linestyle='--', alpha=0.3)

    plt.title('Distribución Espacial de Puntos de Perspectiva (Viewpoints)')
    plt.xlabel('Coordenada X (cm)')
    plt.ylabel('Coordenada Y (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal') # Para que 1cm en X se vea igual que 1cm en Y
    
    # Guardar la imagen
    plot_path = 'scatter_viewpoints.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado como: {plot_path}")
    
    # Mostrar el gráfico en pantalla
    plt.show()

if __name__ == "__main__":
    main()
