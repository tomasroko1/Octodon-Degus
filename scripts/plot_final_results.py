import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def main():
    # Rutas a tus archivos locales
    mat_file = r"C:\Users\Estudio\Downloads\viewpoint_results.mat"
    csv_file = r"C:\Users\Estudio\Downloads\gam_results.csv"

    if not os.path.exists(mat_file):
        print(f"Error: No se encontró {mat_file}")
        return
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró {csv_file}")
        return

    print("Cargando resultados de MATLAB...")
    mat = sio.loadmat(mat_file, squeeze_me=True)

    # Armar DataFrame de MATLAB
    df_mat = pd.DataFrame({
        'Cell_ID': mat['cell_ids'],
        'Status': mat['status'],
        'Significant_MATLAB': mat['is_sig'].astype(bool),
        'Best_X': mat['best_x'],
        'Best_Y': mat['best_y'],
        'Z_Score': mat['z_score_all'],
        'P_Value': mat['p_value_all']
    })
    
    # Filtrar solo las válidas
    df_mat = df_mat[df_mat['Status'] == 'success']

    print("Cargando resultados de PyGAM...")
    df_gam = pd.read_csv(csv_file)

    # Eliminar duplicados para que pandas no cree Best_X_x y Best_X_y
    if 'Best_X' in df_gam.columns:
        df_gam = df_gam.drop(columns=['Best_X', 'Best_Y', 'Status'])

    # Unir ambas tablas usando el Cell_ID
    df_final = pd.merge(df_mat, df_gam, on='Cell_ID', how='inner')
    
    print(f"Total de células procesadas en ambos pasos: {len(df_final)}")

    # Determinar cuáles son "Viewpoint Cells" reales según GAM
    # (Por ejemplo, aquellas donde el Shapley de Viewpoint > 0 y R2 de Viewpoint > 0)
    # Ajusta esta lógica según tu criterio biológico
    viewpoint_cells = df_final[
        (df_final['Shapley_View'] > 0) & 
        (df_final['View_PseudoR2'] > 0)
    ]
    
    other_cells = df_final[~df_final['Cell_ID'].isin(viewpoint_cells['Cell_ID'])]

    # 1. Generar Scatter Plot de Posiciones de Viewpoint
    plt.figure(figsize=(10, 8))
    
    # Fondo: Células que no pasaron el filtro GAM
    plt.scatter(other_cells['Best_X'], other_cells['Best_Y'], 
                color='gray', alpha=0.4, label=f'Otras (n={len(other_cells)})')
    
    # Frente: Viewpoint Cells confirmadas por GAM
    plt.scatter(viewpoint_cells['Best_X'], viewpoint_cells['Best_Y'], 
                color='red', alpha=0.8, s=60, edgecolor='black', 
                label=f'Viewpoint Cells GAM (n={len(viewpoint_cells)})')

    # Límites aproximados de la caja
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=90, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=90, color='k', linestyle='--', alpha=0.3)

    plt.title('Distribución Espacial de Viewpoints (Validadas por PyGAM)')
    plt.xlabel('Coordenada X (cm)')
    plt.ylabel('Coordenada Y (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal') 
    
    plt.tight_layout()
    plt.show()

    # 2. Exportar la tabla combinada final para tu Excel/Prism
    output_csv = r"C:\Users\Estudio\Downloads\resultados_combinados_finales.csv"
    df_final.to_csv(output_csv, index=False)
    print(f"\n¡Tabla final combinada exportada a {output_csv}!")

if __name__ == "__main__":
    main()
