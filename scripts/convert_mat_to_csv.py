import scipy.io as sio
import pandas as pd
import sys
import os

def convert_mat_to_csv(mat_path, csv_path=None):
    if not os.path.exists(mat_path):
        print(f"Error: El archivo {mat_path} no existe.")
        return False
    
    if csv_path is None:
        csv_path = os.path.splitext(mat_path)[0] + '.csv'
        
    print(f"Cargando {mat_path}...")
    try:
        # squeeze_me=True simplifica las dimensiones de Matlab a arrays 1D limpios
        mat = sio.loadmat(mat_path, squeeze_me=True)
    except Exception as e:
        print(f"Error al cargar el archivo .mat: {e}")
        return False
        
    # Extraer variables con fallback por si alguna no existe en el archivo
    # Usamos dict.get para evitar KeyError si la variable no está en el .mat
    cell_ids = mat.get('cell_ids', None)
    if cell_ids is None:
        print("Error: No se encontró 'cell_ids' en el archivo .mat.")
        return False
        
    n_elements = len(cell_ids)
    
    status = mat.get('status', ['unknown'] * n_elements)
    is_sig = mat.get('is_sig', [False] * n_elements)
    best_x = mat.get('best_x', [None] * n_elements)
    best_y = mat.get('best_y', [None] * n_elements)
    stability_dist = mat.get('stability_dist_all', [None] * n_elements)
    persp_indx = mat.get('persp_indx_all', [None] * n_elements)
    z_score = mat.get('z_score_all', [None] * n_elements)
    p_value = mat.get('p_value_all', [None] * n_elements)
    num_spikes = mat.get('num_spikes_all', [None] * n_elements)
    theta0 = mat.get('theta0_all', [None] * n_elements)
    
    # Asegurar que is_sig sea booleano
    if hasattr(is_sig, 'astype'):
        is_sig = is_sig.astype(bool)
    else:
        is_sig = [bool(x) for x in is_sig]
        
    # Armar DataFrame
    df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'Status': status,
        'Significant': is_sig,
        'Best_X': best_x,
        'Best_Y': best_y,
        'Num_Spikes': num_spikes,
        'Stability_Dist': stability_dist,
        'Persp_Index': persp_indx,
        'Theta0': theta0,
        'Z_Score': z_score,
        'P_Value': p_value
    })
    
    try:
        df.to_csv(csv_path, index=False)
        print(f"¡Éxito! Archivo convertido y guardado en: {csv_path}")
        print(f"Total de filas convertidas: {len(df)}")
        return True
    except Exception as e:
        print(f"Error al escribir el archivo CSV: {e}")
        return False

if __name__ == "__main__":
    # Si se pasan argumentos, usar el primero como mat_path
    if len(sys.argv) > 1:
        mat_path = sys.argv[1]
    else:
        mat_path = 'viewpoint_shuffling_results (2).mat'
        
    convert_mat_to_csv(mat_path)
