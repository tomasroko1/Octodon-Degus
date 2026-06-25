import scipy.io as sio
import csv
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
        mat = sio.loadmat(mat_path, squeeze_me=True)
    except Exception as e:
        print(f"Error al cargar el archivo .mat: {e}")
        return False
        
    cell_ids = mat.get('cell_ids', None)
    if cell_ids is None:
        print("Error: No se encontró 'cell_ids' en el archivo .mat.")
        return False
        
    n_elements = len(cell_ids)
    
    status = mat.get('status', ['unknown'] * n_elements)
    is_sig = mat.get('is_sig', [False] * n_elements)
    best_x = mat.get('best_x', [None] * n_elements)
    best_y = mat.get('best_y', [None] * n_elements)
    
    # Asegurar que is_sig sea booleano
    if hasattr(is_sig, 'astype'):
        is_sig = is_sig.astype(bool)
    else:
        is_sig = [bool(x) for x in is_sig]
        
    # Armar lista de diccionarios
    rows = []
    for i in range(n_elements):
        # Mapeamos nombres exactos para que el step2 en python lo lea bien
        row = {
            'Cell_ID': str(cell_ids[i]) if cell_ids[i] else '',
            'status': str(status[i]) if i < len(status) else 'unknown',
            'is_significant': str(is_sig[i]) if i < len(is_sig) else 'False',
            'Best_X': str(best_x[i]) if i < len(best_x) else '',
            'Best_Y': str(best_y[i]) if i < len(best_y) else ''
        }
        rows.append(row)
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Cell_ID', 'status', 'is_significant', 'Best_X', 'Best_Y'])
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"¡Éxito! Archivo convertido y guardado en: {csv_path}")
        print(f"Total de filas convertidas: {len(rows)}")
        return True
    except Exception as e:
        print(f"Error al escribir el archivo CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mat_path = sys.argv[1]
    else:
        mat_path = '../results/viewpoint_results.mat'
        
    convert_mat_to_csv(mat_path)
