"""
carga y preparación de datos
======================================
métodos:
- _cargar_datos_neurona: lee hdf5, extrae spikes, calcula velocidad.
- _bin_and_filter: binning temporal e interpolación con filtro de velocidad.
- load_cell_data: entrada principal multi-animal, parsea cell_id y carga datos.
- preparar_datos_posicion: genera bins temporales de posición (x, y) y conteo de spikes.
- preparar_datos_head_direction: genera bins temporales de head direction y conteo de spikes.
- preparar_datos_mirada: genera bins temporales de posición + hd en radianes y conteo de spikes.
"""
import os
import h5py
import scipy.io as sio
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get('DEGUS_DATA_DIR', os.path.join(BASE_DIR, 'data'))


def _cargar_datos_neurona(sesion, tetrodo, neurona, db_merged_path, db_clnew_path):
    """
    funcion base de carga. lee hdf5, extrae spikes.
    devuelve datos crudos sin centrar ni filtrar por velocidad.

    args:
        sesion: número de sesión (época) dentro de la base de datos.
        tetrodo: número de tetrodo.
        neurona: número de neurona dentro del cluster.
        db_merged_path: ruta al archivo .db (hdf5 mergeado).
        db_clnew_path: ruta al archivo .db_clnew (clusters).
    """
    file = h5py.File(db_merged_path, 'r')

    # 1. trayectoria del degus
    nombre_pos = f'pos_{sesion}'
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()

    dt_video = np.mean(np.diff(pos_t))  # segundos por frame ~0.02

    ## calculo de velocidad
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.maximum(np.diff(pos_t), 1e-6)  # evitamos div x0
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)

    # 2. sacamos los spikes de esta neurona
    nombre_spk = f'spk_ts_{sesion}_{tetrodo}'
    spikes = np.array(file[nombre_spk][0]).flatten()

    all_clust = sio.loadmat(db_clnew_path)['all_clust']
    cluster = all_clust[sesion-1][tetrodo-1]
    indices_celula = cluster[0][neurona-1].flatten().astype(int) - 1
    tiempos_celula = spikes[indices_celula]

    file.close()

    return pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula


def _bin_and_filter(pos_x, pos_y, pos_t, vel, tiempos_celula, angulo=None, bin_size_sec=0.1):
    """
    crea bins temporales, interpola posición/velocidad/ángulo a los centros de bin,
    y aplica filtro de velocidad (vel <= 100).

    args:
        pos_x, pos_y, pos_t: arrays de posición y tiempo.
        vel: array de velocidad instantánea.
        tiempos_celula: tiempos de spike de la neurona.
        angulo: array de ángulos (radianes, ya unwrapped). si es None se omite.
        bin_size_sec: tamaño de bin temporal en segundos.

    returns:
        x_bins, y_bins: posiciones filtradas por bin.
        conteo_spikes: conteo de spikes por bin filtrado.
        ang_bins: ángulos interpolados filtrados (None si angulo es None).
    """
    bins_tiempo = np.arange(pos_t[0], pos_t[-1], bin_size_sec)
    centros_bins = bins_tiempo[:-1] + (bin_size_sec / 2)

    conteo_spikes, _ = np.histogram(tiempos_celula, bins=bins_tiempo)

    x_bins = np.interp(centros_bins, pos_t, pos_x)
    y_bins = np.interp(centros_bins, pos_t, pos_y)
    vel_bins = np.interp(centros_bins, pos_t, vel)

    ang_bins = None
    if angulo is not None:
        ang_bins = np.interp(centros_bins, pos_t, angulo)

    # --- FILTRO DE SALTOS DE TRACKING ---
    mask_valida = vel_bins <= 100
    x_bins = x_bins[mask_valida]
    y_bins = y_bins[mask_valida]
    conteo_spikes = conteo_spikes[mask_valida]
    if ang_bins is not None:
        ang_bins = ang_bins[mask_valida]
    # ------------------------------------

    return x_bins, y_bins, conteo_spikes, ang_bins


def load_cell_data(cell_id, data_dir=None, bin_size_sec=0.1):
    """
    entrada principal multi-animal. parsea cell_id (ej. 'XII-OF-I_T1_N1'),
    construye las rutas a las bases de datos, encuentra la época más larga
    y devuelve posición, head direction y spikes binneados y filtrados.

    args:
        cell_id: string con formato 'ANIMAL-SESSION_T{tetrodo}_N{neurona}'.
        data_dir: directorio con los archivos .db y .db_clnew.
                  si es None se usa DATA_DIR del módulo.
        bin_size_sec: tamaño de bin temporal en segundos.

    returns:
        (x_bins, y_bins, ang_bins_rad, conteo_spikes)
        o None si hay error de carga.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    parts = cell_id.split('_')
    animal_session = parts[0]
    tetrode = int(parts[1][1:])
    neuron = int(parts[2][1:])

    db_merged_path = os.path.join(data_dir, f'S2020_Mark{animal_session}_merged.db')
    db_clnew_path = os.path.join(data_dir, f'S2020_Mark{animal_session}.db_clnew')

    if not os.path.exists(db_merged_path):
        print(f"Error: DB no encontrada {db_merged_path}")
        return None
    if not os.path.exists(db_clnew_path):
        print(f"Error: db_clnew no encontrada {db_clnew_path}")
        return None

    # encontrar la época con la trayectoria más larga
    file = h5py.File(db_merged_path, 'r')

    epoch = None
    max_len = 0
    for k in ['pos_1', 'pos_2', 'pos_3']:
        if k in file and len(file[k]['x'][0].shape) > 0:
            if file[k]['x'][0].shape[0] > max_len:
                max_len = file[k]['x'][0].shape[0]
                epoch = int(k.split('_')[1])

    if epoch is None:
        print(f"No se encontraron posiciones validas en {db_merged_path}")
        file.close()
        return None

    # cargar posición, hd, velocidad
    nombre_pos = f'pos_{epoch}'
    pos_x = np.array(file[nombre_pos]['x']).flatten()
    pos_y = np.array(file[nombre_pos]['y']).flatten()
    pos_t = np.array(file[nombre_pos]['t']).flatten()
    angulo = np.array(file[nombre_pos]['hd']).flatten()
    angulo = np.unwrap(angulo)

    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dt = np.maximum(np.diff(pos_t), 1e-6)
    vel = np.append(np.sqrt(dx**2 + dy**2) / dt, 0)

    # cargar spikes
    nombre_spk = f'spk_ts_{epoch}_{tetrode}'
    if nombre_spk not in file:
        print(f"Spike train {nombre_spk} no encontrado en {db_merged_path}")
        file.close()
        return None

    spikes = np.array(file[nombre_spk][0]).flatten()
    file.close()

    # extraer tiempos de spike de la neurona via clusters
    all_clust = sio.loadmat(db_clnew_path)['all_clust']

    try:
        cluster = all_clust[epoch-1][tetrode-1]
        indices_celula = cluster[0][neuron-1].flatten().astype(int) - 1
        tiempos_celula = spikes[indices_celula]
    except Exception as e:
        print(f"Error cargando clusters de {cell_id}: {e}")
        return None

    # binning y filtro de velocidad
    x_bins, y_bins, conteo_spikes, ang_bins = _bin_and_filter(
        pos_x, pos_y, pos_t, vel, tiempos_celula,
        angulo=angulo, bin_size_sec=bin_size_sec
    )

    return x_bins, y_bins, ang_bins, conteo_spikes


def get_all_cell_ids(data_dir=None, master_csv=None, master_mat=None):
    """
    obtiene la lista de todas las células disponibles.
    si se proporciona master_csv o master_mat, se lee de ahí (ej. AllData2.db de MATLAB).
    si no, escanea los archivos .db_clnew en data_dir para autodetectarlas (modo local fallback).
    """
    import glob
    import csv
    
    if data_dir is None:
        data_dir = DATA_DIR
        
    # 1. Leer de CSV si existe
    if master_csv is not None and os.path.exists(master_csv):
        cell_ids = []
        with open(master_csv, 'r') as f:
            reader = csv.DictReader(f)
            if 'Cell_ID' in reader.fieldnames:
                for row in reader:
                    cell_ids.append(row['Cell_ID'])
        if cell_ids:
            return cell_ids
            
    # 2. Leer del .mat de MATLAB (ej. AllData2.db)
    if master_mat is not None and os.path.exists(master_mat):
        try:
            mat = sio.loadmat(master_mat, squeeze_me=True)
            if 'data' in mat:
                data_table = mat['data']
                cell_ids = []
                for row in data_table:
                    if 'OF' in str(row[1]):
                        degu = str(row[0])
                        sess_roman = str(row[1]).split('-')[-1]
                        tet = str(row[3])
                        cl = str(row[4])
                        cell_ids.append(f"{degu}-OF-{sess_roman}_T{tet}_N{cl}")
                return cell_ids
        except Exception as e:
            print(f"Error leyendo {master_mat}: {e}")
            
    # 3. Autodetección escaneando .db_clnew (fallback para entorno local sin master list)
    cell_ids = []
    clnew_files = glob.glob(os.path.join(data_dir, '*.db_clnew'))
    for f in clnew_files:
        basename = os.path.basename(f)
        if basename.startswith('S2020_Mark'):
            animal_session = basename.replace('S2020_Mark', '').replace('.db_clnew', '')
            try:
                all_clust = sio.loadmat(f)['all_clust']
                for s in range(all_clust.shape[0]):
                    for t in range(all_clust.shape[1]):
                        if len(all_clust[s][t]) > 0 and len(all_clust[s][t][0]) > 0:
                            num_neurons = len(all_clust[s][t][0])
                            for n in range(1, num_neurons + 1):
                                cell_ids.append(f"{animal_session}_T{t+1}_N{n}")
            except Exception as e:
                print(f"Error parseando clusters en {f}: {e}")
                
    return cell_ids
def preparar_datos_posicion(sesion, tetrodo, neurona, db_merged_path, db_clnew_path, bin_size_sec=0.1):
    """
    genera bins temporales de posición (x, y) y conteo de spikes.

    args:
        sesion, tetrodo, neurona: identificadores de la neurona.
        db_merged_path: ruta al archivo .db mergeado.
        db_clnew_path: ruta al archivo .db_clnew de clusters.
        bin_size_sec: tamaño del bin temporal (por defecto 0.1s).

    returns:
        X: array (n_bins, 2) con posiciones [x, y].
        Y: array (n_bins,) con conteo de spikes.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(
        sesion, tetrodo, neurona, db_merged_path, db_clnew_path
    )

    x_bins, y_bins, conteo_spikes, _ = _bin_and_filter(
        pos_x, pos_y, pos_t, vel, tiempos_celula, bin_size_sec=bin_size_sec
    )

    X = np.column_stack((x_bins, y_bins))
    Y = conteo_spikes

    return X, Y


def preparar_datos_head_direction(sesion, tetrodo, neurona, db_merged_path, db_clnew_path, bin_size_sec=0.1):
    """
    carga y prepara la dirección de la cabeza (head direction, hd) y los spikes correspondientes.

    args:
        sesion, tetrodo, neurona: identificadores de la neurona.
        db_merged_path: ruta al archivo .db mergeado.
        db_clnew_path: ruta al archivo .db_clnew de clusters.
        bin_size_sec: tamaño del bin temporal (por defecto 0.1s).

    returns:
        ang_bins_deg: array de la dirección de la cabeza en grados (0 a 360).
        conteo_spikes: array con el conteo de spikes en cada bin de tiempo.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(
        sesion, tetrodo, neurona, db_merged_path, db_clnew_path
    )

    # cargar head direction (hd) real desde la base de datos
    file = h5py.File(db_merged_path, 'r')
    angulo = np.array(file[f'pos_{sesion}']['hd']).flatten()
    file.close()

    # el tracking puede tener ángulos con wrap en -pi/pi, los unwrap-eamos para interpolar bien
    angulo = np.unwrap(angulo)

    _, _, conteo_spikes, ang_bins = _bin_and_filter(
        pos_x, pos_y, pos_t, vel, tiempos_celula,
        angulo=angulo, bin_size_sec=bin_size_sec
    )

    # convertimos a grados y envolvemos en [0, 360)
    ang_bins_deg = np.degrees(ang_bins) % 360

    return ang_bins_deg, conteo_spikes


def preparar_datos_mirada(sesion, tetrodo, neurona, db_merged_path, db_clnew_path, bin_size_sec=0.1):
    """
    carga y prepara la trayectoria 2D del animal, su dirección de cabeza (HD) en radianes
    y los spikes correspondientes.

    args:
        sesion, tetrodo, neurona: identificadores de la neurona.
        db_merged_path: ruta al archivo .db mergeado.
        db_clnew_path: ruta al archivo .db_clnew de clusters.
        bin_size_sec: tamaño del bin temporal.

    returns:
        x_bins: array de la posición X del animal por bin.
        y_bins: array de la posición Y del animal por bin.
        ang_bins: array de la dirección de la cabeza en radianes por bin.
        conteo_spikes: array con el conteo de spikes en cada bin.
    """
    pos_x, pos_y, pos_t, dt_video, vel, tiempos_celula = _cargar_datos_neurona(
        sesion, tetrodo, neurona, db_merged_path, db_clnew_path
    )

    # cargar head direction (hd) real desde la base de datos
    file = h5py.File(db_merged_path, 'r')
    angulo = np.array(file[f'pos_{sesion}']['hd']).flatten()
    file.close()

    # unwrap para interpolación correcta
    angulo = np.unwrap(angulo)

    x_bins, y_bins, conteo_spikes, ang_bins = _bin_and_filter(
        pos_x, pos_y, pos_t, vel, tiempos_celula,
        angulo=angulo, bin_size_sec=bin_size_sec
    )

    return x_bins, y_bins, ang_bins, conteo_spikes
