#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Viewpoint Analysis & Significance Shuffling
Iterates over all available cells, computes the continuous viewpoint tuning,
and performs circular spike shuffling to determine spatial significance.
Outputs: results/viewpoint_results.csv
"""

import os
import sys
import time
import argparse
import pickle
import csv
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_cell_data, get_all_cell_ids
from utils.angulo_preferido import (
    analizar_sintonizacion_perspectiva_continua,
    evaluar_confianza_sintonizacion_perspectiva
)

def run_step1(master_csv=None, master_mat=None, data_dir=None):
    # ---- Parámetros del análisis ----
    bin_size_sec = 0.1
    grid_res = 25              # Resolución para la sintonización de perspectiva
    n_shifts = 100             # Número de desplazamientos circulares (shuffling)
    min_shift_sec = 10.0       # Desplazamiento mínimo en segundos
    min_spikes = 10            # Umbral mínimo de spikes para procesar la neurona
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, 'viewpoint_results.csv')
    pkl_path = os.path.join(results_dir, 'viewpoint_results.pkl')
    
    # Obtener todas las células disponibles
    cells = get_all_cell_ids(data_dir=data_dir, master_csv=master_csv, master_mat=master_mat)
    total_cells = len(cells)
    
    if total_cells == 0:
        print("No se encontraron celulas para procesar.")
        return
        
    print(f"Iniciando Step 1: Viewpoint Analysis para {total_cells} células...")
    
    results_list = []
    completed_keys = set()
    
    # Cargar resultados previos para poder resumir si el script fue interrumpido
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                results_list = list(reader)
            
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    results_list = pickle.load(f)
            
            completed_keys = set(r['Cell_ID'] for r in results_list)
            print(f"Cargados {len(completed_keys)} resultados previos. Continuando...")
        except Exception as e:
            print(f"No se pudieron cargar resultados previos (iniciando desde cero): {e}")
            results_list = []
            
    t_start = time.time()
    
    for idx, cell_id in enumerate(cells):
        if cell_id in completed_keys:
            print(f"[{idx+1}/{total_cells}] {cell_id} ya procesada. Saltando.")
            continue
            
        t_cell = time.time()
        print(f"[{idx+1}/{total_cells}] Procesando {cell_id}...")
        
        cell_res = {
            'Cell_ID': cell_id,
            'num_spikes': 0,
            'Best_X': np.nan,
            'Best_Y': np.nan,
            'persp_index': np.nan,
            'z_score': np.nan,
            'p_value': np.nan,
            'is_significant': False,
            'theta0': np.nan,
            'status': 'success',
            'error_msg': ''
        }
        
        try:
            data = load_cell_data(cell_id, data_dir=data_dir, bin_size_sec=bin_size_sec)
            if data is None:
                raise ValueError("load_cell_data devolvió None")
                
            x_bins, y_bins, ang_bins_rad, spikes = data
            
            # Filtramos tracking malo del angulo si hubiera
            valid_mask = ~np.isnan(ang_bins_rad)
            x_bins = x_bins[valid_mask]
            y_bins = y_bins[valid_mask]
            ang_bins_rad = ang_bins_rad[valid_mask]
            spikes = spikes[valid_mask]
            
            num_spikes = np.sum(spikes)
            cell_res['num_spikes'] = num_spikes
            
            if num_spikes < min_spikes:
                print(f"  Saltada: muy pocos spikes ({num_spikes} < {min_spikes}).")
                cell_res['status'] = 'skipped_low_spikes'
                results_list.append(cell_res)
                _save_results(results_list, csv_path, pkl_path)
                continue
            
            # 2. Análisis de sintonización de perspectiva circular continua
            res_persp = analizar_sintonizacion_perspectiva_continua(
                x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin=bin_size_sec, grid_res=grid_res
            )
            x_best_p, y_best_p = res_persp['best_coord']
            
            # 3. Test de control por spike shifting (shuffling) para evaluar significancia
            res_sig = evaluar_confianza_sintonizacion_perspectiva(
                x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin=bin_size_sec,
                grid_res=grid_res, n_shifts=n_shifts, min_shift_sec=min_shift_sec
            )
            
            cell_res.update({
                'Best_X': x_best_p,
                'Best_Y': y_best_p,
                'persp_index': res_persp['persp_indx'],
                'z_score': res_sig['z_score'],
                'p_value': res_sig['p_value'],
                'is_significant': res_sig['confianza_significativa'],
                'theta0': res_persp['theta0'],
                '_full_res_persp': res_persp,
                '_full_res_sig': res_sig
            })
            
            status_str = "SIGNIFICATIVA" if res_sig['confianza_significativa'] else "no significativa"
            print(f"  Completada en {time.time() - t_cell:.1f}s | Coord: ({x_best_p:.1f}, {y_best_p:.1f}) | Z: {res_sig['z_score']:.2f} | P: {res_sig['p_value']:.4f} | Status: {status_str}")
            
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"  Error procesando {cell_id}: {e}")
            cell_res['status'] = 'error'
            cell_res['error_msg'] = err_msg
            
        results_list.append(cell_res)
        _save_results(results_list, csv_path, pkl_path)
        
    print(f"\n==================================================")
    print(f"Step 1 completado en {time.time() - t_start:.1f} segundos!")
    print(f"Resultados guardados en: {csv_path}")
    print(f"==================================================")

def _save_results(results_list, csv_path, pkl_path):
    scalar_results = []
    for r in results_list:
        scalar_r = {k: v for k, v in r.items() if not str(k).startswith('_')}
        scalar_results.append(scalar_r)
        
    if scalar_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=scalar_results[0].keys())
            writer.writeheader()
            writer.writerows(scalar_results)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_list, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Viewpoint Analysis")
    parser.add_argument("--master_csv", type=str, default=None, help="Ruta a lista maestra CSV")
    parser.add_argument("--master_mat", type=str, default=None, help="Ruta a tabla maestra .mat (ej. AllData2.db)")
    parser.add_argument("--data_dir", type=str, default=None, help="Carpeta de datos")
    args = parser.parse_args()
    
    run_step1(master_csv=args.master_csv, master_mat=args.master_mat, data_dir=args.data_dir)
