#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2: GAMs Analysis
Reads the output from Step 1 (viewpoint_results.csv), filters for significant cells,
and runs the cross-validated Generalized Additive Models to compute tuning parameters
and Shapley values.
Outputs: results/gam_results.csv
"""

import os
import sys
import csv
import contextlib
import argparse
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_cell_data
from convert_mat_to_csv import convert_mat_to_csv
from utils.cross_validation import (
    generate_all_splits,
    find_optimal_lambda_dynamic,
    retrain_best_gam,
    poisson_nll_per_sample,
    null_model_nll,
    pseudo_r2_mcfadden
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def process_cell(cell_id, best_x, best_y, current_idx, total_cells, data_dir=None):
    print(f"\n[{current_idx}/{total_cells}] Procesando {cell_id}...")
    
    data = load_cell_data(cell_id, data_dir=data_dir, bin_size_sec=0.1)
    if data is None:
        print(f"Omitiendo {cell_id} por error de carga.")
        return None
        
    x_bins, y_bins, ang_bins_rad, spikes = data
    
    # Filtrar posibles NaNs en los ángulos
    valid_mask = ~np.isnan(ang_bins_rad)
    x_bins = x_bins[valid_mask]
    y_bins = y_bins[valid_mask]
    ang_bins_rad = ang_bins_rad[valid_mask]
    spikes = spikes[valid_mask]

    if len(spikes) < 100 or np.sum(spikes) < 10:
        print(f"Omitiendo {cell_id} por no tener suficientes datos válidos o spikes ({np.sum(spikes)}).")
        return { 'Cell_ID': cell_id, 'Status': 'skipped_low_spikes' }

    # Computar Viewpoint
    dy = best_y - y_bins
    dx = best_x - x_bins
    ang_hacia_punto = np.arctan2(dy, dx)
    theta_corrected = np.mod(ang_bins_rad - ang_hacia_punto, 2 * np.pi)
    
    hd_rad = np.mod(ang_bins_rad, 2 * np.pi)
    X_all = np.column_stack((x_bins, y_bins, theta_corrected, np.zeros_like(x_bins), hd_rad))
    
    X_pos = X_all[:, [0, 1]]
    X_view = X_all[:, [2]]
    X_hd = X_all[:, [4]]
    X_pos_view = X_all[:, [0, 1, 2]]
    
    folds, held_out_idx, train_pool_idx, _ = generate_all_splits(
        n_muestras=len(spikes),
        bin_size_sec=0.1,
        block_size_sec=60,
        n_folds=5,
        buffer_sec=2
    )
    
    X_pool = X_all[train_pool_idx]
    Y_pool = spikes[train_pool_idx]
    X_held = X_all[held_out_idx]
    Y_held = spikes[held_out_idx]
    
    # Búsqueda dinámica de lambda
    try:
        with open(os.devnull, 'w') as f_null:
            with contextlib.redirect_stdout(f_null):
                lam_pos, edof_pos, _, _ = find_optimal_lambda_dynamic(
                    X_pos, spikes, folds, n_splines=9, model_type="pos", lam_start=1e-3
                )
                lam_view, edof_view, _, _ = find_optimal_lambda_dynamic(
                    X_view, spikes, folds, n_splines=9, model_type="view_angle", lam_start=1e-3
                )
                lam_hd, edof_hd, _, _ = find_optimal_lambda_dynamic(
                    X_hd, spikes, folds, n_splines=9, model_type="hd", lam_start=1e-3
                )
                lam_pos_view, edof_pos_view, _, _ = find_optimal_lambda_dynamic(
                    X_pos_view, spikes, folds, n_splines=9, model_type="pos_view_angle", lam_start=[lam_pos, lam_view]
                )
                
        # Re-entrenar modelos sobre el train pool
        modelo_pos = retrain_best_gam(X_pool[:, [0, 1]], Y_pool, 9, lam_pos, model_type="pos")
        modelo_view = retrain_best_gam(X_pool[:, [2]], Y_pool, 9, lam_view, model_type="view_angle")
        modelo_hd = retrain_best_gam(X_pool[:, [4]], Y_pool, 9, lam_hd, model_type="hd")
        modelo_pos_view = retrain_best_gam(X_pool[:, [0, 1, 2]], Y_pool, 9, lam_pos_view, model_type="pos_view_angle")
        
        # Evaluar NLL final en held-out
        nll_pos = poisson_nll_per_sample(Y_held, modelo_pos.predict(X_held[:, [0, 1]]))
        nll_view = poisson_nll_per_sample(Y_held, modelo_view.predict(X_held[:, [2]]))
        nll_hd = poisson_nll_per_sample(Y_held, modelo_hd.predict(X_held[:, [4]]))
        nll_pos_view = poisson_nll_per_sample(Y_held, modelo_pos_view.predict(X_held[:, [0, 1, 2]]))
        
        null_nll = null_model_nll(Y_pool, Y_held)
        
        pos_r2 = pseudo_r2_mcfadden(nll_pos, null_nll)
        view_r2 = pseudo_r2_mcfadden(nll_view, null_nll)
        hd_r2 = pseudo_r2_mcfadden(nll_hd, null_nll)
        pos_view_r2 = pseudo_r2_mcfadden(nll_pos_view, null_nll)
        
        shapley_pos = 0.5 * pos_r2 + 0.5 * (pos_view_r2 - view_r2)
        shapley_view = 0.5 * view_r2 + 0.5 * (pos_view_r2 - pos_r2)
    except Exception as e:
        print(f"Error entrenando GAM para {cell_id}: {e}")
        return { 'Cell_ID': cell_id, 'Status': f'error_gam_{type(e).__name__}' }
            
    if isinstance(lam_pos_view, (list, tuple)):
        lam_pos_view_str = f"{lam_pos_view[0]:.4g};{lam_pos_view[1]:.4g}"
    else:
        lam_pos_view_str = f"{lam_pos_view:.4g}"
        
    print(f" -> Pos:      lam={lam_pos:.5g}, Held-Out R2={pos_r2:.4f}")
    print(f" -> View:     lam={lam_view:.5g}, Held-Out R2={view_r2:.4f}")
    print(f" -> HD:       lam={lam_hd:.5g}, Held-Out R2={hd_r2:.4f}")
    print(f" -> Pos+View: lam={lam_pos_view_str}, Held-Out R2={pos_view_r2:.4f}")
    print(f" -> Shapley:  Pos={shapley_pos:.5f}, View={shapley_view:.5f}")
    
    return {
        'Cell_ID': cell_id,
        'Status': 'success',
        'Best_X': best_x,
        'Best_Y': best_y,
        'Null_NLL': null_nll,
        'Pos_Lambda': lam_pos,
        'Pos_NLL': nll_pos,
        'Pos_EDoF': edof_pos,
        'Pos_PseudoR2': pos_r2,
        'View_Lambda': lam_view,
        'View_NLL': nll_view,
        'View_EDoF': edof_view,
        'View_PseudoR2': view_r2,
        'HD_Lambda': lam_hd,
        'HD_NLL': nll_hd,
        'HD_EDoF': edof_hd,
        'HD_PseudoR2': hd_r2,
        'PosView_Lambda': lam_pos_view_str,
        'PosView_NLL': nll_pos_view,
        'PosView_EDoF': edof_pos_view,
        'PosView_PseudoR2': pos_view_r2,
        'Shapley_Pos': shapley_pos,
        'Shapley_View': shapley_view
    }

def run_step2(data_dir=None):
    step1_csv = os.path.join(RESULTS_DIR, 'viewpoint_results.csv')
    step1_mat = os.path.join(RESULTS_DIR, 'viewpoint_results.mat')
    
    if not os.path.exists(step1_csv):
        if os.path.exists(step1_mat):
            print("Detectado output de MATLAB. Convirtiendo a CSV automáticamente...")
            if not convert_mat_to_csv(step1_mat, step1_csv):
                return
        else:
            print(f"Error: No se encontró el output del Step 1 en {step1_mat} ni {step1_csv}")
            print("Debes correr 'step1_viewpoint_analysis.m' en MATLAB primero.")
            return
        
    df_sig = []
    with open(step1_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # En CSV los booleanos se leen como strings 'True'/'False'
            if row.get('is_significant') == 'True' and row.get('status') == 'success':
                df_sig.append(row)
            elif 'is_significant' not in row and row.get('status') == 'success':
                df_sig.append(row)
        
    total = len(df_sig)
    print(f"Se encontraron {total} celulas significativas del Step 1 para procesar GAMs.")
    
    out_csv = os.path.join(RESULTS_DIR, 'gam_results.csv')
    results = []
    processed_cells = set()
    
    if os.path.exists(out_csv):
        try:
            with open(out_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                results = list(reader)
            
            if fieldnames and 'PosView_NLL' in fieldnames:
                processed_cells = set(r['Cell_ID'] for r in results)
                print(f"Retomando desde {len(results)} celulas ya procesadas en {out_csv}.")
            else:
                print("CSV previo incompatible. Se reiniciará.")
                results = []
        except Exception:
            pass
            
    t0 = time.time()
    for i, row in enumerate(df_sig, 1):
        cell_id = row['Cell_ID']
        best_x = float(row['Best_X'])
        best_y = float(row['Best_Y'])
        
        if cell_id in processed_cells:
            continue
            
        res = process_cell(cell_id, best_x, best_y, i, total, data_dir=data_dir)
        if res:
            results.append(res)
            # Para que DictWriter no falle si el dict res tiene solo 'Cell_ID' y 'Status'
            # y 'results[-1]' (que puede ser exitoso) tiene mas keys.
            # Escribimos los fieldnames acumulando las keys descubiertas o con un dict de fallback.
            all_keys = set()
            for r in results:
                all_keys.update(r.keys())
            
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(all_keys), extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            
    print(f"\nStep 2 Finalizado en {time.time() - t0:.1f}s. Resultados: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: GAM Analysis")
    parser.add_argument("--data_dir", type=str, default=None, help="Carpeta de datos")
    args = parser.parse_args()
    
    run_step2(data_dir=args.data_dir)
