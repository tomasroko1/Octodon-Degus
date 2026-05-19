import sys
import os
# Agregar el directorio 'scripts' al path para poder importar de manera robusta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.special import gammaln
from pygam import PoissonGAM, te
from utils.data_loader import preparar_datos_posicion

def generate_all_splits(n_muestras, bin_size_sec, block_size_sec=60, n_folds=5, buffer_sec=2):
    """
    Función que asigna a cada bloque temporal su destino (held-out o fold de CV) 
    sobre la línea de tiempo original, y aplica buffers evaluando a los vecinos reales.
    
    Esto evita el bug de "distorsión temporal" que ocurría al separar primero
    el held-out (rompiendo la continuidad) y luego particionar los folds sobre
    una matriz con huecos.
    
    Esquema de asignación intercalada (para n_folds=5):
        Bloque 0 → Held-out  (rol -1)
        Bloque 1 → Fold 0    (rol 0)
        Bloque 2 → Fold 1    (rol 1)
        Bloque 3 → Fold 2    (rol 2)
        Bloque 4 → Fold 3    (rol 3)
        Bloque 5 → Fold 4    (rol 4)
        Bloque 6 → Held-out  (rol -1)
        Bloque 7 → Fold 0    (rol 0)
        ...
    
    Cada (n_folds + 1) bloques, 1 va al held-out y los otros n_folds se reparten
    de forma intercalada entre los folds de CV.

    Args:
        n_muestras: cantidad total de bines de tiempo.
        bin_size_sec: tamaño del bin temporal (ej. 0.1s).
        block_size_sec: duración de cada bloque en segundos.
        n_folds: cantidad de folds para la validación cruzada.
        buffer_sec: segundos a descartar entre bloques con rol distinto.
        
    Returns:
        folds: lista de n_folds tuplas (train_idx, test_idx) con índices sobre
               la matriz original X.
        held_out_idx: array de índices del held-out final.
        roles: array indicando el rol de cada bloque (-1=held-out, 0..n_folds-1=fold).
    """
    bines_por_bloque = int(block_size_sec / bin_size_sec)
    bines_buffer = int(buffer_sec / bin_size_sec)
    num_bloques = n_muestras // bines_por_bloque
    
    # ---------------------------------------------------------------
    # 1. Asignar un "rol" a cada bloque sobre la línea de tiempo real.
    #    Rol -1 = Held-out.  Roles 0..(n_folds-1) = Folds de CV.
    # ---------------------------------------------------------------
    roles = np.zeros(num_bloques, dtype=int)
    cv_fold_counter = 0
    ciclo = n_folds + 1  # cada ciclo: 1 held-out + n_folds bloques de CV
    
    for i in range(num_bloques):
        if i % ciclo == 0:
            roles[i] = -1  # Held-out
        else:
            roles[i] = cv_fold_counter % n_folds
            cv_fold_counter += 1
    
    # ---------------------------------------------------------------
    # 2. Construir índices con buffers basados en vecinos REALES.
    # ---------------------------------------------------------------
    held_out_idx = []
    cv_folds = {k: {'train': [], 'test': []} for k in range(n_folds)}
    # train_pool_idx almacena todos los bines que NO son held-out (sin buffer interno entre folds,
    # solo con buffer frente al held-out). Se usa para re-entrenar el modelo final.
    train_pool_idx = []
    
    for i in range(num_bloques):
        inicio = i * bines_por_bloque
        fin = (i + 1) * bines_por_bloque if i < num_bloques - 1 else n_muestras
        rol_actual = roles[i]
        
        # --- Bloque Held-Out: se toma íntegro ---
        if rol_actual == -1:
            held_out_idx.append(np.arange(inicio, fin))
            continue
        
        # --- Bloque de CV ---
        # Para el fold que coincide con su rol, este bloque es de TEST (íntegro).
        cv_folds[rol_actual]['test'].append(np.arange(inicio, fin))
        
        # Para los DEMÁS folds, este bloque es de TRAIN (se aplica buffer si 
        # el vecino real es de test para ese fold, o es held-out).
        for k in range(n_folds):
            if k == rol_actual:
                continue  # Ya lo agregamos a test
            
            inicio_train = inicio
            fin_train = fin
            
            # Recortar inicio si el bloque anterior es de TEST para el fold k, o si es Held-out
            if i > 0 and (roles[i-1] == k or roles[i-1] == -1):
                inicio_train = min(inicio + bines_buffer, fin)
            
            # Recortar fin si el bloque siguiente es de TEST para el fold k, o si es Held-out
            if i < num_bloques - 1 and (roles[i+1] == k or roles[i+1] == -1):
                fin_train = max(fin - bines_buffer, inicio_train)
            
            if inicio_train < fin_train:
                cv_folds[k]['train'].append(np.arange(inicio_train, fin_train))
        
        # --- Train pool: aplicar buffer solo frente al held-out ---
        inicio_pool = inicio
        fin_pool = fin
        
        if i > 0 and roles[i-1] == -1:
            inicio_pool = min(inicio + bines_buffer, fin)
        if i < num_bloques - 1 and roles[i+1] == -1:
            fin_pool = max(fin - bines_buffer, inicio_pool)
        
        if inicio_pool < fin_pool:
            train_pool_idx.append(np.arange(inicio_pool, fin_pool))
    
    # ---------------------------------------------------------------
    # 3. Empaquetar resultados.
    # ---------------------------------------------------------------
    held_out_final = np.concatenate(held_out_idx) if held_out_idx else np.array([], dtype=int)
    train_pool_final = np.concatenate(train_pool_idx) if train_pool_idx else np.array([], dtype=int)
    
    folds = []
    for k in range(n_folds):
        train_k = np.concatenate(cv_folds[k]['train']) if cv_folds[k]['train'] else np.array([], dtype=int)
        test_k = np.concatenate(cv_folds[k]['test']) if cv_folds[k]['test'] else np.array([], dtype=int)
        folds.append((train_k, test_k))
    
    return folds, held_out_final, train_pool_final, roles


def poisson_nll_per_sample(y_true, mu_pred):
    """
    Calcula la Negative Log-Likelihood Poisson por muestra (promedio).
    NLL = -mean( y*log(mu) - mu - log(y!) )
    """
    mu_pred = np.maximum(mu_pred, 1e-10)
    loglike = y_true * np.log(mu_pred) - mu_pred - gammaln(y_true + 1)
    return -np.mean(loglike)


def null_model_nll(y_train, y_test):
    """
    NLL del modelo nulo (Poisson homogéneo): predice la tasa media del train para todos los bines.
    Este es el baseline para el pseudo R².
    """
    lambda_nulo = np.mean(y_train)
    mu_nulo = np.full_like(y_test, lambda_nulo, dtype=float)
    return poisson_nll_per_sample(y_test, mu_nulo)


def pseudo_r2_mcfadden(nll_modelo, nll_nulo):
    """
    Pseudo R² de McFadden:  1 - (LL_modelo / LL_nulo)
    
    Como trabajamos con NLL (negativo), la fórmula equivalente es:
    R² = 1 - (NLL_modelo / NLL_nulo)
    
    Interpretación:
        0   → el modelo no mejora sobre la tasa media
        1   → predicción perfecta
        >0  → el modelo captura estructura espacial
    """
    if nll_nulo == 0:
        return 0.0
    return 1.0 - (nll_modelo / nll_nulo)


def cross_validate_gam_grid(X, Y, folds, splines_grid, lambdas_grid):
    """
    Realiza la validación cruzada y extrae el Error de Test y el EDoF.
    Devuelve la matriz de errores para el heatmap y la lista de resultados para el scatter plot.
    """
    resultados = []
    error_matrix = np.zeros((len(splines_grid), len(lambdas_grid)))

    for s_idx, n_splines in enumerate(splines_grid):
        for l_idx, lam in enumerate(lambdas_grid):
            errores_test_cv = []
            edofs_cv = []
            
            for train_idx, test_idx in folds:
                X_train, Y_train = X[train_idx], Y[train_idx]
                X_test, Y_test = X[test_idx], Y[test_idx]
                
                # entrenar modelo
                modelo = PoissonGAM(te(0, 1, n_splines=n_splines, lam=lam)).fit(X_train, Y_train)
                
                # calcular NLL en el fold de test usando loglikelihood de pygam
                # el loglikelihood devuelve un número flotante, por lo que dividimos por len(Y_test) para el promedio
                nll_val = -modelo.loglikelihood(X_test, Y_test) / len(Y_test)
                errores_test_cv.append(nll_val)
                edofs_cv.append(modelo.statistics_['edof'])
                
            nll_medio = np.mean(errores_test_cv)
            edof_medio = np.mean(edofs_cv)
            
            error_matrix[s_idx, l_idx] = nll_medio
            
            print(f"sp={n_splines:2d}, lam={lam:10.7f} | edof: {edof_medio:8.4f} | nll test: {nll_medio:.8f}")
            resultados.append((n_splines, lam, edof_medio, nll_medio))
                
    return error_matrix, resultados

def cross_validate_glm_grid(X, Y, folds, bines_grid, alphas_grid):
    """
    Realiza la validación cruzada para el GLM con bases de campanas de Gauss.
    Evalúa sobre una grilla de cantidad de bines (resolución espacial) y alphas (regularización).
    """
    pos_x = X[:, 0]
    pos_y = X[:, 1]
    
    resultados = []
    error_matrix = np.zeros((len(bines_grid), len(alphas_grid)))
    
    for b_idx, n_bines in enumerate(bines_grid):
        n_bases_x = n_bines
        n_bases_y = n_bines
        centros_x = np.linspace(np.min(pos_x), np.max(pos_x), n_bases_x)
        centros_y = np.linspace(np.min(pos_y), np.max(pos_y), n_bases_y)
        sigma_pos = (np.max(pos_x) - np.min(pos_x)) / n_bases_x
        
        X_bases_pos = np.zeros((len(pos_x), n_bases_x * n_bases_y))
        columna = 0
        for cx in centros_x:
            for cy in centros_y:
                dist_cuadrada = (pos_x - cx)**2 + (pos_y - cy)**2
                X_bases_pos[:, columna] = np.exp(- dist_cuadrada / (2 * sigma_pos**2))
                columna += 1
                
        X_glm_pos = sm.add_constant(X_bases_pos)
        
        for a_idx, alpha in enumerate(alphas_grid):
            errores_test_cv = []
            
            for train_idx, test_idx in folds:
                X_train, Y_train = X_glm_pos[train_idx], Y[train_idx]
                X_test, Y_test = X_glm_pos[test_idx], Y[test_idx]
                
                try:
                    modelo = sm.GLM(Y_train, X_train, family=sm.families.Poisson()).fit_regularized(
                        alpha=alpha, L1_wt=0.0
                    )
                    
                    mu_pred = modelo.predict(X_test)
                    mu_pred = np.maximum(mu_pred, 1e-10)
                    
                    loglike = Y_test * np.log(mu_pred) - mu_pred - gammaln(Y_test + 1)
                    nll_val = -np.mean(loglike)
                    errores_test_cv.append(nll_val)
                except Exception as e:
                    errores_test_cv.append(np.nan)
                    
            nll_medio = np.nanmean(errores_test_cv)
            error_matrix[b_idx, a_idx] = nll_medio
            
            print(f"GLM | bines={n_bines:2d}x{n_bines:2d}, alpha={alpha:7.4f} | nll test: {nll_medio:.8f}")
            resultados.append((n_bines, alpha, nll_medio))
            
    return error_matrix, resultados


def retrain_best_gam(X_train, Y_train, best_splines, best_lam):
    """Re-entrena el mejor GAM en todo el train_pool."""
    modelo = PoissonGAM(te(0, 1, n_splines=best_splines, lam=best_lam)).fit(X_train, Y_train)
    return modelo


def retrain_best_glm(X_train, Y_train, best_bines, best_alpha):
    """Re-entrena el mejor GLM en todo el train_pool."""
    pos_x = X_train[:, 0]
    pos_y = X_train[:, 1]
    
    centros_x = np.linspace(np.min(pos_x), np.max(pos_x), best_bines)
    centros_y = np.linspace(np.min(pos_y), np.max(pos_y), best_bines)
    sigma_pos = (np.max(pos_x) - np.min(pos_x)) / best_bines
    
    X_bases = np.zeros((len(pos_x), best_bines * best_bines))
    col = 0
    for cx in centros_x:
        for cy in centros_y:
            dist_sq = (pos_x - cx)**2 + (pos_y - cy)**2
            X_bases[:, col] = np.exp(-dist_sq / (2 * sigma_pos**2))
            col += 1
    
    X_glm = sm.add_constant(X_bases)
    modelo = sm.GLM(Y_train, X_glm, family=sm.families.Poisson()).fit_regularized(
        alpha=best_alpha, L1_wt=0.0
    )
    
    # Devolvemos también los centros y sigma para poder predecir en el held-out
    return modelo, centros_x, centros_y, sigma_pos


def predict_glm_on_new_data(modelo_glm, X_new, centros_x, centros_y, sigma_pos):
    """Genera predicciones del GLM sobre datos nuevos (held-out)."""
    pos_x = X_new[:, 0]
    pos_y = X_new[:, 1]
    n_bases = len(centros_x) * len(centros_y)
    
    X_bases = np.zeros((len(pos_x), n_bases))
    col = 0
    for cx in centros_x:
        for cy in centros_y:
            dist_sq = (pos_x - cx)**2 + (pos_y - cy)**2
            X_bases[:, col] = np.exp(-dist_sq / (2 * sigma_pos**2))
            col += 1
    
    X_glm = sm.add_constant(X_bases, has_constant='add')
    return modelo_glm.predict(X_glm)


def plot_cv_heatmap(error_matrix, x_grid, y_grid, title='Negative Log-Likelihood (Blue is Better - Lower Error)', xlabel='lambda (Smoothing)', ylabel='n_splines (Resolution)'):
    """Grafica el mapa de calor de los errores de validación cruzada."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(error_matrix, annot=True, fmt=".6f", 
                xticklabels=x_grid, yticklabels=y_grid,
                cmap='jet')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("  PIPELINE: CV + HELD-OUT + PSEUDO R²")
    print("=" * 60)
    
    # ==========================================
    # 1. CARGAR DATOS
    # ==========================================
    print("\n1. Cargando datos...")
    sesion, tetrodo, neurona = 2, 3, 2
    bin_size = 0.1
    X, Y = preparar_datos_posicion(sesion, tetrodo, neurona, bin_size)
    print(f"   Total de muestras: {len(X)}")

    # ==========================================
    # 2. PARTICIÓN UNIFICADA: HELD-OUT + FOLDS DE CV
    #    (Todo se asigna sobre la línea de tiempo original para evitar
    #     distorsión temporal en los buffers)
    # ==========================================
    print("\n2. Partición unificada (held-out + CV folds con buffer de 2s)...")
    folds, held_out_idx, train_pool_idx, roles = generate_all_splits(
        n_muestras=len(X),
        bin_size_sec=bin_size,
        block_size_sec=60,
        n_folds=5,
        buffer_sec=2
    )
    
    X_pool, Y_pool = X[train_pool_idx], Y[train_pool_idx]
    X_held, Y_held = X[held_out_idx], Y[held_out_idx]
    
    n_held_out_bloques = np.sum(roles == -1)
    n_cv_bloques = np.sum(roles >= 0)
    print(f"   Bloques totales: {len(roles)} | Held-out: {n_held_out_bloques} | CV: {n_cv_bloques}")
    print(f"   Train pool: {len(X_pool)} muestras ({100*len(X_pool)/len(X):.1f}%)")
    print(f"   Held-out:   {len(X_held)} muestras ({100*len(X_held)/len(X):.1f}%)")
    for k, (tr, te) in enumerate(folds):
        print(f"   Fold {k}: train={len(tr)} | test={len(te)}")

    # ==========================================
    # 3. CV PARA SELECCIÓN DE HIPERPARÁMETROS - GAM
    # ==========================================
    print(f"\n3. CV para selección de hiperparámetros GAM...")
    splines_a_probar = [4, 5, 6, 7]
    #lambdas_a_probar = [0.01, 0.1, 0.5]
    lambdas_a_probar = np.logspace(-4, 2, 11)
    
    error_matrix_gam, resultados_gam = cross_validate_gam_grid(
        X, Y, folds, splines_a_probar, lambdas_a_probar
    )
    
    mejor_gam_cv = sorted(resultados_gam, key=lambda x: x[3])[0]
    best_sp, best_lam = int(mejor_gam_cv[0]), mejor_gam_cv[1]
    print(f"\n   [CV] Mejor GAM: splines={best_sp}, lambda={best_lam} | NLL CV={mejor_gam_cv[3]:.8f}")

    # ==========================================
    # 4. CV PARA SELECCIÓN DE HIPERPARÁMETROS - GLM
    # ==========================================
    print(f"\n4. CV para selección de hiperparámetros GLM...")
    bines_a_probar = [4, 5, 6, 7]
    #alphas_a_probar = [0.00010, 0.00015, 0.00020]
    alphas_a_probar = np.logspace(-6, 0, 11)

    error_matrix_glm, resultados_glm = cross_validate_glm_grid(
        X, Y, folds, bines_a_probar, alphas_a_probar
    )
    
    mejor_glm_cv = sorted(resultados_glm, key=lambda x: x[2])[0]
    best_bines, best_alpha = int(mejor_glm_cv[0]), mejor_glm_cv[1]
    print(f"\n   [CV] Mejor GLM: bines={best_bines}x{best_bines}, alpha={best_alpha:.4f} | NLL CV={mejor_glm_cv[2]:.8f}")

    # ==========================================
    # 5. RE-ENTRENAR MEJORES MODELOS EN TODO EL TRAIN POOL
    # ==========================================
    print(f"\n5. Re-entrenando mejores modelos en todo el train pool...")
    
    gam_final = retrain_best_gam(X_pool, Y_pool, best_sp, best_lam)
    print(f"   GAM final entrenado (splines={best_sp}, lambda={best_lam})")
    
    glm_final, cx, cy, sigma = retrain_best_glm(X_pool, Y_pool, best_bines, best_alpha)
    print(f"   GLM final entrenado (bines={best_bines}, alpha={best_alpha})")

    # ==========================================
    # 6. EVALUACIÓN EN HELD-OUT: NLL + PSEUDO R²
    # ==========================================
    print(f"\n6. Evaluando en held-out set ({len(X_held)} muestras)...")
    
    # Modelo nulo: tasa media del train pool
    nll_nulo = null_model_nll(Y_pool, Y_held)
    print(f"\n   Modelo Nulo (tasa media = {np.mean(Y_pool):.4f}):")
    print(f"   NLL held-out nulo: {nll_nulo:.8f}")
    
    # GAM en held-out
    mu_gam = gam_final.predict(X_held)
    nll_gam_held = poisson_nll_per_sample(Y_held, mu_gam)
    r2_gam = pseudo_r2_mcfadden(nll_gam_held, nll_nulo)
    
    # GLM en held-out
    mu_glm = predict_glm_on_new_data(glm_final, X_held, cx, cy, sigma)
    nll_glm_held = poisson_nll_per_sample(Y_held, mu_glm)
    r2_glm = pseudo_r2_mcfadden(nll_glm_held, nll_nulo)

    # ==========================================
    # 7. REPORTE FINAL
    # ==========================================
    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES EN HELD-OUT")
    print("=" * 60)
    print(f"{'Métrica':<25} {'Modelo Nulo':>14} {'GAM':>14} {'GLM':>14}")
    print("-" * 67)
    print(f"{'NLL (held-out)':<25} {nll_nulo:>14.8f} {nll_gam_held:>14.8f} {nll_glm_held:>14.8f}")
    print(f"{'Pseudo R² (McFadden)':<25} {'---':>14} {r2_gam:>14.6f} {r2_glm:>14.6f}")
    print("-" * 67)
    
    if nll_gam_held < nll_glm_held:
        ganador = "GAM"
        diff = nll_glm_held - nll_gam_held
    else:
        ganador = "GLM"
        diff = nll_gam_held - nll_glm_held
    
    print(f"\n   Ganador: {ganador} (ventaja NLL: {diff:.8f})")
    print(f"   Pseudo R² GAM: {r2_gam:.4f} ({r2_gam*100:.2f}% de varianza explicada vs. modelo nulo)")
    print(f"   Pseudo R² GLM: {r2_glm:.4f} ({r2_glm*100:.2f}% de varianza explicada vs. modelo nulo)")

    # ==========================================
    # 8. HEATMAPS DE CV (para referencia)
    # ==========================================
    print("\nGenerando mapa de calor de NLL para GAM...")
    plot_cv_heatmap(
        error_matrix_gam, 
        lambdas_a_probar, 
        splines_a_probar, 
        title='GAM CV NLL (Blue is Better)', 
        xlabel='lambda (Smoothing)', 
        ylabel='n_splines (Resolution)'
    )
    
    print("Generando mapa de calor de NLL para GLM...")
    plot_cv_heatmap(
        error_matrix_glm, 
        alphas_a_probar, 
        bines_a_probar, 
        title='GLM CV NLL (Blue is Better)', 
        xlabel='alpha (Regularization)', 
        ylabel='n_bases (Resolution)'
    )

if __name__ == "__main__":
    main()