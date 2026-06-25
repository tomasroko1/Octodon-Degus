import numpy as np
from scipy.special import gammaln
from pygam import PoissonGAM, te, s

def _unpack_lambdas(lam, model_type):
    lam_pos = lam
    lam_view = lam
    lam_hd = lam
    lam_dist = lam
    
    if isinstance(lam, (tuple, list)):
        if model_type == "pos_view_angle":
            lam_pos, lam_view = lam[0], lam[1]
        elif model_type == "pos_dist":
            lam_pos, lam_dist = lam[0], lam[1]
        elif model_type == "view_angle_dist":
            lam_view, lam_dist = lam[0], lam[1]
        elif model_type == "pos_view_angle_dist":
            lam_pos, lam_view, lam_dist = lam[0], lam[1], lam[2]
        elif model_type == "pos_hd":
            lam_pos, lam_hd = lam[0], lam[1]
        elif model_type == "pos_hd_dist":
            lam_pos, lam_hd, lam_dist = lam[0], lam[1], lam[2]
        elif model_type == "view_hd":
            lam_view, lam_hd = lam[0], lam[1]
        elif model_type == "pos_view_hd":
            lam_pos, lam_view, lam_hd = lam[0], lam[1], lam[2]
        elif model_type == "view_hd_dist":
            lam_view, lam_hd, lam_dist = lam[0], lam[1], lam[2]
        elif model_type == "pos_view_hd_dist":
            lam_pos, lam_view, lam_hd, lam_dist = lam[0], lam[1], lam[2], lam[3]
        elif model_type.startswith("shapley_"):
            components = model_type.split("_")[1:]
            active_lams = {}
            idx_lam = 0
            for comp in ['pos', 'hd', 'view', 'dist']:
                if comp in components and idx_lam < len(lam):
                    active_lams[comp] = lam[idx_lam]
                    idx_lam += 1
                else:
                    active_lams[comp] = lam[0]
            lam_pos = active_lams.get('pos', lam[0])
            lam_view = active_lams.get('view', lam[0])
            lam_hd = active_lams.get('hd', lam[0])
            lam_dist = active_lams.get('dist', lam[0])
            
    return lam_pos, lam_view, lam_hd, lam_dist


def _build_gam_formula(model_type, n_splines, lam_pos, lam_view, lam_hd, lam_dist, n_splines_circular=None):
    if n_splines_circular is None:
        n_splines_circular = n_splines
        
    if model_type == "pos":
        return te(0, 1, n_splines=n_splines, lam=lam_pos)
    elif model_type == "view_angle" or model_type == "hd":
        return s(0, basis='cp', n_splines=n_splines_circular, lam=lam_view if model_type == "view_angle" else lam_hd, edge_knots=[0.0, 2*np.pi])
    elif model_type == "view_angle_dist":
        return s(0, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(1, n_splines=n_splines, lam=lam_dist)
    elif model_type == "pos_view_angle":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi])
    elif model_type == "pos_view_angle_dist":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(3, n_splines=n_splines, lam=lam_dist)
    elif model_type == "pos_hd":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi])
    elif model_type == "pos_dist":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, n_splines=n_splines, lam=lam_dist)
    elif model_type == "pos_hd_dist":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi]) + s(3, n_splines=n_splines, lam=lam_dist)
    elif model_type == "dist":
        return s(0, n_splines=n_splines, lam=lam_dist)
    elif model_type == "view_hd":
        return s(0, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(1, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi])
    elif model_type == "pos_view_hd":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(3, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi])
    elif model_type == "view_hd_dist":
        return s(0, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(1, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi]) + s(2, n_splines=n_splines, lam=lam_dist)
    elif model_type == "pos_view_hd_dist":
        return te(0, 1, n_splines=n_splines, lam=lam_pos) + s(2, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]) + s(3, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi]) + s(4, n_splines=n_splines, lam=lam_dist)
    elif model_type.startswith("shapley_"):
        components = model_type.split("_")[1:]
        terms = []
        if "pos" in components:
            terms.append(te(0, 1, n_splines=n_splines, lam=lam_pos))
        if "hd" in components:
            terms.append(s(2, basis='cp', n_splines=n_splines_circular, lam=lam_hd, edge_knots=[0.0, 2*np.pi]))
        if "view" in components:
            terms.append(s(3, basis='cp', n_splines=n_splines_circular, lam=lam_view, edge_knots=[0.0, 2*np.pi]))
        if "dist" in components:
            terms.append(s(4, n_splines=n_splines, lam=lam_dist))
        
        if not terms:
            raise ValueError(f"model_type '{model_type}' no contiene componentes validos")
        formula = terms[0]
        for t in terms[1:]:
            formula += t
        return formula
    else:
        raise ValueError(f"model_type '{model_type}' no soportado")


def generate_all_splits(n_muestras, bin_size_sec, block_size_sec=60, n_folds=5, buffer_sec=2):
    bines_por_bloque = int(block_size_sec / bin_size_sec)
    bines_buffer = int(buffer_sec / bin_size_sec)
    num_bloques = n_muestras // bines_por_bloque
    
    roles = np.zeros(num_bloques, dtype=int)
    cv_fold_counter = 0
    ciclo = n_folds + 1
    
    for i in range(num_bloques):
        if i % ciclo == 0:
            roles[i] = -1
        else:
            roles[i] = cv_fold_counter % n_folds
            cv_fold_counter += 1
            
    held_out_idx = []
    cv_folds = {k: {'train': [], 'test': []} for k in range(n_folds)}
    train_pool_idx = []
    
    for i in range(num_bloques):
        inicio = i * bines_por_bloque
        fin = (i + 1) * bines_por_bloque if i < num_bloques - 1 else n_muestras
        rol_actual = roles[i]
        
        if rol_actual == -1:
            held_out_idx.append(np.arange(inicio, fin))
            continue
            
        cv_folds[rol_actual]['test'].append(np.arange(inicio, fin))
        
        for k in range(n_folds):
            if k == rol_actual:
                continue
                
            inicio_train = inicio
            fin_train = fin
            
            if i > 0 and (roles[i-1] == k or roles[i-1] == -1):
                inicio_train = min(inicio + bines_buffer, fin)
                
            if i < num_bloques - 1 and (roles[i+1] == k or roles[i+1] == -1):
                fin_train = max(fin - bines_buffer, inicio_train)
                
            if inicio_train < fin_train:
                cv_folds[k]['train'].append(np.arange(inicio_train, fin_train))
                
        inicio_pool = inicio
        fin_pool = fin
        
        if i > 0 and roles[i-1] == -1:
            inicio_pool = min(inicio + bines_buffer, fin)
        if i < num_bloques - 1 and roles[i+1] == -1:
            fin_pool = max(fin - bines_buffer, inicio_pool)
            
        if inicio_pool < fin_pool:
            train_pool_idx.append(np.arange(inicio_pool, fin_pool))
            
    held_out_final = np.concatenate(held_out_idx) if held_out_idx else np.array([], dtype=int)
    train_pool_final = np.concatenate(train_pool_idx) if train_pool_idx else np.array([], dtype=int)
    
    folds = []
    for k in range(n_folds):
        train_k = np.concatenate(cv_folds[k]['train']) if cv_folds[k]['train'] else np.array([], dtype=int)
        test_k = np.concatenate(cv_folds[k]['test']) if cv_folds[k]['test'] else np.array([], dtype=int)
        folds.append((train_k, test_k))
        
    return folds, held_out_final, train_pool_final, roles


def find_optimal_lambda_dynamic(X, Y, folds, n_splines, model_type="pos", lam_start=1e-3, factor=1.5, max_steps=100, patience=3):
    mejor_nll = float('inf')
    mejor_lam = None
    mejor_edof = None
    pasos_sin_mejora = 0
    lam_actual = lam_start
    resultados_busqueda = []
    
    for i in range(max_steps):
        errores_test_cv = []
        edofs_cv = []
        
        for train_idx, test_idx in folds:
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]
            
            lam_pos, lam_view, lam_hd, lam_dist = _unpack_lambdas(lam_actual, model_type)
            formula = _build_gam_formula(model_type, n_splines, lam_pos, lam_view, lam_hd, lam_dist)
                
            modelo = PoissonGAM(formula).fit(X_train, Y_train)
            
            nll_val = -modelo.loglikelihood(X_test, Y_test) / len(Y_test)
            errores_test_cv.append(nll_val)
            edofs_cv.append(modelo.statistics_['edof'])
            
        nll_medio = np.mean(errores_test_cv)
        edof_medio = np.mean(edofs_cv)
        resultados_busqueda.append((lam_actual, edof_medio, nll_medio))
        
        if nll_medio < mejor_nll:
            mejor_nll = nll_medio
            mejor_lam = lam_actual
            mejor_edof = edof_medio
            pasos_sin_mejora = 0
        else:
            pasos_sin_mejora += 1
            
        if pasos_sin_mejora >= patience:
            break
            
        lam_actual *= factor
        
    return mejor_lam, mejor_edof, mejor_nll, resultados_busqueda


def poisson_nll_per_sample(y_true, mu_pred):
    mu_pred = np.maximum(mu_pred, 1e-10)
    loglike = y_true * np.log(mu_pred) - mu_pred - gammaln(y_true + 1)
    return -np.mean(loglike)


def null_model_nll(y_train, y_test):
    lambda_nulo = np.mean(y_train)
    mu_nulo = np.full_like(y_test, lambda_nulo, dtype=float)
    return poisson_nll_per_sample(y_test, mu_nulo)


def pseudo_r2_mcfadden(nll_modelo, nll_nulo):
    if nll_nulo == 0:
        return 0.0
    return 1.0 - (nll_modelo / nll_nulo)


def cross_validate_gam_grid(X, Y, folds, splines_grid, lambdas_grid, model_type="pos_view_angle"):
    resultados = []
    error_matrix = np.zeros((len(splines_grid), len(lambdas_grid)))

    for s_idx, n_splines in enumerate(splines_grid):
        for l_idx, lam in enumerate(lambdas_grid):
            errores_test_cv = []
            edofs_cv = []
            
            for train_idx, test_idx in folds:
                X_train, Y_train = X[train_idx], Y[train_idx]
                X_test, Y_test = X[test_idx], Y[test_idx]
                
                lam_pos, lam_view, lam_hd, lam_dist = _unpack_lambdas(lam, model_type)
                formula = _build_gam_formula(model_type, n_splines, lam_pos, lam_view, lam_hd, lam_dist)
                
                modelo = PoissonGAM(formula).fit(X_train, Y_train)
                
                nll_val = -modelo.loglikelihood(X_test, Y_test) / len(Y_test)
                errores_test_cv.append(nll_val)
                edofs_cv.append(modelo.statistics_['edof'])
                
            nll_medio = np.mean(errores_test_cv)
            nll_std = np.std(errores_test_cv)
            nll_sem = nll_std / np.sqrt(len(errores_test_cv))
            edof_medio = np.mean(edofs_cv)
            
            error_matrix[s_idx, l_idx] = nll_medio
            
            if isinstance(lam, (tuple, list)):
                lam_str = "[" + ", ".join(f"{l:.4g}" for l in lam) + "]"
            else:
                lam_str = f"{lam:10.7f}"
                
            print(f"sp={n_splines:2d}, lam={lam_str} | edof: {edof_medio:8.4f} | nll test: {nll_medio:.8f} (± {nll_sem:.8f})")
            resultados.append((n_splines, lam, edof_medio, nll_medio, nll_sem, nll_std))
                
    return error_matrix, resultados


def cross_validate_glm_grid(X, Y, folds, bines_grid, alphas_grid):
    import statsmodels.api as sm
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
            nll_std = np.nanstd(errores_test_cv)
            nll_sem = nll_std / np.sqrt(np.sum(~np.isnan(errores_test_cv))) if np.sum(~np.isnan(errores_test_cv)) > 0 else np.nan
            error_matrix[b_idx, a_idx] = nll_medio
            
            print(f"GLM | bins={n_bines:2d}x{n_bines:2d}, alpha={alpha:7.4f} | nll test: {nll_medio:.8f} (± {nll_sem:.8f})")
            resultados.append((n_bines, alpha, nll_medio, nll_sem, nll_std))
            
    return error_matrix, resultados


def select_best_1se_gam(resultados):
    resultados_ordenados = sorted(resultados, key=lambda x: x[3])
    mejor_min = resultados_ordenados[0]
    nll_min = mejor_min[3]
    sem_min = mejor_min[4]
    
    umbral = nll_min + sem_min
    candidatos = [r for r in resultados if r[3] <= umbral]
    mejor_1se = sorted(candidatos, key=lambda x: x[2])[0]
    
    return mejor_min, mejor_1se, umbral


def select_best_1se_glm(resultados):
    resultados_ordenados = sorted(resultados, key=lambda x: x[2])
    mejor_min = resultados_ordenados[0]
    nll_min = mejor_min[2]
    sem_min = mejor_min[3]
    
    umbral = nll_min + sem_min
    candidatos = [r for r in resultados if r[2] <= umbral]
    mejor_1se = sorted(candidatos, key=lambda x: (x[0], -x[1]))[0]
    
    return mejor_min, mejor_1se, umbral


def retrain_best_gam(X_train, Y_train, best_splines, best_lam, model_type="pos_view_angle"):
    lam_pos, lam_view, lam_hd, lam_dist = _unpack_lambdas(best_lam, model_type)
    # The cluster version uses 8 splines for circular variables in retraining.
    formula = _build_gam_formula(model_type, best_splines, lam_pos, lam_view, lam_hd, lam_dist, n_splines_circular=8)
    modelo = PoissonGAM(formula).fit(X_train, Y_train)
    return modelo


def retrain_best_glm(X_train, Y_train, best_bines, best_alpha):
    import statsmodels.api as sm
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
    
    return modelo, centros_x, centros_y, sigma_pos


def predict_glm_on_new_data(modelo_glm, X_new, centros_x, centros_y, sigma_pos):
    import statsmodels.api as sm
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


def plot_cv_errorbars_gam(resultados, title="GAM: EDoF vs NLL con Error Estandar (1-SE Rule)"):
    import matplotlib.pyplot as plt
    splines = np.array([r[0] for r in resultados])
    lambdas = np.array([r[1] for r in resultados])
    edofs = np.array([r[2] for r in resultados])
    errores = np.array([r[3] for r in resultados])
    sems = np.array([r[4] for r in resultados])
    
    mejor_min, mejor_1se, umbral = select_best_1se_gam(resultados)
    
    plt.figure(figsize=(11, 7))
    unique_splines = sorted(list(set(splines)))
    colores = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_splines)))
    
    for sp, color in zip(unique_splines, colores):
        idx = [i for i, s in enumerate(splines) if s == sp]
        edof_sp = edofs[idx]
        err_sp = errores[idx]
        sem_sp = sems[idx]
        
        sort_idx = np.argsort(edof_sp)
        
        plt.errorbar(
            edof_sp[sort_idx], err_sp[sort_idx], yerr=sem_sp[sort_idx],
            fmt='o', color=color, ecolor='gray', elinewidth=1.5,
            capsize=3, markersize=7, label=f'{sp} splines', zorder=3, alpha=0.85
        )
        
        plt.plot(
            edof_sp[sort_idx], err_sp[sort_idx], color=color, linestyle='-',
            linewidth=1.5, alpha=0.4, zorder=2
        )
        
    plt.axhline(y=mejor_min[3], color='red', linestyle='--', alpha=0.6, label=f'Min NLL ({mejor_min[3]:.5f})')
    plt.axhline(y=umbral, color='forestgreen', linestyle=':', alpha=0.7, linewidth=2, label=f'1-SE Threshold ({umbral:.5f})')
    
    plt.scatter([mejor_min[2]], [mejor_min[3]], color='red', marker='*', s=350, edgecolor='black', zorder=5, label='Best Min NLL')
    plt.scatter([mejor_1se[2]], [mejor_1se[3]], color='forestgreen', marker='D', s=200, edgecolor='black', zorder=5, label=f'1-SE Rule Pick (sp={mejor_1se[0]}, lam={mejor_1se[1]:.4g})')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Effective Degrees of Freedom (EDoF)', fontsize=12)
    plt.ylabel('Mean Test NLL (± SEM)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_cv_errorbars_glm(resultados, title="GLM: Bins vs NLL with Standard Error (1-SE Rule)"):
    import matplotlib.pyplot as plt
    bines = np.array([r[0] for r in resultados])
    alphas = np.array([r[1] for r in resultados])
    errores = np.array([r[2] for r in resultados])
    sems = np.array([r[3] for r in resultados])
    
    mejor_min, mejor_1se, umbral = select_best_1se_glm(resultados)
    
    plt.figure(figsize=(11, 7))
    unique_bines = sorted(list(set(bines)))
    colores = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique_bines)))
    
    for bn, color in zip(unique_bines, colores):
        idx = [i for i, b in enumerate(bines) if b == bn]
        alpha_bn = alphas[idx]
        err_bn = errores[idx]
        sem_bn = sems[idx]
        
        sort_idx = np.argsort(alpha_bn)
        
        plt.errorbar(
            alpha_bn[sort_idx], err_bn[sort_idx], yerr=sem_bn[sort_idx],
            fmt='o', color=color, ecolor='gray', elinewidth=1.5,
            capsize=3, markersize=7, label=f'{bn}x{bn} bases', zorder=3, alpha=0.85
        )
        
        plt.plot(
            alpha_bn[sort_idx], err_bn[sort_idx], color=color, linestyle='-',
            linewidth=1.5, alpha=0.4, zorder=2
        )
        
    plt.axhline(y=mejor_min[2], color='red', linestyle='--', alpha=0.6, label=f'Min NLL ({mejor_min[2]:.5f})')
    plt.axhline(y=umbral, color='forestgreen', linestyle=':', alpha=0.7, linewidth=2, label=f'1-SE Threshold ({umbral:.5f})')
    
    plt.scatter([mejor_min[1]], [mejor_min[2]], color='red', marker='*', s=350, edgecolor='black', zorder=5, label='Best Min NLL')
    plt.scatter([mejor_1se[1]], [mejor_1se[2]], color='forestgreen', marker='D', s=200, edgecolor='black', zorder=5, label=f'1-SE Rule Pick (bins={mejor_1se[0]}, alpha={mejor_1se[1]:.4g})')
    
    plt.xscale('log')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Alpha (Regularization - Log Scale)', fontsize=12)
    plt.ylabel('Mean Test NLL (± SEM)', fontsize=12)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_cv_heatmap(error_matrix, x_grid, y_grid, title='Negative Log-Likelihood', xlabel='lambda (Smoothing)', ylabel='n_splines (Resolution)'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    x_labels = []
    for val in x_grid:
        try:
            x_labels.append(f"{float(val):.4g}")
        except (ValueError, TypeError):
            x_labels.append(str(val))

    y_labels = []
    for val in y_grid:
        try:
            y_labels.append(f"{float(val):.4g}")
        except (ValueError, TypeError):
            y_labels.append(str(val))

    plt.figure(figsize=(10, 7))
    sns.heatmap(error_matrix, annot=True, fmt=".6f", xticklabels=x_labels, yticklabels=y_labels, cmap='jet')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()