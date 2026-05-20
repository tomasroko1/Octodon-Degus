import numpy as np

def calcular_angulo_preferido_coseno(angulos_reales, spikes, tiempos_bin, tolerancia_deg=30):
    """
    Calcula el ángulo preferido y amplitud.
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
            delta_R.append(0.0)
            
    delta_R = np.array(delta_R)
    
    rad_cand = np.radians(angulos_candidatos)
    projection = 2.0 * np.mean(delta_R * np.exp(1j * rad_cand))
    
    amplitud_optima = np.abs(projection)
    angulo_preferido_deg = np.degrees(np.angle(projection)) % 360
    baseline = np.mean(delta_R)
    
    funcion = lambda x: amplitud_optima * np.cos(np.radians(x) - np.radians(angulo_preferido_deg)) + baseline
    
    return angulos_candidatos, delta_R, amplitud_optima, angulo_preferido_deg, baseline, funcion   


def evaluar_confianza_angulo_preferido(angulos_reales, spikes, tiempos_bin, tolerancia_deg=30, 
                                       n_shifts=100, min_shift_sec=10.0):
    """
    Evalúa la confianza estadística de la sintonización de Head Direction mediante spike shifting
    utilizando el método de proyección lineal compleja.
    """
    ang_cand, delta_R, original_amp, original_angle, baseline, _ = calcular_angulo_preferido_coseno(
        angulos_reales, spikes, tiempos_bin, tolerancia_deg
    )
    
    angulos_candidatos = np.arange(0, 360, 10)
    rad_cand = np.radians(angulos_candidatos)
    
    angle_masks = []
    for theta in angulos_candidatos:
        diff_angular = np.abs((angulos_reales - theta + 180) % 360 - 180)
        mira_hacia = diff_angular <= tolerancia_deg
        sum_mira = np.sum(mira_hacia)
        sum_no_mira = np.sum(~mira_hacia)
        angle_masks.append((mira_hacia, ~mira_hacia, sum_mira, sum_no_mira))
        
    min_shift_bins = int(np.ceil(min_shift_sec / tiempos_bin))
    L = len(spikes)
    if L < 2 * min_shift_bins:
        raise ValueError("Sesión muy corta para el shift mínimo.")
        
    shuffled_amplitudes = []
    
    rng = np.random.default_rng(42)
    valid_shifts = rng.integers(min_shift_bins, L - min_shift_bins, size=n_shifts)
    
    for s in valid_shifts:
        spikes_shuffled = np.roll(spikes, s)
        delta_R_shuff = []
        
        for mira_hacia, no_mira_hacia, sum_mira, sum_no_mira in angle_masks:
            if sum_mira > 0 and sum_no_mira > 0:
                rate_mira = np.sum(spikes_shuffled[mira_hacia]) / (sum_mira * tiempos_bin)
                rate_no_mira = np.sum(spikes_shuffled[no_mira_hacia]) / (sum_no_mira * tiempos_bin)
                delta_R_shuff.append(rate_mira - rate_no_mira)
            else:
                delta_R_shuff.append(0.0)
                
        delta_R_shuff = np.array(delta_R_shuff)
        
        # Proyección lineal compleja
        shuff_proj = 2.0 * np.mean(delta_R_shuff * np.exp(1j * rad_cand))
        shuffled_amplitudes.append(np.abs(shuff_proj))
        
    shuffled_amplitudes = np.array(shuffled_amplitudes)
    
    p_value = np.sum(shuffled_amplitudes >= original_amp) / n_shifts
    mean_shuff = np.mean(shuffled_amplitudes)
    std_shuff = np.std(shuffled_amplitudes) if np.std(shuffled_amplitudes) > 0 else 1e-6
    z_score = (original_amp - mean_shuff) / std_shuff
    confianza_significativa = (p_value < 0.05) or (z_score > 1.96)
    
    return {
        'original_amplitude': original_amp,
        'original_angle': original_angle,
        'mean_shuffled_amplitude': mean_shuff,
        'std_shuffled_amplitude': std_shuff,
        'shuffled_amplitudes': shuffled_amplitudes,
        'p_value': p_value,
        'z_score': z_score,
        'confianza_significativa': confianza_significativa
    }


def analizar_sintonizacion_perspectiva_continua(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin,
                                               grid_res=20, sigma_hd_deg=17.2, grid_extent=None):
    """
    Calcula la sintonización de perspectiva circular continua.
    Identifica el punto óptimo de referencia en una grilla 2D del entorno.
    """
    if grid_extent is None:
        x_min, x_max = np.min(x_bins) - 10, np.max(x_bins) + 10
        y_min, y_max = np.min(y_bins) - 10, np.max(y_bins) + 10
    else:
        x_min, x_max, y_min, y_max = grid_extent
        
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    XX, YY = np.meshgrid(x_grid, y_grid)
    X_flat = XX.flatten()
    Y_flat = YY.flatten()
    M_points = len(X_flat)
    
    dy = Y_flat[np.newaxis, :] - y_bins[:, np.newaxis]
    dx = X_flat[np.newaxis, :] - x_bins[:, np.newaxis]
    ang_hacia_punto = np.arctan2(dy, dx)
    hd_corrected = np.mod(ang_bins_rad[:, np.newaxis] - ang_hacia_punto, 2 * np.pi)
    
    N_hdbins = 36
    hdbins = np.linspace(0, 2 * np.pi, N_hdbins, endpoint=False)
    sigma_hd_rad = np.radians(sigma_hd_deg)
    sigma_sq_hd = sigma_hd_rad ** 2
    fs = 1.0 / tiempos_bin
    
    hd_corr_persp = np.zeros((M_points, N_hdbins))
    
    for w in range(M_points):
        phi_w = hd_corrected[:, w]
        # diferencia angular continua usando módulo 2*pi (más rápido que sin/cos/arctan2)
        diff = (hdbins[np.newaxis, :] - phi_w[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi
        weights = np.exp(- (diff ** 2) / (2 * sigma_sq_hd))
        
        num = np.mean(spikes[:, np.newaxis] * weights, axis=0)
        den = np.mean(weights, axis=0)
        
        valido = den > 1e-12
        tuning_curve = np.zeros(N_hdbins)
        tuning_curve[valido] = num[valido] / den[valido]
        
        hd_corr_persp[w, :] = fs * tuning_curve
        
    cosine_projection = 2 * np.mean(hd_corr_persp * np.exp(1j * hdbins)[np.newaxis, :], axis=1)
    map_p = np.abs(cosine_projection)
    map_p_matrix = map_p.reshape(grid_res, grid_res)
    
    indx = np.argmax(map_p)
    persp_indx = map_p[indx]
    best_coord = (X_flat[indx], Y_flat[indx])
    theta0 = np.angle(cosine_projection[indx]) % (2 * np.pi)
    
    hdbins_ext = np.concatenate([hdbins - 2*np.pi, hdbins, hdbins + 2*np.pi])
    tuning_ext = np.concatenate([hd_corr_persp[indx, :], hd_corr_persp[indx, :], hd_corr_persp[indx, :]])
    
    rate_J = np.interp(hd_corrected[:, indx], hdbins_ext, tuning_ext)
    r_J = rate_J / fs
    
    rate_mean = np.mean(spikes) * fs
    rate_Jmodel = rate_mean + persp_indx * np.cos(hd_corrected[:, indx] - theta0)
    rate_Jmodel = np.clip(rate_Jmodel, 0, None)
    r_Jmodel = rate_Jmodel / fs
    
    return {
        'XX': XX,
        'YY': YY,
        'map_p_matrix': map_p_matrix,
        'best_coord': best_coord,
        'persp_indx': persp_indx,
        'theta0': theta0,
        'hd_corrected_best': hd_corrected[:, indx],
        'r_J': r_J,
        'r_Jmodel': r_Jmodel,
        'hdbins': hdbins,
        'tuning_curve_best': hd_corr_persp[indx, :]
    }


def evaluar_confianza_sintonizacion_perspectiva(x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin,
                                                grid_res=20, sigma_hd_deg=17.2, grid_extent=None,
                                                n_shifts=100, min_shift_sec=10.0):
    """
    Evalúa la confianza estadística del punto de perspectiva preferido (persp_indx)
    mediante un test de control basado en el desplazamiento circular de spikes ('spike shifting').
    """
    original_res = analizar_sintonizacion_perspectiva_continua(
        x_bins, y_bins, ang_bins_rad, spikes, tiempos_bin, grid_res, sigma_hd_deg, grid_extent
    )
    original_val = original_res['persp_indx']
    
    min_shift_bins = int(np.ceil(min_shift_sec / tiempos_bin))
    L = len(spikes)
    if L < 2 * min_shift_bins:
        raise ValueError("Sesión muy corta para el shift mínimo.")
        
    if grid_extent is None:
        x_min, x_max = np.min(x_bins) - 10, np.max(x_bins) + 10
        y_min, y_max = np.min(y_bins) - 10, np.max(y_bins) + 10
    else:
        x_min, x_max, y_min, y_max = grid_extent
        
    x_grid = np.linspace(x_min, x_max, grid_res)
    y_grid = np.linspace(y_min, y_max, grid_res)
    XX, YY = np.meshgrid(x_grid, y_grid)
    X_flat = XX.flatten()
    Y_flat = YY.flatten()
    M_points = len(X_flat)
    
    dy = Y_flat[np.newaxis, :] - y_bins[:, np.newaxis]
    dx = X_flat[np.newaxis, :] - x_bins[:, np.newaxis]
    ang_hacia_punto = np.arctan2(dy, dx)
    hd_corrected = np.mod(ang_bins_rad[:, np.newaxis] - ang_hacia_punto, 2 * np.pi)
    
    N_hdbins = 36
    hdbins = np.linspace(0, 2 * np.pi, N_hdbins, endpoint=False)
    sigma_hd_rad = np.radians(sigma_hd_deg)
    sigma_sq_hd = sigma_hd_rad ** 2
    fs = 1.0 / tiempos_bin
    
    W_matrix = np.zeros((L, M_points), dtype=complex)
    
    for w in range(M_points):
        phi_w = hd_corrected[:, w]
        diff = (hdbins[np.newaxis, :] - phi_w[:, np.newaxis] + np.pi) % (2 * np.pi) - np.pi
        weights = np.exp(- (diff ** 2) / (2 * sigma_sq_hd))
        
        den = np.mean(weights, axis=0)
        valido = den > 1e-12
        
        term = np.zeros(N_hdbins, dtype=complex)
        term[valido] = np.exp(1j * hdbins[valido]) / den[valido]
        W_matrix[:, w] = (fs * 2.0 / (L * N_hdbins)) * np.sum(weights * term[np.newaxis, :], axis=1)
        
    rng = np.random.default_rng(42)
    valid_shifts = rng.integers(min_shift_bins, L - min_shift_bins, size=n_shifts)
    
    # multiplicación matricial directa (C-Level en NumPy)
    shuffled_vals = []
    for s in valid_shifts:
        spikes_shuffled = np.roll(spikes, s)
        projections = spikes_shuffled @ W_matrix
        shuffled_vals.append(np.max(np.abs(projections)))
        
    shuffled_vals = np.array(shuffled_vals)
    
    p_value = np.sum(shuffled_vals >= original_val) / n_shifts
    mean_shuff = np.mean(shuffled_vals)
    std_shuff = np.std(shuffled_vals) if np.std(shuffled_vals) > 0 else 1e-6
    z_score = (original_val - mean_shuff) / std_shuff
    confianza_significativa = (p_value < 0.05) or (z_score > 1.96)
    
    return {
        'original_coord': original_res['best_coord'],
        'original_persp_indx': original_val,
        'mean_shuffled_persp_indx': mean_shuff,
        'std_shuffled_persp_indx': std_shuff,
        'shuffled_persp_indxs': shuffled_vals,
        'p_value': p_value,
        'z_score': z_score,
        'confianza_significativa': confianza_significativa
    }