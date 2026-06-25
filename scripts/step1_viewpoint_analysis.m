% step1_viewpoint_analysis.m
% =========================================================================
% Step 1: Viewpoint Analysis & Significance Shuffling
% Itera sobre todas las células de Open Field (OF)
% y para cada una corre el análisis de sintonización de perspectiva
% (viewpoint tuning) con test de significancia por spike shifting.
% =========================================================================

fpath = '/mnt/NAS/Degus/merged_files/';
load('/mnt/NAS/Mati/MATLAB/2019-20 _ Degus/Degus-2020-Mati/AllData2.db', '-mat');
close all;

% Filtrar solo sesiones de Open Field
data = data(contains(data(:,2), 'OF'), :);
N = size(data, 1);

% --- Parámetros del análisis ---
Nshuff       = 100;       % Número de shufflings circulares
bin_size_sec = 0.1;       % Tamaño del bin temporal (100 ms)
grid_res     = 25;        % Resolución de la grilla de perspectiva
N_hdbins     = 36;        % Número de bins de head direction (cada 10°)
sigma_hd_deg = 17.2;      % Ancho del kernel gaussiano circular (grados)
min_shift_sec = 10.0;     % Shift mínimo en segundos
min_spikes   = 10;        % Umbral mínimo de spikes para procesar

% ----- Directorio de guardado -----
save_dir = '../results/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
save_path = fullfile(save_dir, 'viewpoint_results.mat');

%% =========================================================================
best_x         = NaN(N, 1);
best_y         = NaN(N, 1);
persp_indx_all = NaN(N, 1);
theta0_all     = NaN(N, 1);
z_score_all    = NaN(N, 1);
p_value_all    = NaN(N, 1);
stability_dist_all = NaN(N, 1);
is_sig         = false(N, 1);
num_spikes_all = zeros(N, 1);
status         = cell(N, 1);
cell_ids       = cell(N, 1);

fprintf('==========================================================\n');
fprintf(' Iniciando Step 1: %d células de Open Field\n', N);
fprintf(' Guardando en: %s\n', save_path);
fprintf('==========================================================\n\n');

%% ====================== LOOP PRINCIPAL ==================================
tic;
for j = 1:N
    fprintf('[%3d/%d] ', j, N);
    
    try
        % ---- Extraer identificadores de la tabla ----
        tokens = regexp(data{j,2}, 'OF-([IVXLCDM]+)', 'tokens');
        degu = data{j,1};
        sess_roman = tokens{1}{1};
        tet = str2double(data{j,4});
        cl  = data{j,5};
        
        cell_ids{j} = sprintf('%s-OF-%s_T%s_N%s', degu, sess_roman, num2str(data{j,4}), num2str(cl));
        
        % ---- Cargar datos (iterador original de MATLAB) ----
        [pos, spk, ~, ~] = get_OF_data(degu, sess_roman, tet, cl);
        
        % ---- Preprocesamiento ----
        x  = pos.x(:) - min(pos.x);
        y  = pos.y(:) - min(pos.y);
        t  = pos.t(:);
        hd_raw = pos.hd(:);
        
        % Interpolar NaNs en posición
        sel = isnan(x) | isnan(y);
        if any(sel)
            x(sel) = interp1(t(~sel), x(~sel), t(sel), 'linear', 'extrap');
            y(sel) = interp1(t(~sel), y(~sel), t(sel), 'linear', 'extrap');
        end
        
        % Interpolar NaNs en head direction
        sel_hd = isnan(hd_raw);
        hd_uw = NaN(size(hd_raw));
        hd_uw(~sel_hd) = unwrap(hd_raw(~sel_hd));
        if any(sel_hd)
            hd_uw(sel_hd) = interp1(t(~sel_hd), hd_uw(~sel_hd), t(sel_hd), 'linear', 'extrap');
        end
        
        % Binning temporal
        bins_tiempo = (t(1) : bin_size_sec : t(end))';
        centros = bins_tiempo(1:end-1) + bin_size_sec / 2;
        
        conteo_spikes = histc(spk, bins_tiempo);
        conteo_spikes = conteo_spikes(1:end-1);
        conteo_spikes = conteo_spikes(:);
        
        x_b   = interp1(t, x, centros, 'linear', 'extrap');
        y_b   = interp1(t, y, centros, 'linear', 'extrap');
        ang_b = interp1(t, hd_uw, centros, 'linear', 'extrap');
        
        x_b = x_b(:); y_b = y_b(:); ang_b = ang_b(:);
        
        total_spk = sum(conteo_spikes);
        num_spikes_all(j) = total_spk;
        
        if total_spk < min_spikes
            fprintf('%s - Saltada: pocos spikes (%d)\n', cell_ids{j}, total_spk);
            status{j} = 'skipped_low_spikes';
            continue;
        end
        
        % ---- 1. Análisis de Perspectiva ----
        x_min = min(x_b) - 30;  x_max = max(x_b) + 30;
        y_min = min(y_b) - 30;  y_max = max(y_b) + 30;
        x_grid = linspace(x_min, x_max, grid_res);
        y_grid = linspace(y_min, y_max, grid_res);

        [p_indx, bx, by, th0] = perspectiva_continua(...
            x_b, y_b, ang_b, conteo_spikes, ...
            bin_size_sec, x_grid, y_grid, N_hdbins, sigma_hd_deg);
        
        best_x(j)         = bx;
        best_y(j)         = by;
        persp_indx_all(j) = p_indx;
        theta0_all(j)     = th0;
        
        % Test de Significancia (Shuffling)
        [z, pval, sig] = shuffling_perspectiva(...
            x_b, y_b, ang_b, conteo_spikes, bin_size_sec, ...
            x_grid, y_grid, N_hdbins, sigma_hd_deg, ...
            Nshuff, min_shift_sec, p_indx);
        
        z_score_all(j) = z;
        p_value_all(j) = pval;
        is_sig(j)      = sig;
        status{j}      = 'success';
        
        if sig; sig_str = '*** SIGNIFICATIVA ***'; else; sig_str = 'no sig.'; end
        fprintf('%s | Coord:(%.1f, %.1f) | Z:%.2f | P:%.3f | %s\n', ...
            cell_ids{j}, bx, by, z, pval, sig_str);
        
    catch err
        fprintf('ERROR: %s\n', err.message);
        status{j} = 'error';
    end
    
    save(save_path, 'best_x', 'best_y', 'persp_indx_all', 'theta0_all', ...
        'z_score_all', 'p_value_all', 'is_sig', 'num_spikes_all', ...
        'status', 'cell_ids', 'data', 'j');
end

elapsed = toc;
n_success = sum(strcmp(status, 'success'));
n_sig     = sum(is_sig);
fprintf('\n==========================================================\n');
fprintf(' Step 1 completado en %.1f minutos\n', elapsed/60);
fprintf(' Guardado en: %s\n', save_path);
fprintf(' Significativas: %d\n', n_sig);
fprintf('==========================================================\n');

%% --- FUNCIONES LOCALES ---
function [persp_indx, best_x, best_y, theta0] = perspectiva_continua(x_bins, y_bins, ang_bins, spikes, bin_size_sec, x_grid, y_grid, N_hdbins, sigma_hd_deg)
    [XX, YY] = meshgrid(x_grid, y_grid); X_flat = XX(:); Y_flat = YY(:); M = numel(X_flat);
    dy = Y_flat' - y_bins; dx = X_flat' - x_bins;
    ang_hacia_punto = atan2(dy, dx);
    hd_corrected = mod(ang_bins - ang_hacia_punto, 2*pi);
    hdbins = linspace(0, 2*pi, N_hdbins + 1); hdbins = hdbins(1:end-1);
    sigma_sq = deg2rad(sigma_hd_deg)^2; fs = 1.0 / bin_size_sec;
    hd_corr_persp = zeros(M, N_hdbins);
    for w = 1:M
        diff_ang = mod(hdbins - hd_corrected(:, w) + pi, 2*pi) - pi;
        weights = exp(-(diff_ang.^2) / (2 * sigma_sq));
        num = mean(spikes .* weights, 1); den = mean(weights, 1);
        valido = den > 1e-12; tuning = zeros(1, N_hdbins); tuning(valido) = num(valido) ./ den(valido);
        hd_corr_persp(w, :) = fs * tuning;
    end
    cosine_proj = 2 * mean(hd_corr_persp .* exp(1i * hdbins), 2);
    [persp_indx, indx] = max(abs(cosine_proj)); best_x = X_flat(indx); best_y = Y_flat(indx); theta0 = mod(angle(cosine_proj(indx)), 2*pi);
end

function [z_score, p_value, is_significant] = shuffling_perspectiva(x_bins, y_bins, ang_bins, spikes, bin_size_sec, x_grid, y_grid, N_hdbins, sigma_hd_deg, Nshuff, min_shift_sec, original_val)
    L = numel(spikes); min_shift_bins = ceil(min_shift_sec / bin_size_sec);
    if L < 2 * min_shift_bins; z_score = NaN; p_value = NaN; is_significant = false; return; end
    [XX, YY] = meshgrid(x_grid, y_grid); X_flat = XX(:); Y_flat = YY(:); M = numel(X_flat);
    dy = Y_flat' - y_bins; dx = X_flat' - x_bins; ang_hacia_punto = atan2(dy, dx);
    hd_corrected = mod(ang_bins - ang_hacia_punto, 2*pi);
    hdbins = linspace(0, 2*pi, N_hdbins + 1); hdbins = hdbins(1:end-1);
    sigma_sq = deg2rad(sigma_hd_deg)^2; fs = 1.0 / bin_size_sec; W_matrix = complex(zeros(L, M));
    for w = 1:M
        diff_ang = mod(hdbins - hd_corrected(:, w) + pi, 2*pi) - pi;
        weights = exp(-(diff_ang.^2) / (2 * sigma_sq)); den = mean(weights, 1); valido = den > 1e-12;
        term = complex(zeros(1, N_hdbins)); term(valido) = exp(1i * hdbins(valido)) ./ den(valido);
        W_matrix(:, w) = (fs * 2.0 / (L * N_hdbins)) * sum(weights .* term, 2);
    end
    rng(42); valid_shifts = randi([min_shift_bins, L - min_shift_bins], 1, Nshuff);
    shuffled_vals = zeros(1, Nshuff);
    for s = 1:Nshuff
        shuffled_vals(s) = max(abs(circshift(spikes, valid_shifts(s))' * W_matrix));
    end
    p_value = sum(shuffled_vals >= original_val) / Nshuff; mean_shuff = mean(shuffled_vals); std_shuff  = std(shuffled_vals);
    if std_shuff < 1e-10; std_shuff = 1e-6; end; z_score = (original_val - mean_shuff) / std_shuff;
    is_significant = (p_value < 0.05) || (z_score > 1.96);
end
