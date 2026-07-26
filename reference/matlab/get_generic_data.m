function [pos, spk, selected_epoch] = get_generic_data(degu, session_str, tet, cl)
% GET_GENERIC_DATA Carga posición y spikes para la época más larga de un registro.
% Equivalente generalizado de get_OF_data. Funciona para OF, LT u otros.
% Inputs:
%   degu        - String con el número de animal, ej. 'IX'
%   session_str - String con el tipo y número de sesión, ej. 'OF-V'
%   tet         - Entero con el número de tetrodo
%   cl          - Entero con el número de clúster (neurona)

    % Usar la ruta original del servidor
    data_dir = '/mnt/NAS/Degus/merged_files/';
    
    merge_file = fullfile(data_dir, sprintf('S2020_Mark%s-%s_merged.db', degu, session_str));
    sorting_file = fullfile(data_dir, sprintf('S2020_Mark%s-%s.db_clnew', degu, session_str));
    
    if ~exist(merge_file, 'file')
        error('File not found: %s', merge_file);
    end
    if ~exist(sorting_file, 'file')
        error('File not found: %s', sorting_file);
    end
    
    % Usar matfile para leer sin cargar todo en memoria
    m = matfile(merge_file);
    vars = who(m);
    
    best_epoch = -1;
    max_len = 0;
    
    % Encontrar la época (pos_X) con la trayectoria más larga
    for i = 1:5  % Típicamente pos_1 a pos_3
        pos_name = sprintf('pos_%d', i);
        if ismember(pos_name, vars)
            p = m.(pos_name);
            len = length(p.x);
            if len > max_len
                max_len = len;
                best_epoch = i;
            end
        end
    end
    
    if best_epoch == -1
        error('No valid position data found in %s', merge_file);
    end
    
    % Cargar la posición seleccionada
    pos_name = sprintf('pos_%d', best_epoch);
    pos = m.(pos_name);
    
    % Cargar los spikes brutos de la época y tetrodo
    spk_name = sprintf('spk_ts_%d_%d', best_epoch, tet);
    if ismember(spk_name, vars)
        spk_all = m.(spk_name);
    else
        error('Spikes %s not found in %s', spk_name, merge_file);
    end
    
    % Cargar los clusters
    s_data = load(sorting_file, 'all_clust', '-mat');
    
    % all_clust es un cell array (n_epochs x n_tetrodes)
    % Cada celda contiene clusters. Extraccion: all_clust{epoch, tet}{cluster}
    try
        cluster_indices = s_data.all_clust{best_epoch, tet}{cl};
        spk = spk_all(cluster_indices);
    catch err
        error('Error extrayendo spikes del cluster %d para tetrodo %d, epoca %d: %s', cl, tet, best_epoch, err.message);
    end
    
    selected_epoch = best_epoch;
end
