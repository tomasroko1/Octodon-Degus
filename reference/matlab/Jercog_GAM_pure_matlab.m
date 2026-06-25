fpath = '/mnt/NAS/Degus/merged_files/';
load('/mnt/NAS/Mati/MATLAB/2019-20 _ Degus/Degus-2020-Mati/AllData2.db',
     '-mat');
% abre el archivo csv close all;

data = data(contains(data( :, 2), 'OF'), :);
Nshuff = 100;
N = size(data, 1);

X0 = NaN(N, Nshuff + 1);
Y0 = NaN(N, Nshuff + 1);
persp_indx = NaN(N, Nshuff + 1);
sp_info = NaN(N, Nshuff + 1);
hd_info = NaN(N, Nshuff + 1);
mvl = NaN(N, Nshuff + 1);

% mdl = cell(1, numel(myjs));
for
  j = 44 % 1 : N disp(j) if j == 47 continue;
end

    tokens = regexp(data{j, 2}, 'OF-([IVXLCDM]+)', 'tokens');
degu = data{j, 1};
sess = tokens{1} {1};

% %
    Signal

        [pos, spk, sess, of_sel] =
    get_OF_data(degu, sess, str2double(data{j, 4}), data{j, 5}); % only last session, all but 3 cells (possibly the same and for some error) have only one session

    x=pos.x-min(pos.x);
y = pos.y - min(pos.y);
sel = isnan(x) | isnan(y);
x(sel) = interp1(pos.t(~sel), x(~sel), pos.t(sel), 'linear', 'extrap');
y(sel) = interp1(pos.t(~sel), y(~sel), pos.t(sel), 'linear', 'extrap');
hd = pos.hd;
sel = isnan(hd);
hd(sel) = angle(interp1(pos.t(~sel), pos.hd(~sel), pos.t(sel), 'linear',
                        'extrap'));

spikes = hist(spk, pos.t)';

    bins = -10 : 3 : 100;
[ xb, yb ] = meshgrid(bins);
sigma_sq = 5 ^ 2;

[ r_spatial, mapr ] = predict_spatial(spikes, x, y, xb, yb, sigma_sq, x, y);

% % Downsmpling ndwn = 50 * .1;
x = downsample(x, ndwn);
y = downsample(y, ndwn);
spikes = round(downsample(spikes, ndwn) * ndwn);
    spikes=mov_gauss(spikes',0)';
    %%

% 1. Clean Data
valid = ~isnan(x) & ~isnan(y);
xv = x(valid);
yv = y(valid);
sv = spikes(valid);

% 2. Define Model Complexity (Knots)
knots_x = 5;
knots_y = 5; % Square arena (90x90)

% 3. Generate 1D Spline Bases for EVERY time point
% This replaces the 'binning' step
Bx = bspline_basis_cubic(xv, knots_x, [0, 90]);
By = bspline_basis_cubic(yv, knots_y, [0, 90]);

% 4. Create 2D Tensor Product Basis Matrix (The "Design Matrix")
% Matrix size: [NumTimePoints x (knots_x * knots_y)]
n_basis = knots_x * knots_y;
M = zeros(length(xv), n_basis);

% Efficient tensor product calculation
for i = 1:knots_x
    for j = 1:knots_y
        col_idx = (i-1)*knots_y + j;
        M(:, col_idx) = Bx(:, i) .* By(:, j);
    end
end

% 5. Fit the GLM (Poisson with Log Link)
% No 'Offset' is needed here because every time-bin is exactly 1 unit of time
mdl = fitglm(M, sv, 'Distribution', 'poisson', 'Link', 'log');

% 6. Generate the Prediction (The Mean Spike Rate per Time-Bin)
% This is your "mean in time" (mu)
predicted_mean_spikes = predict(mdl, M);

% 7. (Optional) Generate a spatial map for visualization
[gx, gy] = meshgrid(0:1:90, 0:1:90);
Bx_grid = bspline_basis_cubic(gx(:), knots_x, [0, 90]);
By_grid = bspline_basis_cubic(gy(:), knots_y, [0, 90]);
M_grid = zeros(numel(gx), n_basis);
for i = 1:knots_x
    for j = 1:knots_y
        col_idx = (i-1)*knots_y + j;
        M_grid(:, col_idx) = Bx_grid(:, i) .* By_grid(:, j);
    end
end

% Predict rate (Hz). Since we fit to raw time bins, we multiply by sampling rate.
fs = 50; % 25 Hz sampling
spatial_rate_map = reshape(predict(mdl, M_grid) * fs, size(gx));


plot([predicted_mean_spikes spikes]);

title(trunc(corr(spikes,predicted_mean_spikes,'rows','pairwise'),2))

ll_spike = calculate_ll_spike(spikes, predicted_mean_spikes)
    
end

% %% Helper Function
% function B = bspline_basis(pos, n_basis, range)
%     knots = linspace(range(1), range(2), n_basis);
%     dx = knots(2) - knots(1);
%     B = zeros(length(pos), n_basis);
%     for i = 1:n_basis
%         % Triangular B-Spline (Linear)
%         B(:,i) = max(0, 1 - abs(pos - knots(i))/dx);
%     end
%     B = bsxfun(@rdivide, B, sum(B, 2)); % Normalize rows
% end