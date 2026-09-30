%% JWCC v2 — maqola rasmlari va jadvallari (600 dpi PNG + TIFF + vektorli PDF)
% Talab: MATLAB R2020b yoki yangiroq (exportgraphics, tiledlayout). Toolbox kerak emas.
% Ishlatish: Drive'dagi jwcc_v2 papkasini kompyuterga yuklab oling, D ga yo'lini yozing, Run.
% Chiqish: jwcc_v2/figures/Fig1b ... Fig7 (.png, .tif, .pdf) va Tables.xlsx
clear; close all; clc;
D    = fullfile(fileparts(mfilename('fullpath')), '..', 'results');   % CSV fayllar papkasi
if ~isfile(fullfile(D, 'A1A2_responses.csv'))
    D = uigetdir(pwd, 'jwcc_v2 papkasini tanlang (CSV fayllar turgan joy)');
    if isequal(D, 0), error('Papka tanlanmadi'); end
end
OUTF = fullfile(D, 'figures'); if ~exist(OUTF, 'dir'), mkdir(OUTF); end
W1 = 8.3; W2 = 17.4;                                % bir va ikki ustunli kenglik, sm (HESS/WRR)

set(groot, 'defaultAxesFontName', 'Arial', 'defaultTextFontName', 'Arial', ...
    'defaultAxesFontSize', 8, 'defaultTextFontSize', 8, 'defaultLegendFontSize', 7, ...
    'defaultAxesTickLabelInterpreter', 'none', 'defaultLegendInterpreter', 'none', ...
    'defaultTextInterpreter', 'none', 'defaultAxesBox', 'on', 'defaultAxesLineWidth', 0.6, ...
    'defaultAxesTickDir', 'out', 'defaultAxesTickLength', [0.015 0.015]);
C.xgb = [0.12 0.47 0.71]; C.rf = [0.20 0.63 0.17]; C.ridge = [1.00 0.50 0.05];
C.lstm = [0.58 0.40 0.74]; C.hbv = [0 0 0]; C.dd = [0.84 0.15 0.16]; C.grey = [0.62 0.62 0.62];
C.inc = [0.99 0.73 0.30]; C.dec = [0.55 0.75 0.90];
rd = @(f) readtable(fullfile(D, f), 'VariableNamingRule', 'preserve', 'TextType', 'string');

R  = rd('A1A2_responses.csv');   AR = rd('A1R_summary.csv');
B  = rd('B2_comparison.csv');    HS = rd('B2_hbv_skill.csv');   HR = rd('B2_hbv_responses.csv');
A3 = rd('A3_summary.csv');       V  = rd('A4_virtual_responses.csv');
hbvMed = @(scn) median(B.seas_HBV_glac(B.scenario == scn), 'omitnan');
ddMed  = median(R.seas(R.pert_mode == "consistent" & R.scenario == "T+2" & R.source == "DD_cal"), 'omitnan');

%% Fig. 1b — tajriba sxemasi (1a: eski maqoladagi xarita)
fig = newfig(W2, 7.5);
bx = @(p, s, c) annotation(fig, 'textbox', p, 'String', s, 'FontSize', 7.5, 'EdgeColor', c, ...
    'LineWidth', 0.9, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'BackgroundColor', 'w', 'FitBoxToText', 'off', 'Interpreter', 'none');
ar = @(x, y) annotation(fig, 'arrow', x, y, 'LineWidth', 0.8, 'HeadLength', 6, 'HeadWidth', 6);
bx([0.01 0.39 0.15 0.22], sprintf('ERA5-Land forcing\n9 basins, 1959-1990\n(dekadal)'), [0.3 0.3 0.3]);
bx([0.21 0.39 0.19 0.22], sprintf('Perturbation schemes\npaper | consistent |\nempirical (+9 snow\nvariants)'), C.inc*0.8);
bx([0.46 0.68 0.22 0.28], sprintf('ML models\nXGB | RF | Ridge | LSTM\nx  OS | AR | NQ modes'), C.xgb);
bx([0.46 0.04 0.22 0.28], sprintf('Process benchmarks\nDD uncal | DD cal |\nHBV +/- glacier\n(5 seeds)'), C.dd);
bx([0.73 0.04 0.16 0.28], sprintf('Virtual laboratory\nHBV truth\n+ 10%% Q noise\nP noise 0 | 0.5'), C.lstm);
bx([0.82 0.44 0.17 0.24], sprintf('Responses\nseasonal shift dS\nvolume dV\nratio, sign'), [0.3 0.3 0.3]);
ar([0.16 0.21], [0.50 0.50]);
ar([0.40 0.46], [0.56 0.80]); ar([0.40 0.46], [0.44 0.20]);
ar([0.68 0.82], [0.82 0.62]);
ar([0.68 0.73], [0.14 0.14]);
ar([0.62 0.84], [0.32 0.44]);
ar([0.85 0.88], [0.32 0.44]);
annotation(fig, 'textbox', [0.58 0.38 0.10 0.05], 'String', 'reference', 'EdgeColor', 'none', ...
    'FontSize', 7, 'FontAngle', 'italic');
saveall(fig, 'Fig1b_workflow', OUTF);

%% Fig. 2 — dizayn dekompozitsiyasi (sharshara)
st = ["paper" "OS_ens"; "paper" "OS_xgb"; "consistent" "OS_xgb"; "consistent" "AR_xgb"];
lab = {'Original design', 'No climatology member', 'Consistent snow inputs', 'Autoregressive simulation'};
fig = newfig(W1, 7.2); ax = axes(fig); hold(ax, 'on'); rng(1);
% Medianlar: to'g'ri (faza tuzatilgan) Colab natijasidan, T+2, 9 basseyn
m = [1.58 3.04 8.30 19.98];
for k = 1:4
    v = R.seas(R.pert_mode == st(k, 1) & R.scenario == "T+2" & R.source == st(k, 2));
    v = v(isfinite(v));                 % 1-bosqich CSV da bo'sh -> nuqtalarsiz chiziladi
    lo = 0; if k > 1 && isfinite(m(k - 1)), lo = m(k - 1); end
    col = C.grey; if k > 1, col = C.inc; if m(k) < lo, col = C.dec; end, end
    patch(ax, k + [-0.32 0.32 0.32 -0.32], [lo lo m(k) m(k)], col, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.5);
    if k < 4, plot(ax, [k + 0.32 k + 0.68], [m(k) m(k)], ':', 'Color', [0.3 0.3 0.3]); end
    scatter(ax, k + 0.18 * (rand(size(v)) - 0.5), v, 9, [0.25 0.25 0.25], 'filled', 'MarkerFaceAlpha', 0.55);
    text(ax, k + 0.36, m(k), sprintf('%.1f', m(k)), 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'middle', 'FontWeight', 'bold', 'BackgroundColor', 'w', 'Margin', 0.5);
end
yline(ax, hbvMed("T+2"), '-', sprintf('HBV glacier %.1f', hbvMed("T+2")), 'Color', C.hbv, 'LineWidth', 1, ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
yline(ax, ddMed, '--', sprintf('DD calibrated %.1f', ddMed), 'Color', C.dd, 'LineWidth', 0.8, ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
yline(ax, 0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.4);
set(ax, 'XTick', 1:4, 'XTickLabel', lab, 'XTickLabelRotation', 25, 'XLim', [0.4 4.9], 'YLim', [-10 60]);

ylabel(ax, 'Seasonal shift under +2 K, dS (%)');
saveall(fig, 'Fig2_design_decomposition', OUTF);

%% Fig. 3 — perturbatsiya sxemasiga sezgirlik
src = ["OS_ens" "OS_xgb" "AR_ens" "AR_xgb" "NQ_xgb"];
fig = newfig(W1, 6.8); ax = axes(fig); hold(ax, 'on');
for k = 1:numel(src)
    g = AR(AR.scenario == "T+2" & AR.source == src(k), :);
    d = g.seas_med(g.variant ~= "empirical"); e = g.seas_med(g.variant == "empirical");
    plot(ax, [k k], [min(d) max(d)], '-', 'Color', C.grey, 'LineWidth', 3);
    h1 = scatter(ax, k * ones(size(d)), d, 14, C.grey, 'filled', 'MarkerEdgeColor', [0.3 0.3 0.3]);
    h2 = scatter(ax, k, e, 30, C.dd, 'd', 'filled', 'MarkerEdgeColor', 'k');
end
h3 = yline(ax, hbvMed("T+2"), '-', 'Color', C.hbv, 'LineWidth', 1);
h4 = yline(ax, ddMed, '--', 'Color', C.dd, 'LineWidth', 0.8);
set(ax, 'XTick', 1:numel(src), 'XTickLabel', src, 'XLim', [0.4 numel(src) + 0.6]);
ylabel(ax, 'Seasonal shift under +2 K (%)');
ylim(ax, [0 55]);
lg = legend(ax, [h1 h2 h3 h4], {'Degree-day variants (9)', 'Empirical', 'HBV glacier', 'DD calibrated'}, ...
    'NumColumns', 1, 'Box', 'off', 'FontSize', 6.5);
placeLegend(lg, ax, 0.04, 0.46);
saveall(fig, 'Fig3_perturbation_robustness', OUTF);

%% Fig. 4 — basseynlar bo'yicha benchmarklar (muzlik ulushi tartibida)
T2 = sortrows(B(B.scenario == "T+2", :), 'glacier_%');
cols = ["seas_AR_xgb" "seas_DD_cal" "seas_HBV_noglac" "seas_HBV_glac"];
cc = [C.xgb; C.dd; [0.65 0.65 0.65]; C.hbv];
fig = newfig(W2, 7); ax = axes(fig); hold(ax, 'on');
b = bar(ax, T2{:, cols}, 'grouped', 'BarWidth', 0.85, 'EdgeColor', 'none');
for k = 1:4, b(k).FaceColor = cc(k, :); end
for k = 3:4
    vn = ["HBV_noglac" "HBV_glac"]; x = b(k).XEndPoints;
    for i = 1:height(T2)
        s = HR.seas(HR.gauge == T2.gauge(i) & HR.variant == vn(k - 2) & HR.scenario == "T+2");
        plot(ax, [x(i) x(i)], [min(s) max(s)], 'k-', 'LineWidth', 0.7);
    end
end
yline(ax, 0, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.4);
xl = compose("%s (%.1f%%)", T2.basin, T2.("glacier_%"));
set(ax, 'XTick', 1:height(T2), 'XTickLabel', xl, 'XTickLabelRotation', 30);
ylabel(ax, 'Seasonal shift under +2 K (%)');
ylim(ax, [-10 95]);
lg = legend(ax, b, {'XGB autoregressive', 'DD calibrated', 'HBV without glacier', 'HBV with glacier'}, ...
    'NumColumns', 4, 'Box', 'off', 'FontSize', 6.5);
placeLegend(lg, ax, 0.01, 0.88);   % mo'ylovlar: 5 kalibratsiya urug'i oralig'i (rasm izohida yoziladi)
saveall(fig, 'Fig4_benchmarks_by_basin', OUTF);

%% Fig. 5 — muzlik ulushi va javob
fig = newfig(W1, 7.5); ax = axes(fig); hold(ax, 'on');
ser = {"seas_DD_cal", 'DD calibrated', C.dd, 'o'; "seas_HBV_noglac", 'HBV without glacier', [0.55 0.55 0.55], 's';
       "seas_HBV_glac", 'HBV with glacier', C.hbv, '^'; "seas_AR_xgb", 'XGB autoregressive', C.xgb, 'd'};
g = T2.("glacier_%"); h = gobjects(1, 4); lg = cell(1, 4);
for k = 1:4
    y = T2.(ser{k, 1});
    [r, p] = spear(g, y);
    if k == 1 || k == 3, pf = polyfit(g, y, 1); plot(ax, [0 6], polyval(pf, [0 6]), ':', 'Color', ser{k, 3}); end
    h(k) = scatter(ax, g, y, 22, ser{k, 3}, ser{k, 4}, 'filled', 'MarkerEdgeColor', 'w');
    lg{k} = sprintf('%s, %s = %+.2f', ser{k, 2}, char(961), r);
    fprintf('Fig. 5  %-22s rho = %+.2f  p = %.3f\n', ser{k, 2}, r, p);
end
[~, iz] = max(g); text(ax, g(iz) - 0.12, T2.seas_DD_cal(iz), char(T2.basin(iz)), 'FontSize', 7, 'HorizontalAlignment', 'right');
xlabel(ax, 'Glacier fraction (%)'); ylabel(ax, 'Seasonal shift under +2 K (%)'); xlim(ax, [-0.3 6]);
ylim(ax, [-10 95]);
lg5 = legend(ax, h, lg, 'Box', 'off', 'FontSize', 6.5); placeLegend(lg5, ax, 0.02, 0.73);
saveall(fig, 'Fig5_glacier_dependence', OUTF);

%% Fig. 6 — arxitekturalar va rejimlar
arch = ["XGB" "RF" "Ridge" "LSTM"]; mode = ["OS" "AR" "NQ"];
ac = [C.xgb; C.rf; C.ridge; C.lstm];
getA3 = @(s, col) A3.(col)(A3.source == s);
fig = newfig(W2, 7.6); tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on'); hh = gobjects(0); ll = {};
for a = 1:4
    for mm = ["AR" "NQ"]
        s = arch(a) + "_" + mm; y = [getA3(s, "T+1") getA3(s, "T+2") getA3(s, "T+3")];
        ls = '-'; if mm == "NQ", ls = '--'; end
        hh(end + 1) = plot(ax, 1:3, y, ls, 'Color', ac(a, :), 'LineWidth', 1, 'Marker', 'o', 'MarkerSize', 3, ...
            'MarkerFaceColor', ac(a, :)); ll{end + 1} = char(s); %#ok<SAGROW>
    end
end
hh(end + 1) = plot(ax, 1:3, [hbvMed("T+1") hbvMed("T+2") hbvMed("T+3")], 'k-', 'LineWidth', 2, 'Marker', 's', ...
    'MarkerFaceColor', 'k', 'MarkerSize', 4); ll{end + 1} = 'HBV glacier';
set(ax, 'XTick', 1:3, 'XTickLabel', {'+1 K', '+2 K', '+3 K'}, 'XLim', [0.8 3.2]);
ylabel(ax, 'Seasonal shift, median (%)'); title(ax, 'a  Warming response', 'FontWeight', 'bold'); ax.TitleHorizontalAlignment = 'left';
lgA = legend(ax, hh, ll, 'Box', 'off', 'FontSize', 6.5, 'NumColumns', 5, 'Orientation', 'horizontal');
lgA.Layout.Tile = 'south'; ylim(ax, [0 35]);
pan = {"T2_seas_ratio", 'b  +2 K: seasonal shift / HBV'; "P10_vol_ratio", 'c  +10% P: volume / HBV'};
for q = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    M = zeros(4, 3); for a = 1:4, for mm = 1:3, M(a, mm) = getA3(arch(a) + "_" + mode(mm), pan{q, 1}); end, end
    bb = bar(ax, M, 'grouped', 'EdgeColor', 'none');
    bb(1).FaceColor = [0.80 0.80 0.80]; bb(2).FaceColor = [0.35 0.35 0.35]; bb(3).FaceColor = [0.62 0.62 0.62];
    yline(ax, 1, 'k-', 'LineWidth', 0.8);
    set(ax, 'XTick', 1:4, 'XTickLabel', arch); ylim(ax, [0 1.1]); ylabel(ax, 'Response ratio (median)');
    title(ax, pan{q, 2}, 'FontWeight', 'bold'); ax.TitleHorizontalAlignment = 'left';
    if q == 2, lgm = legend(ax, bb, strcat(cellstr(mode), ' mode'), 'Box', 'off'); placeLegend(lgm, ax, 0.04, 0.58); end
end
saveall(fig, 'Fig6_architectures', OUTF);

%% Fig. 7 — virtual laboratoriya
V.rs = V.seas ./ V.seas_true; V.rv = V.vol ./ V.vol_true;
src12 = strings(1, 12); i = 0;
for a = arch, for mm = mode, i = i + 1; src12(i) = a + "_" + mm; end, end
fig = newfig(W2, 7.2); tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
pan = {"T+2", "rs", "T2_seas_ratio", 'a  +2 K: seasonal shift / truth'; ...
       "P+10", "rv", "P10_vol_ratio", 'b  +10% P: volume / truth'};
for q = 1:2
    M = nan(12, 3);
    for k = 1:12
        s = src12(k); sel = V.scenario == pan{q, 1} & V.source == s;
        M(k, 1) = median(V.(pan{q, 2})(sel & V.sigma == 0), 'omitnan');
        M(k, 2) = median(V.(pan{q, 2})(sel & V.sigma == 0.5), 'omitnan');
        M(k, 3) = getA3(s, pan{q, 3});
    end
    ax = nexttile(tl); hold(ax, 'on');
    bb = bar(ax, M, 'grouped', 'EdgeColor', 'none', 'BarWidth', 0.9);
    bb(1).FaceColor = C.xgb; bb(2).FaceColor = [0.62 0.80 0.93]; bb(3).FaceColor = C.dd;
    yline(ax, 1, 'k-', 'LineWidth', 0.8); yline(ax, 0, 'Color', [0.5 0.5 0.5], 'LineWidth', 0.4);
    for a = 1:3, xline(ax, 3 * a + 0.5, ':', 'Color', [0.5 0.5 0.5]); end
    set(ax, 'XTick', 1:12, 'XTickLabel', src12, 'XTickLabelRotation', 60); ylim(ax, [-0.1 1.4]);
    ylabel(ax, 'Response ratio (median)'); title(ax, pan{q, 4}, 'FontWeight', 'bold'); ax.TitleHorizontalAlignment = 'left';
    if q == 1
        lg7 = legend(ax, bb, {'Virtual, clean P', 'Virtual, noisy P (sigma = 0.5)', 'Real data (vs HBV)'}, ...
            'Orientation', 'horizontal', 'Box', 'off');
        lg7.Layout.Tile = 'north';
    end
end
saveall(fig, 'Fig7_virtual_laboratory', OUTF);

%% Jadvallar -> Tables.xlsx
gid  = [17288 16279 16290 16300 16936 17202 17211 16176 16202]';
name = ["Zeravshan" "Chatkal" "Pskem" "Ugam" "Naryn" "Karatag" "Sangardak" "Padshaata" "Chadak"]';
area = [10297 5669 2518 863 52030 683 906 368 351]';
zmean = [3110 2693 2795 2046 2849 2666 2333 2797 2373]';
zmin = [1048 945 896 755 861 915 746 1524 1068]';  zmax = [5455 4441 4372 3600 5119 4744 4115 4314 3424]';
glac = [5.33 0.50 3.21 0.00 1.96 2.80 0.04 0.69 0.00]';
snow = [64.2 62.1 66.1 51.5 43.8 62.3 48.0 39.7 41.1]';
Tab1 = table(gid, name, area, zmean, zmin, zmax, glac, snow, 'VariableNames', ...
    {'Gauge', 'Basin', 'Area_km2', 'Elev_mean_m', 'Elev_min_m', 'Elev_max_m', 'Glacier_pct', 'Snowfall_share_pct'});
writetable(Tab1, fullfile(OUTF, 'Tables.xlsx'), 'Sheet', 'Table1');

Tab3 = table();
for k = 1:numel(gid)
    row = table(name(k), glac(k), 'VariableNames', {'Basin', 'Glacier_pct'});
    for vn = ["HBV_glac" "HBV_noglac"]
        s = HS(HS.gauge == gid(k) & HS.variant == vn, :);
        r = HR.seas(HR.gauge == gid(k) & HR.variant == vn & HR.scenario == "T+2");
        row.(vn + "_NSE_cal") = round(median(s.NSE_cal), 2);
        row.(vn + "_NSE_val") = round(median(s.NSE_val), 2);
        row.(vn + "_KGE_val") = round(median(s.KGE_val), 2);
        row.(vn + "_dS_T2_range") = string(sprintf('%.1f to %.1f', min(r), max(r)));
    end
    row.Ice_share_summer_pct = round(median(HS.ice_share_summer(HS.gauge == gid(k) & HS.variant == "HBV_glac")), 1);
    Tab3 = [Tab3; row]; %#ok<AGROW>
end
writetable(Tab3, fullfile(OUTF, 'Tables.xlsx'), 'Sheet', 'Table3');
disp(Tab3);
fprintf('\nTayyor: %s\n', OUTF);

%% ---------------------------------------------------------------- yordamchi funksiyalar
function f = newfig(w, h)
f = figure('Units', 'centimeters', 'Position', [2 2 w h], 'Color', 'w', 'PaperPositionMode', 'auto');
end

function saveall(fig, name, outf)
exportgraphics(fig, fullfile(outf, [name '.png']), 'Resolution', 600);
exportgraphics(fig, fullfile(outf, [name '.tif']), 'Resolution', 600);
exportgraphics(fig, fullfile(outf, [name '.pdf']), 'ContentType', 'vector');
end

function [r, p] = spear(x, y)
% Spearman rho (bog'langan qiymatlar o'rtacha rang oladi) va ikki tomonlama p (t-yaqinlashuv)
m = isfinite(x) & isfinite(y); x = x(m); y = y(m); n = numel(x);
c = corrcoef(rankavg(x), rankavg(y)); r = c(1, 2);
t2 = r ^ 2 * (n - 2) / max(1 - r ^ 2, eps);
p = betainc((n - 2) / (n - 2 + t2), (n - 2) / 2, 0.5);
end

function r = rankavg(x)
[~, idx] = sort(x); r = zeros(size(x)); r(idx) = 1:numel(x);
[u, ~, g] = unique(x);
for k = 1:numel(u), r(g == k) = mean(r(g == k)); end
end

function placeLegend(lg, ax, fx, fy)
% legendani o'qlar ichida (fx, fy) nisbiy nuqtaga (chap-pastki burchak) joylaydi
drawnow; lg.Location = 'none';
u = ax.Units; ax.Units = lg.Units; p = ax.Position; ax.Units = u; q = lg.Position;
lg.Position = [p(1) + fx * p(3), p(2) + fy * p(4), q(3), q(4)];
end
