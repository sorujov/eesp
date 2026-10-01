function eesp_matlab_check(i0, i1)
% MATLAB check of the TCLUST-REG runs of the paper (run in MATLAB Online with FSDA installed).
% Fits TCLUST-REG with the settings of the study (restriction factor 12, no second-level
% trimming, 1000 random starts, 50 refinement steps) at levels 0.25 and 0.40 to data sets
% i0..i1 of units.csv (80 in all) and writes results_<i0>_<i1>.csv next to this file.
% Each call must finish within 15 minutes on MATLAB Online (Basic): run 1-20, 21-40, 41-60, 61-80.
T = readtable('units.csv', 'TextType', 'string', 'Delimiter', ',');
fn = sprintf('results_%02d_%02d.csv', i0, i1);
fo = fopen(fn, 'w');
fprintf(fo, 'unit,method,time,obj,alpha_hat');
for j = 1:12, fprintf(fo, ',b%d', j); end
fprintf(fo, ',idx\n');
levels = [0.25 0.40];
for i = i0:i1
    D = readmatrix(fullfile('data', char(T.name(i) + ".csv")));
    p = size(D, 2) - 3; X = D(:, 1:p); y = D(:, p + 1); K = T.K(i);
    for r = 1:2
        al = levels(r);
        rng(100 * i + r, 'twister');
        t0 = tic;
        out = tclustreg(y, X, K, 12, al, 0, 'plots', 0, 'msg', 0, 'nsamp', 1000, 'refsteps', 50);
        tt = toc(t0);
        bb = NaN(1, 12); v = out.bopt(:)'; bb(1:numel(v)) = v;
        ix = strjoin(arrayfun(@(q) sprintf('%d', q), out.idx(:)', 'UniformOutput', false), ';');
        fprintf(fo, '%s,TCRlong(%.2f),%.4f,%.10g,%.6f', T.name(i), al, tt, out.obj, mean(out.idx <= 0));
        fprintf(fo, ',%.10g', bb);
        fprintf(fo, ',%s\n', ix);
    end
    fprintf('%d/%d %s done\n', i, i1, T.name(i));
end
fclose(fo);
fprintf('wrote %s\n', fn);
end
