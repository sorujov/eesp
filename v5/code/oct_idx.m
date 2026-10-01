% Round 12: as oct_long.m (same seeds), also writing TCLUST-REG's own assignment out.idx.
% TCLUST-REG (FSDA tclustreg, Garcia-Escudero et al. 2010) for the v4 study.
% Run under GNU Octave with FSDA on the path and two shims for MATLAB-only
% calls (verLessThanFS -> false, coder.target('MATLAB') -> true,
% convertStringsToChars -> identity).  No FSDA code is modified.
% Usage (env): FSDA, SHIM, DATA_DIR, UNITS_FILE, OUT_FILE, UNIT_START, UNIT_END
pkg load statistics;
warning('off', 'all');
addpath(genpath(getenv('FSDA')));
addpath(getenv('SHIM'));
data_dir = getenv('DATA_DIR');
units = textscan(fopen(getenv('UNITS_FILE')), '%s %f %f', 'Delimiter', ',', 'HeaderLines', 1);
names = units{1}; Ks = units{2}; seeds = units{3};
a = str2double(getenv('UNIT_START')); b = str2double(getenv('UNIT_END'));
if isnan(a), a = 0; end
if isnan(b) || b < 0, b = numel(names); end
LEVELS = [0.05 0.10 0.15 0.25 0.40];
fo = fopen(getenv('OUT_FILE'), 'w');
fprintf(fo, 'unit,method,time,alpha_hat');
for j = 1:12, fprintf(fo, ',b%d', j); end
fprintf(fo, ',idx\n');
st = str2double(getenv('UNIT_STRIDE')); if isnan(st), st = 1; end
idx = (a + 1):st:b; if ~isempty(getenv('REVERSE')), s0 = str2double(getenv('REVERSE')); idx = (b - (a - s0)):-st:(s0 + 1); end
for i = idx
  nm = names{i}; K = Ks(i);
  D = dlmread(fullfile(data_dir, [nm '.csv']), ',', 1, 0);
  p = size(D, 2) - 3;
  X = D(:, 1:p); y = D(:, p + 1);
  runs = {};
  for al = LEVELS
    runs{end + 1} = {sprintf('TCR(%.2f)', al), al, 0};
  end
  % Long-search check in every design (round 8): TCLUST-REG with 1000 starts and 50 refinement steps: TCLUST-REG at level 0.40 with the study's
  % settings, with more starts and refinement steps, and with a looser restriction factor
  runs = {{'TCRlong(0.25)', 12, 1000, 50, 0.25}, {'TCRlong(0.40)', 12, 1000, 50, 0.40}};
  for r = 1:numel(runs)
    meth = runs{r}{1}; rf = runs{r}{2}; ns = runs{r}{3}; rs = runs{r}{4}; al = runs{r}{5};
    rand('seed', seeds(i) + r); randn('seed', seeds(i) + r);
    t0 = tic; ok = true;
    try
      out = tclustreg(y, X, K, rf, al, 0, 'plots', 0, 'msg', 0, 'nsamp', ns, 'refsteps', rs);
    catch err
      ok = false;
    end
    tt = toc(t0); bb = NaN(1, 12); ah = NaN; ix = '';
    if ok
      B = out.bopt; v = B(:)'; bb(1:numel(v)) = v; ah = mean(out.idx <= 0); ix = strjoin(arrayfun(@(q) sprintf('%d', q), out.idx(:)', 'UniformOutput', false), ';');
    end
    fprintf(fo, '%s,%s,%.4f,%.6f', nm, meth, tt, ah);
    fprintf(fo, ',%.10g', bb);
    fprintf(fo, ',%s\n', ix);
  end
  fflush(fo);
end
fclose(fo);
printf('UNITS_DONE=%d\n', numel((a + 1):st:b));
