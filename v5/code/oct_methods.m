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
fprintf(fo, '\n');
for i = (a + 1):b
  nm = names{i}; K = Ks(i);
  D = dlmread(fullfile(data_dir, [nm '.csv']), ',', 1, 0);
  p = size(D, 2) - 3;
  X = D(:, 1:p); y = D(:, p + 1);
  runs = {};
  for al = LEVELS
    runs{end + 1} = {sprintf('TCR(%.2f)', al), al, 0};
  end
  if ~isempty(strfind(nm, 'D5-leverage')) || ~isempty(strfind(nm, 'D6-uniform'))
    for al = [0.10 0.25]
      runs{end + 1} = {sprintf('TCRx(%.2f)', al), al, 0.05};
    end
  end
  for r = 1:numel(runs)
    meth = runs{r}{1}; al = runs{r}{2}; ax = runs{r}{3};
    rand('seed', seeds(i) + r); randn('seed', seeds(i) + r);
    t0 = tic;
    ok = true;
    try
      out = tclustreg(y, X, K, 12, al, ax, 'plots', 0, 'msg', 0, 'nsamp', 300);
    catch err
      ok = false;
    end
    tt = toc(t0);
    bb = NaN(1, 12);
    ah = NaN;
    if ok
      B = out.bopt;                 % (p+1) x K
      v = B(:)';
      bb(1:numel(v)) = v;
      ah = mean(out.idx <= 0);
    end
    fprintf(fo, '%s,%s,%.4f,%.6f', nm, meth, tt, ah);
    fprintf(fo, ',%.10g', bb);
    fprintf(fo, '\n');
  end
  fflush(fo);
end
fclose(fo);
printf('UNITS_DONE=%d\n', b - a);
