% Round 9 (descriptive): TCLUST-REG variants asked for by the referees, on every study unit.
%  TCRx99(a):  long search (1000 starts, 50 refinement steps), adaptive second-level trimming
%              in the covariates (alphaX = 0.99, FSDA's Bonferronised option), a = 0.25, 0.40
%  TCReqw(0.40), TCRr1(0.40): long search with equal weights, and with equal scales
%              (restriction factor 1); only for units with K >= 3
%  MON(a):     the monitoring path a = 0, 0.05, ..., 0.40 with the study's settings
%              (300 starts, 10 refinement steps), and MONARI: the adjusted Rand index between
%              the classifications (trimmed units as a class) at consecutive levels, in b1..b8
% Usage (env): FSDA, SHIM, DATA_DIR, UNITS_FILE, OUT_FILE, UNIT_START, UNIT_END, UNIT_STRIDE
pkg load statistics;
warning('off', 'all');
addpath(genpath(getenv('FSDA')));
addpath(getenv('SHIM'));
function r = ari(u, v)
  [~, ~, iu] = unique(u); [~, ~, iv] = unique(v);
  C = accumarray([iu(:) iv(:)], 1);
  n = numel(u); c2 = @(x) x .* (x - 1) / 2;
  s = sum(c2(C(:))); a = sum(c2(sum(C, 2))); b = sum(c2(sum(C, 1)));
  e = a * b / c2(n); m = (a + b) / 2;
  if m == e, r = 1; else r = (s - e) / (m - e); end
end
data_dir = getenv('DATA_DIR');
units = textscan(fopen(getenv('UNITS_FILE')), '%s %f %f', 'Delimiter', ',', 'HeaderLines', 1);
names = units{1}; Ks = units{2}; seeds = units{3};
a = str2double(getenv('UNIT_START')); b = str2double(getenv('UNIT_END'));
if isnan(a), a = 0; end
if isnan(b) || b < 0, b = numel(names); end
st = str2double(getenv('UNIT_STRIDE')); if isnan(st), st = 1; end
fo = fopen(getenv('OUT_FILE'), 'w');
fprintf(fo, 'unit,method,time,alpha_hat');
for j = 1:12, fprintf(fo, ',b%d', j); end
fprintf(fo, '\n');
MONLEV = 0:0.05:0.40;
idx = (a + 1):st:b; if ~isempty(getenv('REVERSE')), s0 = str2double(getenv('REVERSE')); idx = (b - (a - s0)):-st:(s0 + 1); end
for i = idx
  nm = names{i}; K = Ks(i);
  D = dlmread(fullfile(data_dir, [nm '.csv']), ',', 1, 0);
  p = size(D, 2) - 3;
  X = D(:, 1:p); y = D(:, p + 1);
  mode = getenv('R9_MODE');
  runs = {};
  if strcmp(mode, 'x99')
    if isempty(regexp(nm, 'D5-|D6-|D7-', 'once')), continue; end
    runs = {{'TCRx99(0.25)', 0.25, 12, 0.99, 300, 10, false}, {'TCRx99(0.40)', 0.40, 12, 0.99, 300, 10, false}};
  end
  if strcmp(mode, 'r10')
    runs = {{'TCReqwL(0.25)', 0.25, 12, 0, 1000, 50, true}, {'TCReqwL(0.40)', 0.40, 12, 0, 1000, 50, true}, ...
            {'TCRlong(0.30)', 0.30, 12, 0, 1000, 50, false}, {'TCRlong(0.35)', 0.35, 12, 0, 1000, 50, false}};
  end
  if strcmp(mode, 'mon') && K >= 3
    runs{end + 1} = {'TCReqw(0.40)', 0.40, 12, 0, 1000, 50, true};
    runs{end + 1} = {'TCRr1(0.40)', 0.40, 1, 0, 1000, 50, false};
  end
  monl = 1:numel(MONLEV); if ~strcmp(mode, 'mon'), monl = []; end
  for l = monl
    runs{end + 1} = {sprintf('MON(%.2f)', MONLEV(l)), MONLEV(l), 12, 0, 300, 10, false};
  end
  labs = {};
  for r = 1:numel(runs)
    R = runs{r}; meth = R{1};
    rand('seed', seeds(i) + 100 + r); randn('seed', seeds(i) + 100 + r);
    t0 = tic; ok = true;
    try
      out = tclustreg(y, X, K, R{3}, R{2}, R{4}, 'plots', 0, 'msg', 0, 'nsamp', R{5}, 'refsteps', R{6}, 'equalweights', R{7});
    catch err
      ok = false;
    end
    tt = toc(t0); bb = NaN(1, 12); ah = NaN;
    if ok
      B = out.bopt; v = B(:)'; bb(1:numel(v)) = v; ah = mean(out.idx <= 0);
      if strncmp(meth, 'MON', 3), lab = out.idx(:); lab(lab < 0) = 0; labs{end + 1} = lab; end
    elseif strncmp(meth, 'MON', 3)
      labs{end + 1} = [];
    end
    fprintf(fo, '%s,%s,%.4f,%.6f', nm, meth, tt, ah);
    fprintf(fo, ',%.10g', bb);
    fprintf(fo, '\n');
  end
  aris = NaN(1, 12);
  for l = 1:(numel(labs) - 1)
    if ~isempty(labs{l}) && ~isempty(labs{l + 1}), aris(l) = ari(labs{l}, labs{l + 1}); end
  end
  if strcmp(mode, 'mon'), fprintf(fo, '%s,MONARI,0,NaN', nm); fprintf(fo, ',%.6f', aris); fprintf(fo, '\n'); end
  fflush(fo);
end
fclose(fo);
printf('UNITS_DONE=%d\n', numel((a + 1):st:b));
