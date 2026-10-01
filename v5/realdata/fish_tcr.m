% TCLUST-REG on the log fishery data, K = 2, several trimming levels and seeds
pkg load statistics; warning('off','all');
addpath(genpath(getenv('FSDA'))); addpath(getenv('SHIM'));
D=dlmread('fishery_log.csv',',',1,0); X=D(:,1); y=D(:,2);
fo=fopen('fish_tcr.csv','w'); fprintf(fo,'alpha,seed,b0_1,b1_1,b0_2,b1_2,trimmed\n');
for al=[0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30]
 for s=1:10
  rand('seed',s); randn('seed',s);
  out=tclustreg(y,X,2,12,al,0,'plots',0,'msg',0,'nsamp',300);
  B=out.bopt; [~,o]=sort(B(1,:)); B=B(:,o);
  fprintf(fo,'%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n',al,s,B(1,1),B(2,1),B(1,2),B(2,2),mean(out.idx<=0));
 end
end
fclose(fo);
