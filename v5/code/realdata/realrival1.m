% One TCLUST-REG long-search fit: env RR_DATA, RR_ALPHA, RR_SEED; appends one row to RR_OUT.
pkg load statistics; warning('off','all');
addpath(genpath(getenv('FSDA'))); addpath(getenv('SHIM'));
f=getenv('RR_DATA'); al=str2double(getenv('RR_ALPHA')); s=str2double(getenv('RR_SEED'));
D=dlmread([f '.csv'],',',1,0); X=D(:,1); y=D(:,2);
rand('seed',s); randn('seed',s);
out=tclustreg(y,X,2,12,al,0,'plots',0,'msg',0,'nsamp',1000,'refsteps',50);
B=out.bopt;
fo=fopen(getenv('RR_OUT'),'w');
fprintf(fo,'%s,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n',f,al,s,B(1,1),B(2,1),B(1,2),B(2,2),mean(out.idx<=0));
fclose(fo);
