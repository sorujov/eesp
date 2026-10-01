% TCLUST-REG on the tone data (K=2) with and without the ten added outliers at (0,4)
pkg load statistics; warning('off','all');
addpath(genpath(getenv('FSDA'))); addpath(getenv('SHIM'));
fo=fopen('tone_tcr.csv','w'); fprintf(fo,'data,alpha,seed,b0_1,b1_1,b0_2,b1_2,trimmed\n');
for f={'tone_clean','tone_out'}
 D=dlmread([f{1} '.csv'],',',1,0); X=D(:,1); y=D(:,2);
 for al=[0.02 0.05 0.10 0.15 0.25 0.40]
  for s=1:10
   rand('seed',s); randn('seed',s);
   out=tclustreg(y,X,2,12,al,0,'plots',0,'msg',0,'nsamp',300);
   B=out.bopt; [~,o]=sort(B(2,:)); B=B(:,o);
   fprintf(fo,'%s,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n',f{1},al,s,B(1,1),B(2,1),B(1,2),B(2,2),mean(out.idx<=0));
  end
 end
end
fclose(fo);
