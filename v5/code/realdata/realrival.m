% Round 8: TCLUST-REG with 1000 starts and 50 refinement steps at levels 0.25 and 0.40 on the
% fishery (log scale) and tone data (original and with ten added points), 10 seeds each.
pkg load statistics; warning('off','all');
addpath(genpath(getenv('FSDA'))); addpath(getenv('SHIM'));
fo=fopen('realrival.csv','w'); fprintf(fo,'data,alpha,seed,b0_1,b1_1,b0_2,b1_2,trimmed\n');
for f={'fishery_log','tone_clean','tone_out'}
 D=dlmread([f{1} '.csv'],',',1,0); X=D(:,1); y=D(:,2);
 for al=[0.25 0.40]
  for s=1:10
   rand('seed',s); randn('seed',s);
   out=tclustreg(y,X,2,12,al,0,'plots',0,'msg',0,'nsamp',1000,'refsteps',50);
   B=out.bopt;
   fprintf(fo,'%s,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n',f{1},al,s,B(1,1),B(2,1),B(1,2),B(2,2),mean(out.idx<=0));
  end
 end
end
fclose(fo);
