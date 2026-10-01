pkg load statistics; warning('off','all');
addpath(genpath(getenv('FSDA'))); addpath(getenv('SHIM'));
randn('seed',1); rand('seed',1);
n=300; x=6*rand(n,3)-3; z=rand(n,1)<.5; y=1.5*x(:,1).*(2*z-1)+x(:,2)+x(:,3)+randn(n,1);
try
  out=tclustreg(y,x,2,12,0.25,0.99,'plots',0,'msg',0,'nsamp',100,'refsteps',10);
  disp(out.bopt)
catch err
  disp(err.message);
  for k=1:min(6,numel(err.stack)), printf('%s line %d\n', err.stack(k).name, err.stack(k).line); end
end
