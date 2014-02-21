function [index] = RFS(fea, gnd, lamda)
[nsmp, nfea] = size(fea);
nc = length(unique(gnd));
lbmtx = zeros(nc, nsmp);

for i = 1:nc
    ci = find(gnd == i);
    lbmtx(i, ci) = 1;
end
fea = [fea'; ones(1, nsmp)];

rho=lamda; 
opts=[];
q = 2;

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=100;   % maximum number of iterations

% Normalization
opts.nFlag=0;       % without normalization

% Regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)

% Group Property
opts.q=q;           % set the value for q

%----------------------- Run the code mcLeastR -----------------------
fprintf('\n mFlag=0, lFlag=0 \n');
opts.mFlag=0;       % treating it as compositive function 
opts.lFlag=0;       % Nemirovski's line search
tic;
[x1, funVal1, ValueL1]= mcLeastR(fea', lbmtx', rho, opts);
toc;
% imagesc(x1), colormap jet, colorbar
vct = zeros(1, nfea);
for i = 1:nfea
    vct(i) = norm(x1(i, :));
end
[junk, index] = sort(vct, 'descend');