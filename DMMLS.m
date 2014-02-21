function [feaIdx] = DMMLS(fea, gnd, lamda, k)

if nargin < 4
    k = 5;
end
options = [];
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = k;
% Newfea = NormalizeFea(fea);
% meansmp = mean(Newfea);
% [nsmp, nfea] = size(Newfea);
% sagma = trace((Newfea - repmat(meansmp, nsmp, 1))'*(Newfea - repmat(meansmp, nsmp, 1))) / nsmp;
% options.t = sqrt(sagma);

Wt = constructW(fea, options);

[nsmp, nfea] = size(fea);
Ww = full(Wt);
for i = 1:nsmp
    for j = 1:nsmp
        if gnd(i) ~= gnd(j)
            Ww(i, j) = 0;
        end
    end
end

Wb = Wt - Ww;
A = lamda*Ww - (1-lamda)*Wb;
D = diag(sum(A, 2));
L = D - A;
Dt = diag(sum(Wt, 2));
dmmls = zeros(nfea, 1);

for i = 1:nfea
    mui = fea(:, i)'*Dt*ones(nsmp, 1)/sum(sum(Dt));
    Lprime = (fea(:, i)-mui*ones(nsmp, 1))'*L*(fea(:, i)-mui*ones(nsmp, 1));
    Dprime = (fea(:, i)-mui*ones(nsmp, 1))'*Dt*(fea(:, i)-mui*ones(nsmp, 1));
%     nmi = mutualinfo(gnd, Dfea(:, i))/max(entropy(gnd), entropy(Dfea(:, i)));
    dmmls(i) = lamda*Lprime/Dprime;
end
[junk, feaIdx] = sort(dmmls);
