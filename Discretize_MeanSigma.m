function DData = Discretize_MeanSigma(Data)

% unsupervized data discretization
% discretize the data into three partitions{-1 0 1} with mean and sigma
% if data<mean-sigma,data=-1,ifdata>mean+sigma,data=+1,else data=0
%
% Inputs: 
%    Data  - a n by p matrix, where n is the number of samples and 
%            p the number of features
%
% Outputs:
%    DData  - discretized data

% Ref: Liang J., Yang S.&Winstanley, A. 2008
% Completed on 5 Nov 2008

% Data=[1.2 3.4 5 8 0.5; 3.5 2.6 4.1 2 6]'
[nSamples, nFeatures] = size(Data);
DData = zeros(nSamples, nFeatures);
dataMean = mean(Data,1);
dataSigma = std(Data,[],1);
Datatmp = Data-repmat((dataMean-dataSigma),nSamples,1);
ind = find(Datatmp < 0);
DData(ind) = -1;
Datatmp = Data-repmat((dataMean+dataSigma),nSamples,1);
ind = find(Datatmp > 0);
DData(ind) = 1;