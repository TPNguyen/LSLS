clc
clear
disp('computing...')
Dataset=4;       
numoffea = 400;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('default');
%所有样本均按行排列，行样本，列特征
TrainingSet = [];
TestingSet = [];
TrainingLabel = [];
TestingLabel = [];
bmetrics = 1;
option1 = [];
option1.bmetrics = 'false';
option1.classifier = 'svm';
set(gca,'FontSize',14);
switch Dataset                   
    case {1}
        load .\DataSets\prostate\prostate_TumorVSNormal_test.data;
        load .\DataSets\prostate\prostate_TumorVSNormal_train.data;
        %postive {+1}
        TrainingSet = prostate_TumorVSNormal_train(:, 1:end-1);
        TrainingLabel = prostate_TumorVSNormal_train(:, end);
        TestingLabel=prostate_TumorVSNormal_test(:, end);
        TestingSet=prostate_TumorVSNormal_test(:, 1:end-1);
        clear prostate_TumorVSNormal_test;
        clear prostate_TumorVSNormal_train;
    case {2}
        load .\DataSets\ALL-AML_Leukemia\AMLALL_train.data;
        load .\DataSets\ALL-AML_Leukemia\AMLALL_test.data;
        
        TrainingSet = AMLALL_train(:, 1:end-1);
        TrainingLabel = AMLALL_train(:, end);
        TestingLabel=AMLALL_test(:, end);
        TestingSet=AMLALL_test(:, 1:end-1);
        clear AMLALL_test;
        clear AMLALL_train;
        %postive{+2}
    case {3}
        load .\DataSets\DLBCLTrainingTest.mat
        
        
    case {4}
        load .\DataSets\lungcancer\lungCancer_train.data
        load .\DataSets\lungcancer\lungCancer_test.data
        TrainingSet = lungCancer_train(:, 1:end-1);
        TrainingLabel = lungCancer_train(:, end);
        TestingLabel=lungCancer_test(:, end);
        TestingSet=lungCancer_test(:, 1:end-1);
        clear lungCancer_test;
        clear lungCancer_train;

    case {5}
        load .\DataSets\MLL_leukemia\MLL_train.data
        load .\DataSets\MLL_leukemia\MLL_test.data
        TrainingSet = MLL_train(:, 1:end-1);
        TrainingLabel = MLL_train(:, end);
        TestingLabel=MLL_test(:, end);
        TestingSet=MLL_test(:, 1:end-1);
        clear MLL_test;
        clear MLL_train;

    case {6}
%           load .\DataSets\ALL\TEL-AML1-train.data
%           load .\DataSets\ALL\TEL-AML1-test.data
%           load .\DataSets\ALL\BCR-ABL-train.data
%           load .\DataSets\ALL\BCR-ABL-test.data
%           load .\DataSets\ALL\E2A-PBX1-train.data
%           load .\DataSets\ALL\E2A-PBX1-test.data
%           load .\DataSets\ALL\Hyperdip50-train.data
%           load .\DataSets\ALL\Hyperdip50-test.data
%           load .\DataSets\ALL\MLL-train.data
%           load .\DataSets\ALL\MLL-test.data
%           load .\DataSets\ALL\OTHERS-train.data
%           load .\DataSets\ALL\OTHERS-test.data
%           load .\DataSets\ALL\T-ALL-train.data
%           load .\DataSets\ALL\T-ALL-test.data
% 
%           TrainingSet = [BCR_ABL_train; E2A_PBX1_train(:, 1:end-1); Hyperdip50_train(:, 1:end-1); MLL_train(:, 1:end-1); OTHERS_train(:, 1:end-1); 
%               TEL_AML1_train(:, 1:end-1); T_ALL_train(:, 1:end-1)];
%           TrainingLabel = [ones(size(BCR_ABL_train, 1), 1); E2A_PBX1_train(:, end); Hyperdip50_train(:, end); MLL_train(:, end); OTHERS_train(:, end); 
%               TEL_AML1_train(:, end); T_ALL_train(:, end)];
%           TestingLabel=[ones(size(BCR_ABL_test, 1), 1); E2A_PBX1_test(:, end); Hyperdip50_test(:, end); MLL_test(:, end); OTHERS_test(:, end); 
%               TEL_AML1_test(:, end); T_ALL_test(:, end)];
%           TestingSet=[BCR_ABL_test(:, 1:end-1); E2A_PBX1_test(:, 1:end-1); Hyperdip50_test(:, 1:end-1); MLL_test(:, 1:end-1); OTHERS_test(:, 1:end-1); 
%               TEL_AML1_test(:, 1:end-1); T_ALL_test(:, 1:end-1)];
        load .\DataSets\SRBCTTrainingTest.mat
end
disp('Data loading is completed...');
%分别处理两类问题与多累问题
nc = length(unique(TrainingLabel));
classes = unique(TrainingLabel);
FeaNumCandi = [10:10:numoffea];

result_Relieff  =[];
result_KW = [];
result_SP = [];
result_ERLMS = [];
result_gini = [];

%%%%%%%%%%%%%%%%%%%%

postive = classes(1);
if sum(TrainingLabel == classes(1)) > sum(TrainingLabel == classes(2))
    postive = classes(2);
end
gndTrain = relabemin(TrainingLabel, postive);
gndTest = relabemin(TestingLabel, postive);

tic
fea_kw = fsKruskalWallis(TrainingSet, gndTrain);
espt = toc;
disp(['kruskalwallis took ' num2str(espt) ' seconds.'])

tic
option = [];
option.WeightMode = 'HeatKernel';
option.NeighborMode = 'Supervised';
option.t = 1;
option.gnd = TrainingLabel;
w = constructW(TrainingSet, option);
fea_sp = fsSpectrum(w, TrainingSet, 0);
[~, fea_sp] = sort(fea_sp, 'descend');
espt = toc;
disp(['Spectrum feature slection took ' num2str(espt) ' seconds.'])
fea_kw = fea_kw.fList(1:numoffea);


tic
fea_ERLMS = DMMLS(TrainingSet, gndTrain, alpha);
espt = toc;
disp(['LSLS took ' num2str(espt) ' seconds.'])

fea_dmmls = fea_ERLMS(1:numoffea);
tic;
fea_releiff = relieff(TrainingSet, TrainingLabel, 3);
espt = toc;
disp(['relieff feature selection took ' num2str(espt) ' seconds.'])
fea_releiff = fea_releiff(1:numoffea);

fea_sp = fea_sp(1:numoffea);

SVMParameter = sprintf('-t 0');
PredictModelkw = svmtrain(gndTrain, TrainingSet(:, fea_kw(1:150)), SVMParameter);
PredictModelrelieff = svmtrain(gndTrain, TrainingSet(:, fea_releiff(1:150)), SVMParameter);
PredictModelerlms = svmtrain(gndTrain, TrainingSet(:, fea_ERLMS(1:150)), SVMParameter);
PredictModelsp = svmtrain(gndTrain, TrainingSet(:, fea_sp(1:150)), SVMParameter);
       

[predict_label,mse,deci] = svmpredict(gndTest, TestingSet(:, fea_kw(1:150)), PredictModelkw);
[xkw, ykw, t, auc1] = perfcurve(gndTest, deci*PredictModelkw.Label(1), 1);
[predict_label,mse,deci] = svmpredict(gndTest, TestingSet(:, fea_releiff(1:150)), PredictModelrelieff);
[xre, yre, t, auc2] = perfcurve(gndTest, deci*PredictModelrelieff.Label(1), 1);
[predict_label,mse,deci] = svmpredict(gndTest, TestingSet(:, fea_ERLMS(1:150)), PredictModelerlms);
[xls, yls, t, auc3] = perfcurve(gndTest, deci*PredictModelerlms.Label(1), 1);
[predict_label,mse,deci] = svmpredict(gndTest, TestingSet(:, fea_sp(1:150)), PredictModelsp);
[xsp, ysp, t, auc4] = perfcurve(gndTest, deci*PredictModelsp.Label(1), 1);

switch Dataset
    case {1}
        plot(xre, yre, '-+m');
        hold on
        plot(ysp, xsp, '-ko');
        hold on
        plot(xkw, ykw, '-g*');
        hold on
        plot(xls, yls, '-rv');
        hold on
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('Prostate');
        legend('Relieff', 'SPFS','KW', 'LSLS','Location', 'SouthEast');
    case {2}
        plot(xre, yre, '-+m');
        hold on
        plot(xsp, ysp, '-ko');
        hold on
        plot(xkw, ykw, '-g*');
        hold on
        plot(xls, yls, '-rv');
        hold on
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('AMLALL');
        legend('Relieff', 'SPFS','KW', 'LSLS','Location', 'SouthEast');
    case {3}
        plot(xre, yre, '-+m');
        hold on
        plot(xsp, ysp, '-ko');
        hold on
        plot(xkw, ykw, '-g*');
        hold on
        plot(xls, yls, '-rv');
        hold on
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('DLBCL');
        legend('Relieff', 'SPFS','KW', 'LSLS','Location', 'SouthEast');
    case {4}
        plot(xre, yre, '-+m');
        hold on
        plot(xsp, ysp, '-ko');
        hold on
        plot(xkw, ykw, '-g*');
        hold on
        plot(xls, yls, '-rv');
        hold on
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title('Lung');
        legend('Relieff', 'SPFS','KW', 'LSLS','Location', 'SouthEast');
end
    
    


