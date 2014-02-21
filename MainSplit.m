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
% 分层十折交叉验证选取alpha
% c = 0.1:0.1:0.9;
% res = zeros(1, length(c));
% for j = 1:length(c)
%     cvp = cvpartition(TrainingLabel, 'kfold', 10);
%     err = zeros(cvp.NumTestSets, 1);
%     for i = 1:cvp.NumTestSets
%         trIdx = cvp.training(i);
%         teIdx = cvp.test(i);
%         fea_ERLMS = DMMLS(TrainingSet(trIdx, :), TrainingLabel(trIdx), c(j));
%         fea_ERLMS = fea_ERLMS(1:200);
%         err(i) = knnwrapper(TrainingSet(trIdx, fea_ERLMS), TrainingLabel(trIdx), TrainingSet(teIdx, fea_ERLMS), TrainingLabel(teIdx));
%     end
%     res(j) = mean(err);
% end
% [~, idm] = min(res);
% alpha = c(idm);
% 
% 
% disp('Parameter optimization is completed...')
%%%%%%%%%%%%%%%%%%%%
if nc == 2
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
    
    for j = 1:length(FeaNumCandi)
        
        result_KW = [result_KW PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_kw(1:FeaNumCandi(j)))),gndTrain,zscore(TestingSet(:, fea_kw(1:FeaNumCandi(j)))),gndTest, option1)];
        result_Relieff = [result_Relieff PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_releiff(1:FeaNumCandi(j)))),gndTrain,zscore(TestingSet(:, fea_releiff(1:FeaNumCandi(j)))),gndTest, option1)];
        result_ERLMS = [result_ERLMS PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_dmmls(1:FeaNumCandi(j)))),gndTrain,zscore(TestingSet(:, fea_dmmls(1:FeaNumCandi(j)))),gndTest, option1)];
        result_SP = [result_SP PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_sp(1:FeaNumCandi(j)))),gndTrain,zscore(TestingSet(:, fea_sp(1:FeaNumCandi(j)))),gndTest, option1)];
    
    end
    
    
else
    
    fea_ERLMS = DMMLS(TrainingSet, TrainingLabel, alpha);
    
    fea_relieff = relieff(TrainingSet, TrainingLabel, 3);
    fea_relieff = fea_relieff(1:numoffea);

    fea_dmmls = fea_ERLMS(1:numoffea);
    
    fea_kw = fsKruskalWallis(TrainingSet, TrainingLabel);
    fea_kw = fea_kw.fList(1:numoffea);
    
    option = [];
    option.WeightMode = 'HeatKernel';
    option.NeighborMode = 'Supervised';
    option.t = 1;
    option.gnd = TrainingLabel;
    w = constructW(TrainingSet, option);
    fea_sp = fsSpectrum(w, TrainingSet, 0);
    
    [~, fea_sp] = sort(fea_sp, 'descend');
    fea_sp = fea_sp(1:numoffea);
    for i = 1:nc
        result_Relieff1 = [];
        result_KW1 = [];
        result_SP1 = [];
        result_ERLMS1 = [];
        restlt_gini1 = [];
        newTrainingLabel = TrainingLabel;
        newTestingLabel = TestingLabel;
        j = (TrainingLabel == classes(i));
        newTrainingLabel(j) = 1;
        newTrainingLabel(~j) = -1;
        j = (TestingLabel == classes(i));
        newTestingLabel(j) = 1;
        newTestingLabel(~j) = -1;
        
        newTrainingLabel = newTrainingLabel(:);
        newTestingLabel = newTestingLabel(:);
        
        
        
        for j = 1:length(FeaNumCandi)
            result_KW1 = [result_KW1 PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_kw(1:FeaNumCandi(j)))),newTrainingLabel(:),zscore(TestingSet(:, fea_kw(1:FeaNumCandi(j)))),newTestingLabel(:), option1)];
            result_SP1 = [result_SP1 PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_sp(1:FeaNumCandi(j)))),newTrainingLabel(:),zscore(TestingSet(:, fea_sp(1:FeaNumCandi(j)))),newTestingLabel(:), option1)];
            result_Relieff1 = [result_Relieff1 PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_relieff(1:FeaNumCandi(j)))),newTrainingLabel(:),zscore(TestingSet(:, fea_relieff(1:FeaNumCandi(j)))),newTestingLabel(:), option1)];
            
            result_ERLMS1 = [result_ERLMS1 PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_dmmls(1:FeaNumCandi(j)))),newTrainingLabel(:),zscore(TestingSet(:, fea_dmmls(1:FeaNumCandi(j)))),newTestingLabel(:), option1)];
        end
        
        result_KW = [result_KW; result_KW1];
        result_SP = [result_SP; result_SP1];
        result_ERLMS = [result_ERLMS; result_ERLMS1];
        result_Relieff = [result_Relieff; result_Relieff1];
        
    end
    result_KW = mean(result_KW);
    result_SP = mean(result_SP);
    result_ERLMS = mean(result_ERLMS);
    result_Relieff = mean(result_Relieff);
    
end


switch Dataset 
    case {1}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmprostate.mat');
        else
            save('.\results\knnprostate.mat');
        end
    case {2}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmAMLALL.mat');
        else
            save('.\results\knnAMLALL.mat');
        end
    case {3}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmDLBCL.mat');
        else
            save('.\results\knnDLBCL.mat');
        end
    case {4}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmlung.mat');
        else
            save('.\results\knnlung.mat');
        end
    case {5}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmMLL.mat');
        else
            save('.\results\knnMLL.mat');
        end

    case {6}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\svmSRBCT.mat');
        else
            save('.\results\knnSRBCT.mat');
        end
end
fprintf(1,'finished...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%关机了
% system('shutdown -s');

