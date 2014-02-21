clc
clear
disp('computing...')
Dataset=5;       
numoffea = 150;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('default');
%所有样本均按行排列，行样本，列特征
TrainingSet = [];
TestingSet = [];
TrainingLabel = [];
TestingLabel = [];

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

meas = [TrainingSet; TestingSet];
species = [TrainingLabel; TestingLabel];


CVO = cvpartition(species, 'Leaveout');

opt = statset('display','iter');

option = [];
option.WeightMode = 'HeatKernel';
option.NeighborMode = 'Supervised';
option.t = 1;
option.gnd = species;
w = constructW(meas, option);
fea_sp = fsSpectrum(w, meas, 0);
[~, fea_sp] = sort(fea_sp, 'descend');


fea_lsls = DMMLS(meas, species, alpha);

fea_kw = fsKruskalWallis(meas, species);
fea_kw = fea_kw.fList(1:numoffea);
fea_lsls = fea_lsls(1:numoffea);

fea_relieff = relieff(meas, species, 3);

fea_relieff = fea_relieff(1:numoffea);

fea_sp = fea_sp(1:numoffea);
[fs_sp, history_sp] = sequentialfs(@knnwrapper, meas(:, fea_sp), species, 'cv', CVO, 'options', opt, 'direction', 'backward');
[fs_relieff, history_relieff] = sequentialfs(@knnwrapper, meas(:, fea_relieff), species, 'cv', CVO, 'options', opt, 'direction', 'backward');
[fs_lsls, history_lsls] = sequentialfs(@knnwrapper, meas(:, fea_lsls), species, 'cv', CVO, 'options', opt, 'direction', 'backward');
[fs_kw, history_kw] = sequentialfs(@knnwrapper, meas(:, fea_kw), species, 'cv', CVO, 'options', opt, 'direction', 'backward');

switch Dataset 
    case {1}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcpprostate.mat');
        else
            save('.\results\knnprostate.mat');
        end
    case {2}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcpAMLALL.mat');
        else
            save('.\results\knnAMLALL.mat');
        end
    case {3}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcpDLBCL.mat');
        else
            save('.\results\knnDLBCL.mat');
        end
    case {4}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcplung.mat');
        else
            save('.\results\knnlung.mat');
        end
    case {5}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcpMLL.mat');
        else
            save('.\results\knnMLL.mat');
        end

    case {6}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\bcpSRBCT.mat');
        else
            save('.\results\knnSRBCT.mat');
        end
end
fprintf(1,'finished...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%关机了
% system('shutdown -s');

