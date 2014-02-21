clc
clear
disp('computing...')
Dataset=1;       
numoffea = 50;
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
nc = length(unique(TrainingLabel));
classes = unique(TrainingLabel);
FeaNumCandi = [5:5:numoffea];
meas = [TrainingSet; TestingSet];
species = [TrainingLabel; TestingLabel];


CVO = cvpartition(species, 'Leaveout');

result_sp = [];
result_kw = [];
result_relieff = [];
result_lsls = [];

for j = 1:length(FeaNumCandi)
    result_sp1 = [];
    result_kw1 = [];
    result_relieff1 = [];
    result_lsls1 = [];
    
    for i = 1:CVO.NumTestSets
        trIdx = CVO.training(i);
        teIdx = CVO.test(i);
        cvTrainingLabel = species(trIdx);
        cvTestingLabel = species(teIdx);
        cvTrainingSet = meas(trIdx, :);
        cvTestingSet = meas(teIdx, :);
        
        fea_kw = fsKruskalWallis(cvTrainingSet, cvTrainingLabel);
        option = [];
        option.WeightMode = 'HeatKernel';
        option.NeighborMode = 'Supervised';
        option.t = 1;
        option.gnd = cvTrainingLabel;
        w = constructW(cvTrainingSet, option);
        fea_sp = fsSpectrum(w, cvTrainingSet, 0);
        [~, fea_sp] = sort(fea_sp, 'descend');
        
        fea_kw = fea_kw.fList(1:numoffea);
        
        fea_lsls = DMMLS(cvTrainingSet, cvTrainingLabel, alpha);
        
        
        fea_lsls = fea_lsls(1:numoffea);
       
        fea_relieff = relieff(cvTrainingSet, cvTrainingLabel, 3);
        
        fea_relieff = fea_relieff(1:numoffea);
    
        fea_sp = fea_sp(1:numoffea);
        

        cvTrainingSetkw = meas(trIdx, fea_kw(1:FeaNumCandi(j)));
        cvTestingSetkw = meas(teIdx, fea_kw(1:FeaNumCandi(j)));
        
        
        cvTrainingSetsp = meas(trIdx, fea_sp(1:FeaNumCandi(j)));
        cvTestingsetsp = meas(teIdx, fea_sp(1:FeaNumCandi(j)));
        
        cvTrainingrelieff = meas(trIdx, fea_relieff(1:FeaNumCandi(j)));
        cvTestingsetrelieff = meas(teIdx, fea_relieff(1:FeaNumCandi(j)));
        
        cvTestingsetlsls = meas(teIdx, fea_lsls(1:FeaNumCandi(j)));
        cvTrainingsetlsls = meas(trIdx, fea_lsls(1:FeaNumCandi(j)));
        
        result_kw1 = [result_kw1; PredictCrossDatasetWithCG(zscore(cvTrainingSetkw), cvTrainingLabel, zscore(cvTestingSetkw), cvTestingLabel, option1)];
        result_sp1 = [result_sp1; PredictCrossDatasetWithCG(zscore(cvTrainingSetsp), cvTrainingLabel, zscore(cvTestingsetsp), cvTestingLabel, option1)];
        result_relieff1 = [result_relieff1; PredictCrossDatasetWithCG(zscore(cvTrainingrelieff), cvTrainingLabel, zscore(cvTestingsetrelieff), cvTestingLabel, option1)];
        result_lsls1 = [result_lsls1; PredictCrossDatasetWithCG(zscore(cvTrainingsetlsls), cvTrainingLabel, zscore(cvTestingsetlsls), cvTestingLabel, option1)];
    end
    result_kw = [result_kw, mean(result_kw1)];
    result_sp = [result_sp, mean(result_sp1)];
    result_relieff = [result_relieff, mean(result_relieff1)];
    result_lsls = [result_lsls, mean(result_lsls1)];
end


switch Dataset 
    case {1}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvprostate.mat');
        else
            save('.\results\knnprostate.mat');
        end
    case {2}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvAMLALL.mat');
        else
            save('.\results\knnAMLALL.mat');
        end
    case {3}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvDLBCL.mat');
        else
            save('.\results\knnDLBCL.mat');
        end
    case {4}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvlung.mat');
        else
            save('.\results\knnlung.mat');
        end
    case {5}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvMLL.mat');
        else
            save('.\results\knnMLL.mat');
        end

    case {6}
        if strcmpi(option1.classifier, 'svm')
            save('.\results\loocvSRBCT.mat');
        else
            save('.\results\knnSRBCT.mat');
        end
end
fprintf(1,'finished...\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%关机了
% system('shutdown -s');

