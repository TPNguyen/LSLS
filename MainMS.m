clc
clear
disp('computing...')
Dataset=2;       
numoffea = 400;
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

nc = length(unique(TrainingLabel));
classes = unique(TrainingLabel);
FeaNumCandi = 150;

result_Relieff  =[];
result_KW = [];
result_SP = [];
result_ERLMS = [];
result_gini = [];

c = 0.1:0.1:0.9;
k = 3:20;
[C, K] = meshgrid(c, k);
Z = zeros(9, 18);


% 
for i = 1:length(c)
    for j = 1:length(k)

        fea_ERLMS = DMMLS(TrainingSet, TrainingLabel, c(i), k(j));
        fea_dmmls = fea_ERLMS(1:150);
        z = PredictCrossDatasetWithCG(zscore(TrainingSet(:, fea_dmmls(1:FeaNumCandi))),TrainingLabel,zscore(TestingSet(:, fea_dmmls(1:FeaNumCandi))),TestingLabel, option1);
        Z(i, j) = z;
        
    end
end
set(gca,'FontSize',14);
% surf(C', K', Z);
bar3(Z);
colorbar
xlabel('Alpha')
ylabel('K');
zlabel('Accuracy');
switch Dataset 
    case {1}
        title('Prostate');
    case {2}
        title('AMLALL');
    case {3}
        title('DLBCL');
    case {4}
        title('Lung');
    case {5}
        title('MLL');
    case {6}
        title('SRBCT');
end
fprintf(1,'finished...\n');



    
    
    


