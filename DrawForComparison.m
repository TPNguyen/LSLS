clc;
clear all;
Dataset=1;
option = [];
option.classifier = 'svm';
%figure1=figure('PaperSize',[16.98 25.68]);
set(gca,'FontSize',14);
switch Dataset  
    case {1}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmprostate.mat');
        else
            load('.\results\knnprostate.mat');
        end
        title('Prostate')
        hold on
    case {2}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmAMLALL.mat');
        else
            load('.\results\knnAMLALL.mat');
        end
        title('AMLALL')
    case {3}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmDLBCL.mat');
        else
            load('.\results\knnDLBCL.mat');
        end
        title('DLBCL')
    case {4}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmlung.mat');
        else
            load('.\results\knnlung.mat');
        end
        title('Lung')
    case {5}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmMLL.mat');
        else
            load('.\results\knnMLL.mat');
        end
        title('MLL')
    case {6}
        if strcmpi(option.classifier, 'svm')
            load('.\results\svmSRBCT.mat');
        else
            load('.\results\knnSRBCT.mat');
        end
        title('SRBCT')
end

X_Coord =FeaNumCandi;


% figure1=figure('PaperSize',[16.98 25.68]);



box('on');
hold on
plot(X_Coord, result_Relieff, '-+m');
hold on;

plot(X_Coord, result_SP, '-ko');
hold on;
plot(X_Coord, result_KW, '-g*');
hold on;
plot(X_Coord,result_ERLMS,'-rv');
hold on;
ylim([0.1,1])

legend('Relieff', 'SPFS','KW', 'LSLS','Location', 'SouthEast');
xlabel('Number of genes','fontsize',14);
ylabel('Prediction accuracy','fontsize',14);

    