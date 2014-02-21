function [acc, pre, rec, fs, area] = PredictCrossDatasetWithCG(TrainingSet,TrainingLabel,TestingSet,TestingLabel, option)

if ~isfield(option, 'bmetrics')
    option.bmetrics = 'false';
end

if nargin < 5
    option.classifier = 'svm';
    option.bmetrics = 'false';
end

if strcmpi(option.classifier, 'svm')
%     [Acc Fc Fg] = LIBSVM(TrainingSet,TrainingLabel,5);
%     SVMParameter = sprintf('-t 2 -c %d -g %f',Fc,Fg);
    SVMParameter = sprintf('-t 0');
    PredictModel = svmtrain(TrainingLabel, TrainingSet, SVMParameter);
    [predicted_label, LoocvTemp, decision_values] = svmpredict(TestingLabel,TestingSet,PredictModel);
    acc = LoocvTemp(1) / 100;
    if strcmpi(option.bmetrics, 'true')
        [pre, rec, fs, area] = EvaluatMetric(decision_values.*PredictModel.Label(1), TestingLabel(:));
    end
elseif strcmpi(option.classifier, 'knn')
    [idx, d] = knnsearch(TrainingSet, TestingSet);
    acc = sum(TrainingLabel(idx) == TestingLabel) / length(TestingLabel);
    if strcmpi(option.bmetrics, 'true')
        [pre, rec, fs, area] = EvaluatMetric(d(:) .* TrainingLabel(idx), TestingLabel(:));
    end
else
    error('no such classifier!');
end
return