function [pre, rec, fs, area] = EvaluatMetricc(dec, gnd)

pre = precision(dec, gnd);
rec = recall(dec, gnd);
area = auc(dec, gnd);
fs = fscore(dec, gnd);
% acc = accuracy(dec, gnd);
end


function ret = precision(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
if tp_fp == 0;
    disp(sprintf('warning: No positive predict label.'));
    ret = 0;
else
    ret = tp / tp_fp;
end
disp(sprintf('Precision = %g%% (%d/%d)', 100.0 * ret, tp, tp_fp));
end

function ret = recall(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fn = sum(label == 1);
if tp_fn == 0;
    disp(sprintf('warning: No postive true label.'));
    ret = 0;
else
    ret = tp / tp_fn;
end
disp(sprintf('Recall = %g%% (%d/%d)', 100.0 * ret, tp, tp_fn));
end

function ret = fscore(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
tp_fn = sum(label == 1);
if tp_fp == 0;
    disp(sprintf('warning: No positive predict label.'));
    precision = 0;
else
    precision = tp / tp_fp;
end
if tp_fn == 0;
    disp(sprintf('warning: No postive true label.'));
    recall = 0;
else
    recall = tp / tp_fn;
end
if precision + recall == 0;
    disp(sprintf('warning: precision + recall = 0.'));
    ret = 0;
else
    ret = 2 * precision * recall / (precision + recall);
end
disp(sprintf('F-score = %g', ret));
end

function ret = auc(dec, label)
[dec idx] = sort(dec, 'descend');
label = label(idx);
tp = cumsum(label == 1);
fp = sum(label == -1);
ret = sum(tp(label == -1));
if tp == 0 | fp == 0;
    disp(sprintf('warning: Too few postive true labels or negative true labels'));
    ret = 0;
else
    ret = ret / tp(end) / fp;
end
    disp(sprintf('AUC = %g', ret));
end
  
% function ret = accuracy(dec, label)
% correct = sum(dec .* label >= 0);
% total = length(dec);
% ret = correct / total;
% disp(sprintf('Accuracy = %g%% (%d/%d)', 100.0 * ret, correct, total));
% end

