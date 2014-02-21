function err = svmwrapper(xTrain, yTrain, xTest, yTest)
%[Acc Fc, Fg] = LIBSVM(xTrain,yTrain,5);
SVMParameter = sprintf('-t 0');

model = svmtrain(yTrain, xTrain, SVMParameter);
err = sum(svmpredict(yTest, xTest, model) ~= yTest);
end