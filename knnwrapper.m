function err = knnwrapper(xTrain, yTrain, xTest, yTest)
c = knnsearch(xTrain, xTest);

err = sum(yTrain(c) ~= yTest);
end