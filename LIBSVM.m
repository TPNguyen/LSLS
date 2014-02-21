function [Acc Fc Fg]=LIBSVM(CX,Label,kfold)
%%% Cx--data
%%% Label--label

%[row column]=size(C_x);
%x=C_x(:,[1:column-1]);
%y=C_x(:,column);

Cvalue = 0;
%Fold = 4;  %10 100%
C=0;
G=0;
Loocv=10;

for i = -1:10
    Cvalue=2^i;
    for j =-10:1
        Gamma=2^j;
        SVMParameter = sprintf('-t 2 -c %d -g %f -v %d',Cvalue,Gamma,kfold);
        LoocvTemp = svmtrain(Label,CX, SVMParameter);
            
        if LoocvTemp > Loocv
           Loocv=LoocvTemp;
           C=Cvalue;
           G=Gamma;
        end
        if Loocv>99.999
           break
        end
    end
    if Loocv>99.999
      break
    end
end
 Acc=Loocv;
 Fc=C;
 Fg=G;

return