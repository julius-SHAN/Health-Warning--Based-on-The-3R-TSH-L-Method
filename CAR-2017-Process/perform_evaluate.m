function  per_eva=perform_evaluate(Conf_Mat)
n=size(Conf_Mat,1);
per_eva=zeros(n,8);
for i=1:n
    TP=Conf_Mat(i,i);%真阳性
    FN=sum(Conf_Mat(i,:))-Conf_Mat(i,i);%假阴性
    FP=sum(Conf_Mat(:,i))-Conf_Mat(i,i);%假阳性
    TN=sum(Conf_Mat(:))-TP-FN-FP;%真阴性
    PRE=TP/(TP+FP);%精确率
    REC=TP/(TP+FN);%召回率或者灵敏度
    SPC=TN/(TN+FP);%特异性
    F1=(2*REC*PRE)/(REC+PRE);
    per_eva(i,:)=[TP,FN,FP,TN,PRE*100,REC*100,SPC*100,F1];
end
end