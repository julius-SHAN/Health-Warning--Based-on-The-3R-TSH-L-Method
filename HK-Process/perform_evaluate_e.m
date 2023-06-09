N=17;
Conf_Mat=[ACC_all(N,3),ACC_all(N,4)-ACC_all(N,3);...
    ACC_all(N,7)-ACC_all(N,6),ACC_all(N,6)];
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
pe=(Conf_Mat(1,1)*sum(Conf_Mat(1,:))+Conf_Mat(2,2)*sum(Conf_Mat(2,:)))/sum(Conf_Mat(:))^2;
kappa=(ACC_all(N,1)/100-pe)/(1-pe);
p_e_m=mean(per_eva)/100;
F1_mean=(2*p_e_m(5)*p_e_m(6))/(p_e_m(5)+p_e_m(6));
per=[F1_mean,kappa];
