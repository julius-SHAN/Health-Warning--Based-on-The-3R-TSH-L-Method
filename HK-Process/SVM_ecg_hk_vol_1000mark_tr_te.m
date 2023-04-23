clc;clear;close all;
% 作用：将根据label_1000生成的训练集和测试集输入进RF中，并提取TS特征进行分类识别
st=tic;
%% 导入数据
path='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\ecg_hk_tr_te\';
filename=[path,'label_1000_16_tr_te.mat'];
load(filename);
%% 特征提取
fprintf('Feature extracting and normalizing...\n');
Fs=360;
op=1;
len_tr=size(traindata,1);
len_te=size(testdata,1);
Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];
len_Y_fr=length(Y_fr);
trainfeat=zeros(len_tr,4+len_Y_fr);
testfeat=zeros(len_te,4+len_Y_fr);
for i=1:len_tr
    zind=find(traindata(i,:),1,'last');
    sig=traindata(i,7:zind);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Y=abs(s(start-1:end));
    deltaf=Fs/len_sig;
    Y_uniband=zeros(1,len_Y_fr);
    Ycount=[2,zeros(1,len_Y_fr)];
    for n=1:len_Y_fr
        Ycount(n+1)=round((Y_fr(n))/deltaf)+1;
        temp=sqrt(sum(Y(Ycount(n):Ycount(n+1)).^2));
        Y_uniband(n)=log10(temp);
    end
    trainfeat(i,:)=[traindata(i,4)/360,traindata(i,5)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1)];
end
%测试集特征
for i=1:len_te
    zind=find(testdata(i,:),1,'last');
    sig=testdata(i,7:zind);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Y=abs(s(start-1:end));
    deltaf=Fs/len_sig;
    Y_uniband=zeros(1,len_Y_fr);
    Ycount=[2,zeros(1,len_Y_fr)];
    for n=1:len_Y_fr
        Ycount(n+1)=round((Y_fr(n))/deltaf)+1;
        temp=sqrt(sum(Y(Ycount(n):Ycount(n+1)).^2));
        Y_uniband(n)=log10(temp);
    end
    testfeat(i,:)=[testdata(i,4)/360,testdata(i,5)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1)];
end
% 归一化处理
[feat_tr,ps]=mapminmax(trainfeat(:,1:4)',0,1);%按行进行归一化
trainfeat(:,1:4)=feat_tr';
feat_te=mapminmax('apply',testfeat(:,1:4)',ps);
testfeat(:,1:4)=feat_te';
fprintf('Finished!\n');
fprintf('=============================================================\n');
%% GS选择最佳的SVM参数c&g
tic
% 首先进行粗略选择: c&g 的变化范围是 2^(-10),2^(-9),...,2^(10)
[bestacc,bestc,bestg] = SVMcgForClass(trainlabel,trainfeat,-2,10,-2,10);
% 打印粗略选择结果
disp('打印粗略选择结果');
str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestacc,bestc,bestg);
disp(str);
% % 根据粗略选择的结果图再进行精细选择: c 的变化范围是 2^(-2),2^(-1.5),...,2^(4), g 的变化范围是 2^(-4),2^(-3.5),...,2^(4),
% [bestacc,bestc,bestg] = SVMcgForClass(train_wine_labels,train_wine,-2,4,-4,4,3,0.5,0.5,0.9);
% % 打印精细选择结果
% disp('打印精细选择结果');
% str = sprintf( 'Best Cross Validation Accuracy = %g%% Best c = %g Best g = %g',bestacc,bestc,bestg);
% disp(str);
disp('交叉验证及参数寻优完成！')
toc
fprintf('=============================================================\n');
%% 保存参数寻优图
path1='..\model\GS-SVM-HK\';
figname=[path1,'M_GS_hk2_wt_fir_3R_sta2_t2_sb16-c',...
    num2str(bestc),'-g',num2str(bestg),'_opt.fig'];
savefig(figname);
%% 利用最佳的参数进行SVM训练
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
c=bestc;
g=bestg;
% cmd = '-c 0.5 -g 0.5';
tic
model= libsvmtrain(trainlabel,trainfeat,cmd);
disp('训练完成！')
traintime=toc;
fprintf('训练时间为：%.4f s\n',traintime);
fprintf('=============================================================\n');
%% 利用SVM训练出的模型进行预测
tic
[predlabel,~,~]=libsvmpredict(testlabel,testfeat,model);
disp('测试完成！')
testtime=toc;
fprintf('测试时间为：%.4f s\n',testtime);
fprintf('=============================================================\n');
%% 计算统计矩阵
figure;
confusionchart(testlabel,predlabel,'RowSummary','row-normalized','ColumnSummary','column-normalized');
len_txt=16;
ACC_all=zeros(len_txt+1,7);
for i=1:len_txt+1
    if i<=len_txt
        Conmat=confusionmat(testlabel(nums(i):nums(i+1)-1),predlabel(nums(i):nums(i+1)-1));
    else
        Conmat=confusionmat(testlabel,predlabel);
    end
    if length(Conmat)>1
        ACC=zeros(1,3);
        ACC(1)=(Conmat(1,1)+Conmat(2,2))/sum(Conmat(:))*100;
        ACC(2)=Conmat(1,1)/(Conmat(1,1)+Conmat(1,2))*100;
        ACC(3)=Conmat(2,2)/(Conmat(2,1)+Conmat(2,2))*100;
        ACC_all(i,:)=[ACC(1),ACC(2),Conmat(1,1),Conmat(1,1)+Conmat(1,2),ACC(3),Conmat(2,2),Conmat(2,1)+Conmat(2,2)];
    else
        ACC_all(i,1:4)=[100,100,Conmat(1,1),Conmat(1,1)];
    end
end
%% 保存数据
filename1=[path1,'M_hk2_wt_fir_3R_sta2_t2_sb16_c',num2str(c),...
    '_g',num2str(g),'.mat'];
save(filename1,'model','Conmat','testtime','ACC_all','predlabel')
filename2=[path1,'M_hk2_wt_fir_3R_sta2_t2_sb16_c',num2str(c),...
    '_g',num2str(g),'.png'];
frame=getframe(gcf);
im=frame2im(frame);
imwrite(im,filename2);
toc(st)