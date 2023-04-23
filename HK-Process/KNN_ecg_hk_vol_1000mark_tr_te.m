clc;clear;close all;
% 作用：将根据label_1000生成的训练集和测试集输入进KNN中，并提取TH特征进行分类识别
st=tic;
%% 导入数据
path='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\ecg_hk_tr_te\';
filename=[path,'label_1000_16_tr_te.mat'];
load(filename);
%% 特征提取
fprintf('Feature extracting and normalizing...\n');
Fs=360;
len_tr=size(traindata,1);
len_te=size(testdata,1);
trainfeat=zeros(len_tr,4+90);
testfeat=zeros(len_te,4+90);
for i=1:len_tr
    zind=find(traindata(i,:),1,'last');
    sig=traindata(i,7:zind);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
    trainfeat(i,:)=[traindata(i,4)/360,traindata(i,5)/360,kurtosis(sig),skewness(sig),Yharmonic_r'];
end
%测试集特征
for i=1:len_te
    zind=find(testdata(i,:),1,'last');
    sig=testdata(i,7:zind);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
    testfeat(i,:)=[testdata(i,4)/360,testdata(i,5)/360,kurtosis(sig),skewness(sig),Yharmonic_r'];
end
% 归一化处理
[feat_tr,ps]=mapminmax(trainfeat(:,1:4)',0,1);%按行进行归一化
trainfeat(:,1:4)=feat_tr';
feat_te=mapminmax('apply',testfeat(:,1:4)',ps);
testfeat(:,1:4)=feat_te';
fprintf('Finished!\n');
fprintf('=============================================================\n');
%% 使用KNN进行训练
tic
%     model = fitcknn(trainfeat,Labeltrain,'NumNeighbors',10,'Distance','cosine');
rng(1)
model = fitcknn(trainfeat,trainlabel,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
traintime=toc;
k=model.NumNeighbors;
d=model.Distance;
disp('训练完成！')
fprintf('训练时间为：%.4f s\n',traintime);
fprintf('=============================================================\n');
%% 利用KNN训练出的模型进行预测
tic
predlabel = predict(model,testfeat);
testtime=toc;
disp('测试完成！')
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
path='..\model\KNN-HK\';
filename1=[path,'M_hk2_wt_fir_3R_sta2_t2_harm_stft90_K',num2str(k),...
    '_D',d,'.mat'];
save(filename1,'model','Conmat','testtime','ACC_all','predlabel')
filename2=[path,'M_hk2_wt_fir_3R_sta2_t2_harm_stft90_K',num2str(k),...
    '_D',d,'.png'];
frame=getframe(gcf);
im=frame2im(frame);
imwrite(im,filename2);
toc(st)