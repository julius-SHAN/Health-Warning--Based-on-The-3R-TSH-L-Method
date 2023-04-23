clc;clear;close all
% 作用：将提取的TSH特征输入LSTM中进行分类识别
%% 载入数据和参数初始化；
all_params_acc=zeros(1,19);
st=tic;
filename='.\TR_TE\tr_te_3r_rr_label_s_b.mat';
load(filename);
%% 特征提取
fprintf('Feature extracting and normalizing...\n');
tic
Fs=200;
len_tr=size(traindata,1);
len_te=size(testdata,1);
Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];
len_Y_fr=length(Y_fr);
trainfeat=zeros(len_tr,4+len_Y_fr+65);
testfeat=zeros(len_te,4+len_Y_fr+65);
for i=1:len_tr
    zind=find(traindata(i,:),1,'last');
    sig=traindata(i,1:zind);
    %         sig=mapminmax(sig,0,1);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Y=abs(s(start-1:end));
    deltaf=Fs/len_sig;
    Y_uniband=zeros(1,len_Y_fr);
    Ycount=[2,zeros(1,len_Y_fr)];
    for k=1:len_Y_fr
        Ycount(k+1)=round((Y_fr(k))/deltaf)+1;
        temp=sqrt(sum(Y(Ycount(k):Ycount(k+1)).^2));
        Y_uniband(k)=log10(temp);
    end
    Yharmonic_r=abs(s(start+1:start+65)./s(start));
    trainfeat(i,:)=[tr_rr(i,1)/Fs,tr_rr(i,2)/Fs,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1),Yharmonic_r'];
end
%测试集特征
for i=1:len_te
    zind=find(testdata(i,:),1,'last');
    sig=testdata(i,1:zind);
    %         sig=mapminmax(sig,0,1);
    len_sig=length(sig);
    %频域特征
    [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
    start=ceil(len_sig/2)+1;
    Y=abs(s(start-1:end));
    deltaf=Fs/len_sig;
    Y_uniband=zeros(1,len_Y_fr);
    Ycount=[2,zeros(1,len_Y_fr)];
    for k=1:len_Y_fr
        Ycount(k+1)=round((Y_fr(k))/deltaf)+1;
        temp=sqrt(sum(Y(Ycount(k):Ycount(k+1)).^2));
        Y_uniband(k)=log10(temp);
    end
    Yharmonic_r=abs(s(start+1:start+65)./s(start));
    testfeat(i,:)=[te_rr(i,1)/Fs,te_rr(i,2)/Fs,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1),Yharmonic_r'];
end
%清除多余数据
clear  Traindata Testdata tr_rr te_rr
% 归一化处理
[feat_tr,ps]=mapminmax(trainfeat(:,1:4)',0,1);%按行进行归一化
trainfeat(:,1:4)=feat_tr';
feat_te=mapminmax('apply',testfeat(:,1:4)',ps);
testfeat(:,1:4)=feat_te';
%切换数据格式
%元胞中的数组进行转置
%训练集
TrainFeature=num2cell(trainfeat,2);
for i=1:len_tr
    TrainFeature{i}=TrainFeature{i}';
end
TestFeature=num2cell(testfeat,2);
for i=1:len_te
    TestFeature{i}=TestFeature{i}';
end
%categorical分类数组可用来有效地存储并方便地处理非数值数据，同时还为数值赋予有意义的名称
LabTrain_name=categorical(trainlabel(1:len_tr),[1,2,3,4,5,6,7,8,9,10,11],...
    {'AFF','AVB','Bigeminy','EAR','IVR','Junctional','Sinus','SVT',...
    'Trigeminy','VT','Wenckebach',});
%统计分类向量的种类数
numClasses=numel(categories(LabTrain_name));
LabTest_name=categorical(testlabel(1:len_te),[1,2,3,4,5,6,7,8,9,10,11],...
    {'AFF','AVB','Bigeminy','EAR','IVR','Junctional','Sinus','SVT',...
    'Trigeminy','VT','Wenckebach',});
sequenceInput=size(TrainFeature{1},1);
%清除多余数据
clear Labeltrain Labeltest trainfeat testfeat
toc;
fprintf('Finished!\n');
fprintf('=============================================================\n');
%=========================设计LSTM网络================================
% // 创建用于sequence-to-label分类的LSTM步骤如下：
% // 1. 创建sequence input layer
% // 2. 创建若干个LSTM layer
% // 3. 创建一个fully connected layer
% // 4. 创建一个softmax layer
% // 5. 创建一个classification outputlayer
% // 注意将sequence input layer的size设置为所包含的特征类别数，本例中，1或2或3，取决于你用了几种特征。fully connected layer的参数为分类数，本例中为8.
numHiddenUnits1 = 200;
numHiddenUnits2 = 100;
drop=0.2;
layers = [ ...
    sequenceInputLayer(sequenceInput)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(drop)
    lstmLayer(numHiddenUnits2,'OutputMode','last')
    dropoutLayer(drop)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];

maxEpochs=400;
miniBatchSize=100;
% // 如果不想展示训练过程，
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto',...
    'SequenceLength', 'longest',...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',true);
% 'plots','training-progress', ...
%======================训练网络=========================
tic;
net = trainNetwork(TrainFeature,LabTrain_name,layers,options);
traintime=toc;
%======================测试网路==========================
tic
testPred = classify(net,TestFeature);
testtime=toc;
disp('测试完成！')
fprintf('测试时间为：%.4f s\n',testtime);
fprintf('=============================================================\n');
%% 计算并绘制混淆矩阵
figure;
plotconfusion(LabTest_name(1:len_te),testPred(1:len_te),'Testing Accuracy')
Conmat=confusionmat(LabTest_name(1:len_te),testPred(1:len_te));
ACC=zeros(1,12);
sum_Conmat=0;
for i=1:11
    sum_Conmat=sum_Conmat+Conmat(i,i);
end
ACC(1)=sum_Conmat/sum(Conmat(:))*100;
for i=1:11
    ACC(i+1)=Conmat(i,i)/sum(Conmat(i,:))*100;
end
all_params_acc=[numHiddenUnits1,numHiddenUnits2,drop,maxEpochs,...
    miniBatchSize,ACC,traintime,testtime...
    ];
path='.\model\LSTM\';
filename1=[path,'M_car11_fir_3R_sta2_t2_sb16_harm_stft90_s_b_',num2str(numHiddenUnits1),...
    '_',num2str(numHiddenUnits2),'_',num2str(maxEpochs),'_',num2str(miniBatchSize),...
    '.mat'];
save(filename1,'net','Conmat','testtime','ACC')
filename2=[path,'M_car11_fir_3R_sta2_t2_sb16_harm_stft90_s_b_',num2str(numHiddenUnits1),...
    '_',num2str(numHiddenUnits2),'_',num2str(maxEpochs),'_',num2str(miniBatchSize),...
    '.png'];
frame=getframe(gcf);
im=frame2im(frame);
imwrite(im,filename2);
toc(st)%求单个数据集被训练和测试的运行时间

