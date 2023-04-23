clc;clear;close all
% 作用：将提取的TSH特征输入LSTM中进行分类识别
%% 载入数据和参数初始化；
all_params_acc=zeros(4,10);
for n=1:4
    close all;
    fprintf('Loading data...\n');
    st=tic;
    filename=['..\train_test_set\mitdb2_wt_fir_3R_RR_set',num2str(n),'.mat'];
    load(filename);
    fprintf('Finished!\n');
    fprintf('=============================================================\n');
    if n==1
        Traindata=Traindata1;
        RR_tr=RR_tr1;
        clear Traindata1 RR_tr1
    elseif n==2
        Traindata=Traindata2;
        RR_tr=RR_tr2;
        clear Traindata2 RR_tr2
    elseif n==3
        Traindata=Traindata3;
        RR_tr=RR_tr3;
        clear Traindata3 RR_tr3
    elseif n==4
        Traindata=Traindata4;
        RR_tr=RR_tr4;
        clear Traindata4 RR_tr4
    end
    %% 特征提取
    fprintf('Feature extracting and normalizing...\n');
    tic
    Fs=360;
    len_tr=size(Traindata,1);
    len_te=size(Testdata,1);
    Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];
    len_Y_fr=length(Y_fr);
    trainfeat=zeros(len_tr,4+len_Y_fr+90);
    testfeat=zeros(len_te,4+len_Y_fr+90);
    for i=1:len_tr
        zind=find(Traindata(i,:),1,'last');
        sig=Traindata(i,1:zind);
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
        Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
        trainfeat(i,:)=[RR_tr(i,1)/360,RR_tr(i,2)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1),Yharmonic_r'];
    end
    %测试集特征
    for i=1:len_te
        zind=find(Testdata(i,:),1,'last');
        sig=Testdata(i,1:zind);
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
        Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
        testfeat(i,:)=[RR_te(i,1)/360,RR_te(i,2)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1),Yharmonic_r'];
    end
    %清除多余数据
    clear  Traindata Testdata RR_tr RR_te
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
    LabTrain_name=categorical(Labeltrain(1:len_tr),[1,2],{'Normal','Abnormal'});
    %统计分类向量的种类数
    numClasses=numel(categories(LabTrain_name));
    LabTest_name=categorical(Labeltest(1:len_te),[1,2],{'Normal','Abnormal'});
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
    drop=0;
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
    
    maxEpochs=150;
    miniBatchSize=250;
    % // 如果不想展示训练过程，
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'auto',...
        'SequenceLength', 'longest',...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'InitialLearnRate', 0.001, ...
        'GradientThreshold', 1, ...
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
    ACC=zeros(1,3);
    ACC(1)=(Conmat(1,1)+Conmat(2,2))/sum(Conmat(:))*100;
    for i=1:2
        ACC(i+1)=Conmat(i,i)/sum(Conmat(i,:))*100;
    end
    all_params_acc(n,:)=[numHiddenUnits1,numHiddenUnits2,drop,maxEpochs,...
        miniBatchSize,ACC,traintime,testtime...
        ];
    path='..\model\LSTM\';
    filename1=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16_harm_stft90_',num2str(numHiddenUnits1),...
        '_',num2str(numHiddenUnits2),'_',num2str(maxEpochs),'_',num2str(miniBatchSize),...
        '_',num2str(n),'.mat'];
    save(filename1,'net','Conmat','testtime','ACC')
    filename2=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16_harm_stft90_',num2str(numHiddenUnits1),...
        '_',num2str(numHiddenUnits2),'_',num2str(maxEpochs),'_',num2str(miniBatchSize),...
        '_',num2str(n),'.png'];
    frame=getframe(gcf);
    im=frame2im(frame);
    imwrite(im,filename2);
    toc(st)%求单个数据集被训练和测试的运行时间
end
if n==4
    filename3=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16_harm_stft90_params_acc','.mat'];
    save(filename3,'all_params_acc');
end