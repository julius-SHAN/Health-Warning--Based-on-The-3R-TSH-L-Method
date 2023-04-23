clc;clear;close all
% 作用：将提取的SH特征输入KNN中进行分类识别
%% 载入数据和参数初始化；
all_params_acc=cell(4,7);
for n=1:4
    fprintf('Loading data...\n');
    st=tic;
    filename=['..\train_test_set\mitdb2_wt_fir_3R_RR_set',num2str(n),'.mat'];
    load(filename);
    fprintf('Finished!\n');
    fprintf('=============================================================\n');
    if n==1
        Traindata=Traindata1;
        clear Traindata1 RR_tr1
    elseif n==2
        Traindata=Traindata2;
        clear Traindata2 RR_tr2
    elseif n==3
        Traindata=Traindata3;
        clear Traindata3 RR_tr3
    elseif n==4
        Traindata=Traindata4;
        clear Traindata4 RR_tr4
    end
    %% 特征提取
    fprintf('Feature extracting and normalizing...\n');
    tic
    Fs=360;
    len_tr=size(Traindata,1);
    len_te=size(Testdata,1);
    Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];%mitdb2intra数据集的子带谱统计
    len_Y_fr=length(Y_fr);
    trainfeat=zeros(len_tr,len_Y_fr+90);
    testfeat=zeros(len_te,len_Y_fr+90);
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
        trainfeat(i,:)=[mapminmax(Y_uniband,0,1),Yharmonic_r'];
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
        testfeat(i,:)=[mapminmax(Y_uniband,0,1),Yharmonic_r'];
    end
    %清除多余数据
    clear  Traindata Testdata RR_tr RR_te
    % 归一化处理
    %     [feat_tr,ps]=mapminmax(trainfeat(:,1:2)',0,1);
    %     trainfeat(:,1:2)=feat_tr';
    %     feat_te=mapminmax('apply',testfeat(:,1:2)',ps);
    %     testfeat(:,1:2)=feat_te';
    fprintf('Finished!\n');
    fprintf('=============================================================\n');
    %% 使用KNN进行训练
    tic
%     model = fitcknn(trainfeat,Labeltrain,'NumNeighbors',10,'Distance','cosine');
    rng(1)
    model = fitcknn(trainfeat,Labeltrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
    traintime=toc;
    disp('训练完成！')
    fprintf('训练时间为：%.4f s\n',traintime);
    fprintf('=============================================================\n');
    %% 利用KNN训练出的模型进行预测
    tic
    ptest = predict(model,testfeat);
    testtime=toc;
    disp('测试完成！')
    fprintf('测试时间为：%.4f s\n',testtime);
    fprintf('=============================================================\n');
    Conmat=confusionmat(Labeltest,ptest);
    k=model.NumNeighbors;
    d=model.Distance;
    figure;
    confusionchart(Labeltest,ptest,'RowSummary','row-normalized','ColumnSummary','column-normalized');
    ACC=zeros(1,3);
    ACC(1)=sum(diag(Conmat))/sum(Conmat(:))*100;
    for i=1:2
        ACC(i+1)=Conmat(i,i)/sum(Conmat(i,:))*100;
    end
    all_params_acc{n,1}=k;
    all_params_acc{n,2}=d;
    all_params_acc{n,3}=ACC(1);
    all_params_acc{n,4}=ACC(2);
    all_params_acc{n,5}=ACC(3);
    all_params_acc{n,6}=traintime;
    all_params_acc{n,7}=testtime;
    path='..\model\KNN\';
    filename1=[path,'M_mitdb2_wt_fir_3R_sb16_harm_stft90_K',num2str(k),...
        '_D',d,'_',num2str(n),'.mat'];
    save(filename1,'model','Conmat','testtime','ACC')
    filename2=[path,'M_mitdb2_wt_fir_3R_sb16_harm_stft90_K',num2str(k),...
        '_D',d,'_',num2str(n),'.png'];
    frame=getframe(gcf);
    im=frame2im(frame);
    imwrite(im,filename2);
    toc(st)%求单个数据集被训练和测试的运行时间
end
if n==4
    filename3=[path,'M_mitdb2_wt_fir_3R_sb16_harm_stft90_params_acc','.mat'];
    save(filename3,'all_params_acc');
end