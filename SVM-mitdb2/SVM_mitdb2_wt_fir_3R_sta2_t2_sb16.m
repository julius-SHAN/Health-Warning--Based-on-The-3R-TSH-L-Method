clc;clear;close all
% 作用：将提取的TS特征输入SVM中进行分类识别
%% 载入数据和参数初始化；
all_params_acc=zeros(4,7);
for n=1:4
    close all;
    fprintf('Loading data...\n');
    st=tic;
    filename=['..\train_test_set\mitdb2_wt_fir_3R_RR_set',num2str(n),'.mat'];
    load(filename);
    matpath='..\model\GS-SVM\';
    matname=[matpath,'M_GS_mitdb2_wt_fir_3R_sta2_t2_sb16_params_acc','.mat'];
    load(matname);
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
    Fs=360;
    len_tr=size(Traindata,1);
    len_te=size(Testdata,1);
    Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];
    len_Y_fr=length(Y_fr);
    trainfeat=zeros(len_tr,4+len_Y_fr);
    testfeat=zeros(len_te,4+len_Y_fr);
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
        trainfeat(i,:)=[RR_tr(i,1)/360,RR_tr(i,2)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1)];
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
        testfeat(i,:)=[RR_te(i,1)/360,RR_te(i,2)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1)];
    end
    %清除多余数据
    clear  Traindata Testdata RR_tr RR_te
    % 归一化处理
    [feat_tr,ps]=mapminmax(trainfeat(:,1:4)',0,1);%按行进行归一化
    trainfeat(:,1:4)=feat_tr';
    feat_te=mapminmax('apply',testfeat(:,1:4)',ps);
    testfeat(:,1:4)=feat_te';
    %% 使用SVM训练
    c=all_gsparams_acc(n,1);
    g=all_gsparams_acc(n,2);
    cmd = ['-c ',num2str(c),' -g ',num2str(g)];
    tic
    model= libsvmtrain(Labeltrain,trainfeat,cmd);
    disp('训练完成！')
    traintime=toc;
    fprintf('训练时间为：%.4f s\n',traintime);
    fprintf('=============================================================\n');
    %% 利用SVM训练出的模型进行预测
    tic
    [ptest,acc,~]=libsvmpredict(Labeltest,testfeat,model);
    disp('测试完成！')
    testtime=toc;
    fprintf('测试时间为：%.4f s\n',testtime);
    fprintf('=============================================================\n');
    Conmat=confusionmat(Labeltest,ptest);
    figure;
    confusionchart(Labeltest,ptest,'RowSummary','row-normalized','ColumnSummary','column-normalized');
    ACC=zeros(1,3);
    ACC(1)=sum(diag(Conmat))/sum(Conmat(:))*100;
    for i=1:2
        ACC(i+1)=Conmat(i,i)/sum(Conmat(i,:))*100;
    end
    all_params_acc(n,:)=[c,g,ACC,traintime,testtime];
    path='..\model\SVM\';
    filename1=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16-c',...
        num2str(c),'-g',num2str(g),'_',num2str(n),'_1.mat'];
    save(filename1,'model','Conmat','testtime','ACC')
    filename2=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16-c',...
        num2str(c),'-g',num2str(g),'_',num2str(n),'_1.png'];
    frame=getframe(gcf);
    im=frame2im(frame);
    imwrite(im,filename2);
    toc(st)%求单个数据集被训练和测试的运行时间
end
if n==4
    filename3=[path,'M_mitdb2_wt_fir_3R_sta2_t2_sb16_params_acc_1','.mat'];
    save(filename3,'all_params_acc');
end