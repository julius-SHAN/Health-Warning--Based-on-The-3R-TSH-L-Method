% 作用：将提取的TH特征输入RF中进行参数寻优和分类识别
%% 载入数据;
for n=1:4
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
    %% 使用少量数据进行SVM的参数
    len=2000;
    len_tr=size(Traindata,1);
    len_te=size(Testdata,1);
    num1=randperm(len_tr);
    num2=randperm(len_te);
    Traindata=Traindata(num1(1:len),:);
    Labeltrain=Labeltrain(num1(1:len));
    RR_tr=RR_tr(num1(1:len),:);
    Testdata=Testdata(num2(1:len),:);
    Labeltest=Labeltest(num2(1:len));
    RR_te=RR_te(num2(1:len),:);

    %% 特征提取
    fprintf('Feature extracting and normalizing...\n');
    Fs=360;
    trainfeat=zeros(len,4+90);
    testfeat=zeros(len,4+90);
    for i=1:len
        zind=find(Traindata(i,:),1,'last');
        sig=Traindata(i,1:zind);
%         sig=mapminmax(sig,0,1);
        len_sig=length(sig);
        %频域特征
        [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
        start=ceil(len_sig/2)+1;
        Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
        trainfeat(i,:)=[RR_tr(i,1)/360,RR_tr(i,2)/360,kurtosis(sig),skewness(sig),Yharmonic_r'];
    end
    %测试集特征
    for i=1:len
        zind=find(Testdata(i,:),1,'last');
        sig=Testdata(i,1:zind);
%         sig=mapminmax(sig,0,1);
        len_sig=length(sig);
        %频域特征
        [s,f]=stft(sig, Fs,'Window' ,kaiser(len_sig,5), 'OverlapLength' ,len_sig-1, 'FFTLength' ,len_sig);
        start=ceil(len_sig/2)+1;
        Yharmonic_r=abs(s(start+1:start+89+1)./s(start));
        testfeat(i,:)=[RR_te(i,1)/360,RR_te(i,2)/360,kurtosis(sig),skewness(sig),Yharmonic_r'];
    end
    %清除多余数据
    clear  Traindata Testdata RR_tr RR_te
    % 归一化处理
    [feat_tr,ps]=mapminmax(trainfeat(:,1:4)',0,1);%按行进行归一化
    trainfeat(:,1:4)=feat_tr';
    feat_te=mapminmax('apply',testfeat(:,1:4)',ps);
    testfeat(:,1:4)=feat_te';
    fprintf('Finished!\n');
    fprintf('=============================================================\n');
    %% 使用RF进行训练
    % 参数优化
    RFLeaf=[1,2,3,4,5,10,20,50,100,200,500];
    len_L=length(RFLeaf);
    ACC=zeros(len_L,500);
    figure('Name','RF Leaves and Trees');
    for i=1:len_L
        RFModel=TreeBagger(500,trainfeat,Labeltrain,'Method','classification','OOBPrediction','On','MinLeafSize',RFLeaf(i));
        ACC(i,:)=1-oobError(RFModel);
        plot(1-oobError(RFModel));
        hold on
    end
    [M,I]=max(ACC');
    xlabel('Number of Grown Trees');
    ylabel('ACC') ;
    title('Recognition rate of different number of leaves and trees');
    LeafTreelgd=legend({'1' '2' '3' '4' '5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
    title(LeafTreelgd,'Number of Leaves');
    hold off;
    path='..\model\PO-RF\';
    filename1=[path,'M_PO_mitdb2_wt_fir_3R_sta2_t2_harm_stft90_',num2str(n),',opt.mat'];
    save(filename1,'ACC')
    pngname=[path,'M_PO_mitdb2_wt_fir_3R_sta2_t2_harm_stft90_',num2str(n),',opt.png'];
    frame=getframe(gcf);
    im=frame2im(frame);
    imwrite(im,pngname);
    toc(st);
end
