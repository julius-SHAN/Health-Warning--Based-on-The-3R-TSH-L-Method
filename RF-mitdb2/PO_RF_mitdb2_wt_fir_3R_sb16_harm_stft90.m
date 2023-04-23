clc;clear;close all
% 作用：将提取的SH特征输入RF中进行参数寻优和分类识别
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
    %% 使用少量数据进行SVM的参数
    len=2000;
    len_tr=size(Traindata,1);
    len_te=size(Testdata,1);
    num1=randperm(len_tr);
    num2=randperm(len_te);
    Traindata=Traindata(num1(1:len),:);
    Labeltrain=Labeltrain(num1(1:len));
    Testdata=Testdata(num2(1:len),:);
    Labeltest=Labeltest(num2(1:len));
    %% 特征提取
    fprintf('Feature extracting and normalizing...\n');
    Fs=360;
    Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];%mitdb2intra数据集的子带谱统计
    len_Y_fr=length(Y_fr);
    trainfeat=zeros(len,len_Y_fr+90);
    testfeat=zeros(len,len_Y_fr+90);
    for i=1:len
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
    for i=1:len
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
    filename1=[path,'M_PO_mitdb2_wt_fir_3R_sb16_harm_stft90_',num2str(n),',opt.mat'];
    save(filename1,'ACC')
    pngname=[path,'M_PO_mitdb2_wt_fir_3R_sb16_harm_stft90_',num2str(n),',opt.png'];
    frame=getframe(gcf);
    im=frame2im(frame);
    imwrite(im,pngname);
    toc(st);
end