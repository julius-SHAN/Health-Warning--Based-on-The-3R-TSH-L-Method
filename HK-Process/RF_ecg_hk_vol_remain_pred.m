clc;clear;close all;
% 作用：将ecg_hk_3r剩余未标记的样本输入RF训练的模型中，进行样本类型预测
%% 导入模型
st=tic;
path='..\model\RF-HK\';
modelname='M_hk2_wt_fir_3R_sta2_t2_sb16_T100_L1';
prefix='M_hk2_wt_fir_3R_sta2_t2_sb16';
ind=findstr(modelname,'16');
name=modelname(ind+2:end);
filename=[path,modelname,'.mat'];
load(filename);
path2='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\remain_3r_rr\';
path3=[path2,'RF',name,'\'];
mkdir(path3)
clear ACC_all Conmat testtime predlabel
%% 导入数据
xlsepath='..\ECG-HK\ECG-HK.xlsx';
[~,txt,~]=xlsread(xlsepath,'Sheet1');
len_txt=length(txt);
path1='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r\';
% 读取label，根据label求取剩余的样本
xlsepath1='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\label_1000.xlsx';
[label_1000,~,~]=xlsread(xlsepath1,'Sheet1');
s_flag=0;
Y_fr=[2,3.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5,31.5,35.5,39.5,43.5,47.5];
len_Y_fr=length(Y_fr);
pred_ecg_hk=zeros(len_txt,5);
for k=1:len_txt
    matname=txt{k,1};
    filename1=[path1,matname,'_3R_RR','.mat'];
    load(filename1);
    %% 特征提取
    zind=find(label_1000(:,k),1,'last');
    len=size(ecg_hk_3r,1);
    if len>zind
        Testdata=ecg_hk_3r(zind+1:end,:);
        clear ecg_hk_3r
        fprintf('Feature extracting and normalizing...\n');
        Fs=360;
        len_te=find(Testdata(:,1),1,'last');
        testfeat=zeros(len_te,len_Y_fr+4);
        %测试数据特征
        for i=1:len_te
            zind=find(Testdata(i,:),1,'last');
            sig=Testdata(i,7:zind);
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
            testfeat(i,:)=[Testdata(i,4)/360,Testdata(i,5)/360,kurtosis(sig),skewness(sig),mapminmax(Y_uniband,0,1)];
        end
        %特征归一化处理
        feat_te=mapminmax(testfeat(:,1:4)',0,1);
        testfeat(:,1:4)=feat_te';
        fprintf('Finished!\n');
        fprintf('=============================================================\n');
        %% 自采集数据类型预测
        % 利用RF训练出的模型进行预测
        tic
        [predlabel,scores] = predict(model, testfeat);
        testtime=toc;
        predlabel=str2num(cell2mat(predlabel));
        disp('测试完成！')
        fprintf('测试时间为：%.4f s\n',testtime);
        fprintf('=============================================================\n');
        %% 计算统计矩阵
        %fprintf('测试者：%s的心电类型预测\n',matname);
        %tabulate(predlabel)
        pred=tabulate(predlabel);
        if size(pred,1)>1
            pred_ecg_hk(k,:)=[pred(1,2)+pred(2,2),pred(1,2),pred(1,3),pred(2,2),pred(2,3)];
        else
            pred_ecg_hk(k,1:3)=[pred(1,2),pred(1,2),pred(1,3)];
        end
        %% 保存数据
        filename2=[path3,matname,'_remian_pred_16.mat'];
        save(filename2,'predlabel','pred','testtime');
        if s_flag==1
            filename3=[path2,matname,'_remian_3R_RR.mat'];
            save(filename3,'Testdata');
        end
    end
end
%% 保存数据
filename4=[path3,'ecg_hk_remian_pred_all_16.mat'];
save(filename4,'pred_ecg_hk');
toc(st)