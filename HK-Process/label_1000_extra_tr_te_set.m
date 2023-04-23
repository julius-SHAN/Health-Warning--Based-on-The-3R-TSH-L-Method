clc;clear;close all;
% 作用：根据label_1000将vol_3R_RR划分出训练集和测试集
%% 导入数据
tic
path='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\';
xlsepath=[path,'ECG-HK_predict_mark','.xlsx'];
[~,txt,~]=xlsread(xlsepath,'Sheet1');
len_txt=length(txt);
% len_txt=13;
xlsepath1=[path,'label_1000','.xlsx'];
[num,~,~]=xlsread(xlsepath1,'Sheet1');
path1='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r\';
traindata=[];
testdata=[];
trainlabel=[];
testlabel=[];
nums=ones(len_txt+1,1);
nums_1=zeros(len_txt,1);
nums_2=zeros(len_txt,1);
for k=1:len_txt
    matname=txt{k};
    filename=[path1,matname,'_3R_RR','.mat'];
    load(filename);
    zind=find(num(:,k),1,'last');
    label=num(1:zind,k); 
    len_label=length(label);
    [num1,~,num2]=dividerand(len_label,0.7,0,0.3);
    nums(k+1)=nums(k)+length(num2);%叠加测试集数目
    nums_1(k)=length(num1);
    nums_2(k)=length(num2);
    %% 分割数据
    traindata=[traindata;ecg_hk_3r(num1,:)];
    testdata=[testdata;ecg_hk_3r(num2,:)];
    trainlabel=[trainlabel;label(num1)];
    testlabel=[testlabel;label(num2)];
    clear ecg_hk_3r label
end
path2='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r_pred\ecg_hk_tr_te\';
filename1=[path2,'label_1000_16_tr_te.mat'];
save(filename1,'traindata','testdata','trainlabel','testlabel','len_txt','nums');
toc