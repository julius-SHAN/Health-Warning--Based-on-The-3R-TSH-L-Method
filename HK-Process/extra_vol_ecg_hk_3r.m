clc;clear;close all;
%作用：根据函数extra_volat_3R.m批量选取一定波动值范围内的3R样本
%% 读取数据
tic
xlsepath='..\ECG-HK\ECG-HK.xlsx';
[~,txt,~]=xlsread(xlsepath,'Sheet1');
len_txt=length(txt);
for n=1:len_txt
    matname=txt{n,1};
    extra_vol_3R(matname,70,170);
end
toc
