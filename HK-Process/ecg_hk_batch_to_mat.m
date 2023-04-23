clc;clear;close all;
filepath='..\ECG-HK\';
filename=[filepath,'ECG-HK.xlsx'];
[~,txt,~]=xlsread(filename,'Sheet1');
len_txt=length(txt);
n=2;
for i=n:n
    close all;
    ecg_hk_wt_fir(txt{i,1},360,2);
end



