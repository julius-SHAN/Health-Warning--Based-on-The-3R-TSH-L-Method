clc;clear;close all;
%作用：对_wt_fir_3R心电样本进行绘图
st=tic;
filename=['..\ecg_beat_wt_fir\mitdb2_wt_fir_3R_RR','.mat'];
load(filename)
for i=1:50
    figure;
    subplot(121);
    zind=find(Nb(i,:),1,'last');
    sig1=Nb(i,1:zind);
    plot(sig1);
    zind=find(Arr(i,:),1,'last');
    sig2=Arr(i,1:zind);
    subplot(122);
    plot(sig2);
end
toc(st)

