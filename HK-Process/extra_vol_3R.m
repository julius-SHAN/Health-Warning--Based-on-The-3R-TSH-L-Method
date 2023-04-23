function   usingtime=extra_vol_3R(matname,thr1,thr2)
close all;
%% 作用:提取一定波动值范围内的3R样本
tic
% 导入数据
% matname='WJ_ECG_360Hz_2022_6_7';
filepath='..\ecg_hk_3r_wt_fir\';
filename=[filepath,matname,'_3R_RR.mat'];
load(filename);
volat=ecg_hk_3r(:,6);
len_volat=length(volat);
vol=zeros(ceil(len_volat*0.9),1);
count=0;
for i=1:len_volat
    if volat(i)>=thr1&&volat(i)<=thr2
        count=count+1;
        vol(count)=i;
    end
end
zind=find(vol,1,'last');
vol=vol(1:zind);
ecg_hk_3r=ecg_hk_3r(vol,:);
%% 保存数据
path='..\ecg_hk_3r_wt_fir\ecg_hk_vol_3r\';
filename1=[path,matname,'_3R_RR.mat'];
save(filename1,'ecg_hk_3r');
usingtime=toc;
end

