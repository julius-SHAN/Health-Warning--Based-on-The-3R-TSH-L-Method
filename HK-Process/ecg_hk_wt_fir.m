function   usingtime=ecg_hk_wt_fir(filename,Fs,draw)
close all;
%自定义函数：作用：对格式为TXT的心电数据进行读取，并使用500阶的48HzFIR等波纹低通滤波器来滤除噪声
%ECG-H心电数据中存在的工频干扰和肌电干扰
%函数功能：对TXT中的心电数据进行处理，并保存数据为mat
%输入参数：
%filename：即是TXT文件名，也是保存为mat文件的文件名；Fs:信号采样率
%filename 示例 ：'LQS_ECG_5kHz_2021_10_11'
%% 是否绘图
%draw: 1 绘图；0/else 不绘图
if nargin<3
    draw=0;
end
tic
%% 读取TXT文件
path=['..\ECG-HK\',filename,'.txt'];
fid=fopen(path);%通过fopen获取文件标识,并打开文件
formatSpec='%d';%指定读取格式
% Fs=360;
len_1h=2*Fs*3600;%一小时的数据量
% len_1min=1*Fs*60;%一分钟的数据量
signal=fscanf(fid,formatSpec,[len_1h,Inf] ); %读取文件数据并存为n*m矩阵
fclose(fid);%关闭文件
[hang,lie]=size(signal);

%% 处理16进制数据
ECGDATA=zeros(len_1h/2,lie);
sig=zeros(len_1h/2,lie);
for j=1:lie
    i=0;
    if mod(len_1h,2)==0%TXT数据个数为偶数
        DATA=zeros(hang/2,1);
        while(i<(hang/2))
            i=i+1;
            DATA(i)=signal(i*2,j)*256+signal(i*2-1,j);%数据处理，将两个字节数据进行合并得到ADC采样数值
            ECGDATA(i,j)=DATA(i)*(3.3/4095);
        end
    else%TXT数据个数为奇数
        DATA=zeros((hang-1)/2,1);
        while(i<((hang-1)/2))
            i=i+1;
            DATA(i)=signal(i*2,j)*256+signal(i*2-1,j);
            ECGDATA(i,j)=DATA(i)*(3.3/4095);
        end
    end
    sig1=ECGDATA(:,j);
    %% 小波阈值滤波
    [C,L]=wavedec(sig1,9,'db5');
    [len1,len2]=size(C);
    if len1>len2
        C=C';
    end
    cA9=C(1:L(1))*0;
    cD9=C(sum(L(1:1))+1:sum(L(1:2)))*0;
    cD8_1=C(sum(L(1:2))+1:sum(L(1:10)))*1;
    C1=[cA9,cD9,cD8_1];
    sig2= waverec(C1,L,'db5');
    %     figure;
    %     a2(1)=subplot(211);plot(sig);
    %     title('心电信号');
    %     a2(2)=subplot(212);plot(sig1);
    %     title('小波阈值滤波后信号波形图');
    %     linkaxes(a2,'x')

    %% 低通滤波
    %导入FDATOOL设计的FIR等波纹低通滤波器系数
    load('..\filter_params\lowpass_FIR_360Hz_48_256ord.mat')
    sig_lp=filter(Num,1,sig2);
    %     fvtool(Num,1,Fs);%绘制幅频和相频特性
    %     freqz(Num,1,N,Fs);%低通滤波器特性显示%N?
    %     Y_sig2=abs(fft(sig2,Fs)/Fs);
    %     Y_siglp=abs(fft(sig_lp,Fs)/Fs);
    %     figure;
    %     a2(1)=subplot(211);plot(Y_sig2);
    %     title('心电信号频谱图');
    %     a2(2)=subplot(212);plot(Y_siglp);
    %     title('滤波后信号频谱图');
    %     linkaxes(a2,'x')
    %     figure;
    %     a2(1)=subplot(211);plot(sig1);
    %     title('心电信号波形图');
    %     a2(2)=subplot(212);plot(sig_lp);
    %     title('滤波后信号波形图');
    %     linkaxes(a2,'x')
    sig(:,j)=sig_lp;
    %%  ECG波形绘图
    if draw==1
        figure;
        a2(1)=subplot(211);plot(sig1);
        title('心电信号波形图');
        a2(2)=subplot(212);plot(sig_lp);
        title('滤波后信号波形图');
        linkaxes(a2,'x')
    elseif draw==2
        len_ECGDATA=size(ECGDATA(:,j));
        t=1/360:1/360:len_ECGDATA/360;
        figure;
        a2(1)=subplot(211);plot(t,ECGDATA(:,j));
        hold on,set(gca,'xgrid','on','gridlinestyle','--','Gridalpha',0.4)
        hold on,set(gca,'ygrid','on','gridlinestyle','--','Gridalpha',0.4)
        hold on,grid minor;
        ylabel('电压值/V');title(['第' num2str(j),'小时ECG波形']);
        a2(2)=subplot(212);plot(t,sig(:,j));
        hold on,set(gca,'xgrid','on','gridlinestyle','--','Gridalpha',0.4)
        hold on,set(gca,'ygrid','on','gridlinestyle','--','Gridalpha',0.4)
        hold on,grid minor;
        xlabel('采样点');ylabel('电压值/V');
        title(['滤波后第' num2str(j),'小时ECG波形']);
        linkaxes(a2,'x')
    end
end
%% 保存数据
% ECGDATA=ECGDATA(:,1:16);
% sig=sig(:,1:16);
% filepath='..\ECG-HK-wt_fir\';
% path1 = [filepath,filename,'.mat'];
% save(path1,'sig');
usingtime=toc;
end

%版本记录：
%日期           作者         修改日期       修改日期
%2021-11-11     刘青山       2022-6-1       2023-2-16

