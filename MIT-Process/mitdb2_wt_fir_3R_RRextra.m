clc;clear;close all;
%作用：对小波阈值滤波和低通滤波后的MIT-BIH信号进行正常与异常类型的3R心电样本提取
%并保留相应的RR间期
st=tic;
Name_whole=[100,101,103,105,106,107,108,109,111,112,...
    113,114,115,116,117,118,119,121,122,123,...
    124,200,201,202,203,205,207,208,209,210,...
    212,213,214,215,217,219,220,221,222,223,...
    228,230,231,232,233,234];
% 102和104不要，而114选第二行
Nb=zeros(73000,1000);%正常搏动
Arr=zeros(29000,1000);
NbRRfeat=zeros(73000,2);
ArrRRfeat=zeros(29000,2);
Fs=360;
delay=128;%低通滤波群延迟
count1=0;
count2=0;
for na=1:46
    tic
    if na==12
        NLII=2;
    else
        NLII=1;
    end
    Name=num2str(Name_whole(na));
    PATH= '..\MIT-BIH\'; % 指定数据的储存路径
    HEADERFILE=strcat(Name, '.hea');% .hea 格式，头文件，可用记事本打开
    ATRFILE=strcat(Name, '.atr'); % .atr 格式，属性文件，数据格式为二进制数
    DATAFILE=strcat(Name, '.dat');% .dat 格式，ECG 数据
    SAMPLES2READ=650000;          % 指定需要读入的样本数
    %------ LOAD HEADER DATA --------------------------------------------------
    %------ 读入头文件数据 -----------------------------------------------------
    % 示例：用记事本打开的117.hea 文件的数据
    %
    %      117 2 360 650000
    %      117.dat 212 200 11 1024 839 31170 0 MLII
    %      117.dat 212 200 11 1024 930 28083 0 V2
    %      # 69 M 950 654 x2
    %      # None
    %-------------------------------------------------------------------------
    signalh= fullfile(PATH, HEADERFILE);    % 通过函数 fullfile 获得头文件的完整路径
    fid1=fopen(signalh,'r');    % 打开头文件，其标识符为 fid1 ，属性为'r'--“只读”
    z= fgetl(fid1);             % 读取头文件的第一行数据，字符串格式
    A= sscanf(z, '%*s %d %d %d',[1,3]); % 按照格式 '%*s %d %d %d' 转换数据并存入矩阵 A 中
    nosig= A(1);    % 信号通道数目
    sfreq=A(2);     % 数据采样频率
    clear A;        % 清空矩阵 A ，准备获取下一行数据
    for k=1:nosig           % 读取每个通道信号的数据信息
        z= fgetl(fid1);
        A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
        dformat(k)= A(1);           % 信号格式; 这里只允许为 212 格式
        gain(k)= A(2);              % 每 mV 包含的整数个数
        bitres(k)= A(3);            % 采样精度（位分辨率）
        zerovalue(k)= A(4);         % ECG 信号零点相应的整数值
        firstvalue(k)= A(5);        % 信号的第一个整数值 (用于偏差测试)
    end
    fclose(fid1);
    clear A;
    %------ LOAD BINARY DATA --------------------------------------------------
    %------ 读取 ECG 信号二值数据 ----------------------------------------------
    %dformat=[212,212];
    if dformat~= [212,212], error('this script does not apply binary formats different to 212.'); end
    signald= fullfile(PATH, DATAFILE);            % 读入 212 格式的 ECG 信号数据
    fid2=fopen(signald,'r');
    A= fread(fid2, [3, SAMPLES2READ], 'uint8')';  % matrix with 3 rows, each 8 bits long, = 2*12bit
    fclose(fid2);
    % 通过一系列的移位（bitshift）、位与（bitand）运算，将信号由二值数据转换为十进制数
    M2H= bitshift(A(:,2), -4);        %字节向右移四位，即取字节的高四位
    M1H= bitand(A(:,2), 15);          %取字节的低四位
    PRL=bitshift(bitand(A(:,2),8),9);     % sign-bit   取出字节低四位中最高位，向右移九位
    PRR=bitshift(bitand(A(:,2),128),5);   % sign-bit   取出字节高四位中最高位，向右移五位
    M( : , 1)= bitshift(M1H,8)+ A(:,1)-PRL;
    M( : , 2)= bitshift(M2H,8)+ A(:,3)-PRR;
    if M(1,:) ~= firstvalue, error('inconsistency in the first bit values'); end
    switch nosig
        case 2
            M( : , 1)= (M( : , 1)- zerovalue(1))/gain(1);
            M( : , 2)= (M( : , 2)- zerovalue(2))/gain(2);
            TIME=(0:(SAMPLES2READ-1))/sfreq;
        case 1
            M( : , 1)= (M( : , 1)- zerovalue(1));
            M( : , 2)= (M( : , 2)- zerovalue(1));
            M=M';
            M(1)=[];
            sM=size(M);
            sM=sM(2)+1;
            M(sM)=0;
            M=M';
            M=M/gain(1);
            TIME=(0:2*(SAMPLES2READ)-1)/sfreq;
        otherwise  % this case did not appear up to now!
            % here M has to be sorted!!!
            disp('Sorting algorithm for more than 2 signals not programmed yet!');
    end
    clear A M1H M2H PRR PRL;
    %------ LOAD ATTRIBUTES DATA ----------------------------------------------
    atrd= fullfile(PATH, ATRFILE);      % attribute file with annotation data
    fid3=fopen(atrd,'r');
    A= fread(fid3, [2, inf], 'uint8')';
    fclose(fid3);
    ATRTIME=[];
    ANNOT=[];
    sa=size(A);
    saa=sa(1);
    i=1;
    while i<=saa
        annoth=bitshift(A(i,2),-2);
        if annoth==59
            ANNOT=[ANNOT;bitshift(A(i+3,2),-2)];
            ATRTIME=[ATRTIME;A(i+2,1)+bitshift(A(i+2,2),8)+...
                bitshift(A(i+1,1),16)+bitshift(A(i+1,2),24)];
            i=i+3;
        elseif annoth==60
            % nothing to do!
        elseif annoth==61
            % nothing to do!
        elseif annoth==62
            % nothing to do!
        elseif annoth==63
            hilfe=bitshift(bitand(A(i,2),3),8)+A(i,1);
            hilfe=hilfe+mod(hilfe,2);
            i=i+hilfe/2;
        else
            ATRTIME=[ATRTIME;bitshift(bitand(A(i,2),3),8)+A(i,1)];
            ANNOT=[ANNOT;bitshift(A(i,2),-2)];
        end
        i=i+1;
    end
    ANNOT(length(ANNOT))=[];       % last line = EOF (=0)
    ATRTIME(length(ATRTIME))=[];   % last line = EOF
    clear A;
    ATRTIME= (cumsum(ATRTIME))/sfreq;
    ind= find(ATRTIME <= TIME(end));
    ATRTIMED= ATRTIME(ind);
    len3=length(ATRTIMED);
    ANNOT=round(ANNOT);
    ANNOTD= ANNOT(ind);
    fprintf('%s数据读取完毕！\n',Name);
    toc
    %% 数据处理
    tic
    % M是2维信号
    sig=M(:,NLII)';%不同导联
    clear M
    %% 小波阈值滤波
    [C,L]=wavedec(sig,9,'db5');
    [len1,len2]=size(C);
    if len1>len2
        C=C';
    end
    cA9=C(1:L(1))*0;
    cD9=C(sum(L(1:1))+1:sum(L(1:2)))*0;
    cD8_1=C(sum(L(1:2))+1:sum(L(1:10)))*1;
    C1=[cA9,cD9,cD8_1];
    sig1= waverec(C1,L,'db5');
    %     figure;
    %     a2(1)=subplot(211);plot(sig);
    %     title(Name,'心电信号');
    %     a2(2)=subplot(212);plot(sig1);
    %     title('小波阈值滤波后信号波形图');
    %     linkaxes(a2,'x')

    %% 低通滤波
    %导入FDATOOL设计的FIR等波纹低通滤波器系数
    load('..\filter_params\lowpass_FIR_360Hz_58_256ord.mat')
    sig_lp=filter(Num,1,sig1);
    %     fvtool(Num,1,Fs);%绘制幅频和相频特性
    %     figure;
    %     a2(1)=subplot(211);plot(sig);
    %     title(Name,'心电信号波形图');
    %     a2(2)=subplot(212);plot(sig_lp);
    %     title('滤波后信号波形图');
    %     linkaxes(a2,'x')

    %% 精确定位R波位置
    R_TIME=ATRTIMED(ANNOTD==1 | ANNOTD==2 | ANNOTD==3 | ANNOTD==4 | ANNOTD==5 | ANNOTD==6 | ANNOTD==7 ...
        | ANNOTD==8 | ANNOTD==9 | ANNOTD==10 | ANNOTD==11 | ANNOTD==12 | ANNOTD==31 | ANNOTD==38);
    REF_ind=round(R_TIME'.*360);%不同种类心电信号的R波标记
    REF_ind=REF_ind+delay;
    ann=ANNOTD(ANNOTD==1 | ANNOTD==2 | ANNOTD==3 | ANNOTD==4 | ANNOTD==5 | ANNOTD==6 | ANNOTD==7 ...
        | ANNOTD==8 | ANNOTD==9 | ANNOTD==10 | ANNOTD==11 | ANNOTD==12 | ANNOTD==31 | ANNOTD==38);%心电信号种类标记
    out_ind=find(REF_ind>length(sig),1);
    REF_ind(out_ind)=[];
    ann(out_ind)=[];
    nums_REF_ind=length(REF_ind);
    for i=2:nums_REF_ind-1
        PreRR=REF_ind(i)-REF_ind(i-1)+1;
        BackRR=REF_ind(i+1)-REF_ind(i)+1;
        thr1=Fs/(45/60);
        thr2=floor(Fs/(150/60));
        if (thr2<=PreRR&&PreRR<=thr1)&&(BackRR<=thr1&&BackRR>=thr2)
            SEG=sig_lp( REF_ind(i-1):REF_ind(i+1));%选取心拍
            %% 提取时域特征
            if ann(i)==1
                count1=count1+1;
                len_SEG=size(SEG,2);
                Nb(count1,1:len_SEG)=SEG;%行添加
                NbRRfeat(count1,:)=[PreRR,BackRR];
            else
                count2=count2+1;
                len_SEG=size(SEG,2);
                Arr(count2,1:len_SEG)=SEG;%行添加
                ArrRRfeat(count2,:)=[PreRR,BackRR];
            end
        end
    end
    fprintf('%s数据处理完毕！\n',Name);
    toc
end
zind=find(NbRRfeat(:,1),1,'last');
len_NbRR=length(NbRRfeat(:,1));
if zind<len_NbRR
    Nb=Nb(1:zind,:);
    NbRRfeat=NbRRfeat(1:zind,:);
end
zind=find(ArrRRfeat(:,1),1,'last');
len_ArrRR=length(ArrRRfeat(:,1));
if zind<len_ArrRR
    Arr=Arr(1:zind,:);
    ArrRRfeat=ArrRRfeat(1:zind,:);
end
path='..\ecg_beat_wt_fir\';
filename=[path,'mitdb2_wt_fir_3R_RR','.mat'];
save(filename,'Nb','Arr','NbRRfeat','ArrRRfeat')
toc(st)

