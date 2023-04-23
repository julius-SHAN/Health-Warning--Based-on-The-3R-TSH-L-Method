clc;clear;close all;
tic
% 作用：从cardiol_may_2017中提取3R心电样本和对应的标签
files = dir(fullfile('.\CARDIOL_MAY_2017\*.mat')); % 读取文件夹内的mat格式的文件
len_files = length(files); %所有文件的数量

%% 参数初始化
Fs=200;
gr=0;
traindata=zeros(len_files*22,1100);
tr_rr=zeros(len_files*22,2);
trainlabel=zeros(len_files*22,1);
trainlabel_n=cell(len_files*22,1);
testdata=zeros(len_files*10,1100);
te_rr=zeros(len_files*10,2);
testlabel=zeros(len_files*10,1);
testlabel_n=cell(len_files*10,1);
num_use=0;
ord=50;
name_ind=[];
name_all=1:328;
%% 导入单导联心电数据和求取对应的心电标签，进行噪声样本的去除
for i=1:len_files
    %导入数据
    matname=['.\CARDIOL_MAY_2017\',files(i).name];
    load(matname)
    %导入标签
    json=dir(['.\CARDIOL_MAY_2017\',files(i).name(1:end-4),'_grp?.episodes.json']);
    jsonfile = fileread(fullfile(json.folder,json.name));
    jsonstr = jsondecode(jsonfile);
    num_3r=length(jsonstr.episodes);
    if num_3r==1 && jsonstr.episodes(num_3r).rhythm_code==9999
        name_ind=[name_ind,i];
    end 
end
%% 进行训练集和测试集的划分
% 全盲实验
name_all(name_ind)=[];
len_a=length(name_all);
[num11,~,num12]=dividerand(len_a,0.9,0,0.1);
name_tr=name_all(num11);
name_te=name_all(num12);
len11=length(num11);
len12=length(num12);%训练集数目提高，测试数目30条
%% 提取训练集3R心电样本和RR间期
count=1;
for i=name_tr
    matname=['.\CARDIOL_MAY_2017\',files(i).name];
    load(matname)
    %% 对信号进行滤波
    %导入FDATOOL设计的FIR等波纹低通滤波器系数
    load('..\filter_params\lowpass_FIR_200Hz_48_100ord.mat')
%     fvtool(Num,1,Fs);%绘制幅频和相频特性
    sig=filter(Num,1,ecg);
    %     figure;
    %     a2(1)=subplot(211);plot(ecg);
    %     title('心电信号');
    %     a2(2)=subplot(212);plot(sig);
    %     title('FIR 48Hz低通滤波后信号波形图');
    %     linkaxes(a2,'x')
    sig=sig(ord+1:end);
    [~,r_ind,delay]=my_pan_tompkin(sig,Fs,gr);
    %导入标签
    json=dir(['.\CARDIOL_MAY_2017\',files(i).name(1:end-4),'_grp?.episodes.json']);
    jsonfile = fileread(fullfile(json.folder,json.name));
    jsonstr = jsondecode(jsonfile);
    if length(r_ind)>=12 && length(r_ind)<=80
        num_use=num_use+1;
        re_r_ind=[];
        %% 对R波峰值位置进行修正
        r_ind=r_ind-delay+2;
        st_ind=find(r_ind>0,1);
        end_ind=find(r_ind<=(6000-ord),1,'last');
        r_ind=r_ind(st_ind:end_ind);
        len_ind=length(r_ind);
        pre=-4;
        back=4;
        for k=1:len_ind
            st_loc=r_ind(k)+pre;
            e_loc=r_ind(k)+back;
            if st_loc>0 && e_loc<=6000-ord
                [~,loc]=max(ecg(st_loc:e_loc));
                r_ind(k)=st_loc+loc-1;
            end
        end
        %% 根据标签位置进行3R心电样本的选取
        num_3r=length(jsonstr.episodes);
        for n=1:num_3r
            if jsonstr.episodes(n).rhythm_code~=9999
                st=jsonstr.episodes(n).onset-50;%标记样本的起始位置
                et=jsonstr.episodes(n).offset-50;%标记样本的终止位置
                index=find(r_ind>=st & r_ind<=et);
                len_index=length(index);
                %thr1=Fs/(45/60);
                %thr2=floor(Fs/(150/60));
                thr1=550;
                thr2=floor(Fs/(180/60));
                for k=2:len_index-1
                    PreRR=r_ind(index(k))-r_ind(index(k-1))+1;
                    BackRR=r_ind(index(k+1))-r_ind(index(k))+1;
                    %超出心率阈值为假R波，故弃用
                    if (thr2<=PreRR&&PreRR<=thr1)&&(BackRR<=thr1&&BackRR>=thr2)
                        re_r_ind=[re_r_ind;r_ind(index(k))];
                        len_3r=r_ind(index(k+1))-r_ind(index(k-1))+1;
                        traindata(count,1:len_3r)=sig(r_ind(index(k-1)):r_ind(index(k+1)));
                        tr_rr(count,:)=[PreRR,BackRR];
                        switch jsonstr.episodes(n).rhythm_name
                            case 'AFL'
                                trainlabel(count)=1;
                                trainlabel_n{count}='AFF';
                            case 'AFIB'
                                trainlabel(count)=1;
                                trainlabel_n{count}='AFF';
                            case 'AVB_TYPE2'
                                trainlabel(count)=2;
                                trainlabel_n{count}='AVB';
                            case 'SUDDEN_BRADY'
                                trainlabel(count)=2;
                                trainlabel_n{count}='AVB';
                            case 'BIGEMINY'
                                trainlabel(count)=3;
                                trainlabel_n{count}='Bigeminy';
                            case 'EAR'
                                trainlabel(count)=4;
                                trainlabel_n{count}='EAR';
                            case 'IVR'
                                trainlabel(count)=5;
                                trainlabel_n{count}='IVR';
                            case 'JUNCTIONAL'
                                trainlabel(count)=6;
                                trainlabel_n{count}='Junctional';
                            case 'NSR'
                                trainlabel(count)=7;
                                trainlabel_n{count}='Sinus';
                            case 'SVT'
                                trainlabel(count)=8;
                                trainlabel_n{count}='SVT';
                            case 'TRIGEMINY'
                                trainlabel(count)=9;
                                trainlabel_n{count}='Trigeminy';
                            case 'VT'
                                trainlabel(count)=10;
                                trainlabel_n{count}='VT';
                            case 'WENCKEBACH'
                                trainlabel(count)=11;
                                trainlabel_n{count}='Wenckebach';
                        end
                        count=count+1;
                    end
                end
            end
        end
%         figure;
%         plot(sig);hold on
%         scatter(re_r_ind,sig(re_r_ind));hold off
%         fprintf("一次结束！\n")
%         close all;
    end
end
zind=find(trainlabel>0,1,'last');
traindata=traindata(1:zind,:);
tr_rr=tr_rr(1:zind,:);
trainlabel=trainlabel(1:zind);
trainlabel_n=trainlabel_n(1:zind);
tbl_tr1=tabulate(trainlabel);
tbl_tr2=tabulate(trainlabel_n);

%% 提取测试集3R心电样本和RR间期
count=1;
for i=name_te
    matname=['.\CARDIOL_MAY_2017\',files(i).name];
    load(matname)
    %% 对信号进行滤波
    %导入FDATOOL设计的FIR等波纹低通滤波器系数
    load('..\filter_params\lowpass_FIR_200Hz_48_100ord.mat')
    sig=filter(Num,1,ecg);
    %     figure;
    %     a2(1)=subplot(211);plot(ecg);
    %     title('心电信号');
    %     a2(2)=subplot(212);plot(sig);
    %     title('FIR 48Hz低通滤波后信号波形图');
    %     linkaxes(a2,'x')
    sig=sig(ord+1:end);
    [~,r_ind,delay]=my_pan_tompkin(sig,Fs,gr);
    %导入标签
    json=dir(['.\CARDIOL_MAY_2017\',files(i).name(1:end-4),'_grp?.episodes.json']);
    jsonfile = fileread(fullfile(json.folder,json.name));
    jsonstr = jsondecode(jsonfile);
    if length(r_ind)>=12 && length(r_ind)<=80
        num_use=num_use+1;
        re_r_ind=[];
        %% 对R波峰值位置进行修正
        r_ind=r_ind-delay+2;
        st_ind=find(r_ind>0,1);
        end_ind=find(r_ind<=(6000-ord),1,'last');
        r_ind=r_ind(st_ind:end_ind);
        len_ind=length(r_ind);
        pre=-4;
        back=4;
        for k=1:len_ind
            st_loc=r_ind(k)+pre;
            e_loc=r_ind(k)+back;
            if st_loc>0 && e_loc<=6000-ord
                [~,loc]=max(ecg(st_loc:e_loc));
                r_ind(k)=st_loc+loc-1;
            end
        end
        %% 根据标签位置进行3R心电样本的选取
        num_3r=length(jsonstr.episodes);
        for n=1:num_3r
            if jsonstr.episodes(n).rhythm_code~=9999
                st=jsonstr.episodes(n).onset-50;%标记样本的起始位置
                et=jsonstr.episodes(n).offset-50;%标记样本的终止位置
                index=find(r_ind>=st & r_ind<=et);
                len_index=length(index);
                %thr1=Fs/(45/60);
                %thr2=floor(Fs/(150/60));
                thr1=550;
                thr2=floor(Fs/(180/60));
                for k=2:len_index-1
                    PreRR=r_ind(index(k))-r_ind(index(k-1))+1;
                    BackRR=r_ind(index(k+1))-r_ind(index(k))+1;
                    %超出心率阈值为假R波，故弃用
                    if (thr2<=PreRR&&PreRR<=thr1)&&(BackRR<=thr1&&BackRR>=thr2)
                        re_r_ind=[re_r_ind;r_ind(index(k))];
                        len_3r=r_ind(index(k+1))-r_ind(index(k-1))+1;
                        testdata(count,1:len_3r)=sig(r_ind(index(k-1)):r_ind(index(k+1)));
                        te_rr(count,:)=[PreRR,BackRR];
                        switch jsonstr.episodes(n).rhythm_name
                            case 'AFL'
                                testlabel(count)=1;
                                testlabel_n{count}='AFF';
                            case 'AFIB'
                                testlabel(count)=1;
                                testlabel_n{count}='AFF';
                            case 'AVB_TYPE2'
                                testlabel(count)=2;
                                testlabel_n{count}='AVB';
                            case 'SUDDEN_BRADY'
                                testlabel(count)=2;
                                testlabel_n{count}='AVB';
                            case 'BIGEMINY'
                                testlabel(count)=3;
                                testlabel_n{count}='Bigeminy';
                            case 'EAR'
                                testlabel(count)=4;
                                testlabel_n{count}='EAR';
                            case 'IVR'
                                testlabel(count)=5;
                                testlabel_n{count}='IVR';
                            case 'JUNCTIONAL'
                                testlabel(count)=6;
                                testlabel_n{count}='Junctional';
                            case 'NSR'
                                testlabel(count)=7;
                                testlabel_n{count}='Sinus';
                            case 'SVT'
                                testlabel(count)=8;
                                testlabel_n{count}='SVT';
                            case 'TRIGEMINY'
                                testlabel(count)=9;
                                testlabel_n{count}='Trigeminy';
                            case 'VT'
                                testlabel(count)=10;
                                testlabel_n{count}='VT';
                            case 'WENCKEBACH'
                                testlabel(count)=11;
                                testlabel_n{count}='Wenckebach';
                        end
                        count=count+1;
                    end
                end
            end
        end
%         figure;
%         plot(sig);hold on
%         scatter(re_r_ind,sig(re_r_ind));hold off
%         fprintf("一次结束！\n")
%         close all;
    end
end
zind=find(testlabel>0,1,'last');
testdata=testdata(1:zind,:);
te_rr=te_rr(1:zind,:);
testlabel=testlabel(1:zind);
testlabel_n=testlabel_n(1:zind);
tbl_te1=tabulate(testlabel);
tbl_te2=tabulate(testlabel_n);

path='.\TR_TE\';
filename=[path,'tr_te_3r_rr_label_blind.mat'];
save(filename,"traindata","tr_rr",'trainlabel','trainlabel_n',...
    'testdata','te_rr','testlabel','testlabel_n')
toc
