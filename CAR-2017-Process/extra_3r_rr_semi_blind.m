clc;clear;close all;
tic
% 作用：从cardiol_may_2017中提取3R心电样本和对应的标签
files = dir(fullfile('.\CARDIOL_MAY_2017\*.mat')); % 读取文件夹内的mat格式的文件
len_files = length(files); %所有文件的数量

%% 参数初始化
Fs=200;
gr=0;
car_3r=zeros(len_files*30,1100);
car_rr=zeros(len_files*30,2);
car_label=zeros(len_files*30,1);
car_label_name=cell(len_files*30,1);
count=1;
num_use=0;
ord=50;
%% 导入单导联心电数据和求取对应的心电标签，去除噪声样本
for i=1:len_files
    matname=['.\CARDIOL_MAY_2017\',files(i).name];
    load(matname)
    %% 对信号进行滤波
    %导入FDATOOL设计的FIR等波纹低通滤波器系数
    load('..\filter_params\lowpass_FIR_200Hz_48_100ord.mat')
    sig=filter(Num,1,ecg);
    %     fvtool(Num,1,Fs);%绘制幅频和相频特性
    %     figure;
    %     a2(1)=subplot(211);plot(ecg);
    %     title('心电信号');
    %     a2(2)=subplot(212);plot(sig);
    %     title('小波阈值滤波后信号波形图');
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
                        car_3r(count,1:len_3r)=sig(r_ind(index(k-1)):r_ind(index(k+1)));
                        car_rr(count,:)=[PreRR,BackRR];
                        switch jsonstr.episodes(n).rhythm_name
                            case 'AFL'
                                car_label(count)=1;
                                car_label_name{count}='AFF';
                            case 'AFIB'
                                car_label(count)=1;
                                car_label_name{count}='AFF';
                            case 'AVB_TYPE2'
                                car_label(count)=2;
                                car_label_name{count}='AVB';
                            case 'SUDDEN_BRADY'
                                car_label(count)=2;
                                car_label_name{count}='AVB';
                            case 'BIGEMINY'
                                car_label(count)=3;
                                car_label_name{count}='Bigeminy';
                            case 'EAR'
                                car_label(count)=4;
                                car_label_name{count}='EAR';
                            case 'IVR'
                                car_label(count)=5;
                                car_label_name{count}='IVR';
                            case 'JUNCTIONAL'
                                car_label(count)=6;
                                car_label_name{count}='Junctional';
                            case 'NSR'
                                car_label(count)=7;
                                car_label_name{count}='Sinus';
                            case 'SVT'
                                car_label(count)=8;
                                car_label_name{count}='SVT';
                            case 'TRIGEMINY'
                                car_label(count)=9;
                                car_label_name{count}='Trigeminy';
                            case 'VT'
                                car_label(count)=10;
                                car_label_name{count}='VT';
                            case 'WENCKEBACH'
                                car_label(count)=11;
                                car_label_name{count}='Wenckebach';
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
zind=find(car_label>0,1,'last');
car_3r=car_3r(1:zind,:);
car_rr=car_rr(1:zind,:);
car_label=car_label(1:zind);
car_label_name=car_label_name(1:zind);
tbl1=tabulate(car_label);
tbl2=tabulate(car_label_name);
%% 训练集和测试集的划分
len_l=length(car_label);
[num11,~,num12]=dividerand(len_l,0.9,0,0.1);
traindata=car_3r(num11,:);
trainlabel=car_label(num11,:);
trainlabel_n=car_label_name(num11,:);
tr_rr=car_rr(num11,:);
testdata=car_3r(num12,:);
testlabel=car_label(num12,:);
testlabel_n=car_label_name(num12,:);
te_rr=car_rr(num12,:);
path='.\TR_TE\';
filename=[path,'tr_te_3r_rr_label_s_b.mat'];
save(filename,"traindata","tr_rr",'trainlabel','trainlabel_n',...
    'testdata','te_rr','testlabel','testlabel_n')
toc
