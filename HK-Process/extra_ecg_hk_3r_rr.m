clc;clear;close all;
%作用：对滤波后的心电数据进行失真信号筛选，并利用QRS波定位算法提取3R心电样本和RR间期
tic
%% 读取数据
filepath='..\ECG-HK-wt_fir\';
xlsepath='..\ECG-HK\ECG-HK.xlsx';
[param,txt,~]=xlsread(xlsepath,'Sheet1');
len_txt=length(txt);
flag_s=1;
for n=1:len_txt
    close all
    matname=txt{n,1};
    path=[filepath,matname,'.mat'];
    load(path)
    %% 参数初始化
    [hang,lie]=size(sig);
    %     lie=1;
    signal=zeros(hang,lie);
    Fs=360;
    gr=0;
    T=1;
    len=3000*lie;
    ecg_hk_3r=zeros(len,1000);
    count=0;
    thr11=1.4;
    thr21=-1.1;
    zero=0;
    thr_n2=zeros(lie,2);
    for i=param(n,1):param(n,2)
        %%  根据阈值去除失真数据
        zind=find(sig(:,i),1,'last');
        s=sig(1:zind,i);
        %初双门限筛选信号
        [row1,~,~]=find(s>=thr11);
        [row2,~,~]=find(s<=thr21);
        row=[row1;row2];
        row=sort(row,'ascend');
        len_row=length(row);
        for j=2:len_row
            if row(j)-row(j-1)<=Fs*T
                len=row(j)-row(j-1)+1;
                s(row(j-1):row(j))=zero;
            end
        end
        %         figure;
        %         a2(1)=subplot(211);
        %         plot(sig(:,i));
        %         a2(2)=subplot(212);
        %         plot(s);hold on
        %         linkaxes(a2,'x')
        [~,index,delay]=my_pan_tompkin(s,Fs,gr);
        index=index+delay;
        mean_r=mean(s(index));
        if mean_r>0.15
            mean_r=0.15;
        end
        % 对R波峰值位置进行修正
        len_ind=length(index);
        pre=-4;
        back=4;
        ind=[];
        for k=1:len_ind
            st_loc=index(k)+pre;
            e_loc=index(k)+back;
            if st_loc>=0 && e_loc<=zind
                [m,loc]=max(s(st_loc:e_loc));
                if m<mean_r
                    ind=[ind;k];
                end
                index(k)=st_loc+loc-1;
            end
        end
        index(ind)=[];
        %         figure;
        %         plot(s);hold on;
        %         scatter(index,s(index));
        %         hold off;
        %第二次门限筛选
        mean_r=mean(s(index));
        max_r=max(s(index));
        ind_more=find(s(index)>=mean_r);
        coef1=length(ind_more)/length(index);
        thr_n2(i,1)=(max_r-mean_r)*coef1+mean_r;
        thr1=0.15;
        [pks,~]=findpeaks(-s,'MinPeakHeight',thr1);
        mean_p=mean(pks);
        if mean_p>0.55
            ind_more1=find(pks>=0.75);
            pks=pks(ind_more1);
        end
        max_p=max(pks);
        mean_p=mean(pks);
        ind_more=find(pks>=mean_p);
        coef2=length(ind_more)/length(pks);
        thr_n2(i,2)=-((max_p-mean_p)*coef2+mean_p);
        [row1,~,~]=find(s>=thr_n2(i,1));
        [row2,~,~]=find(s<=thr_n2(i,2));
        row=[row1;row2];
        row=sort(row,'ascend');
        len_row=length(row);
        for j=2:len_row
            if row(j)-row(j-1)<=Fs*8*T
                len=row(j)-row(j-1)+1;
                s(row(j-1):row(j))=zero;
            end
        end
        figure;
        a2(1)=subplot(211);
        plot(sig(1:zind,i));
        signal(1:zind,i)=s;
        a2(2)=subplot(212);
        plot(s);
        linkaxes(a2,'x')
        %% 除失真数据后定位R波
        [~,index1,delay]=my_pan_tompkin(s,Fs,gr);
        index1=index1+delay;
        len_ind1=length(index1);
        mean_r1=mean(s(index1));
        if mean_r1>0.15
            mean_r1=0.15;
        end
        % 对R波峰值位置进行修正
        ind=[];
        for k=1:len_ind1
            st_loc=index1(k)+pre;
            e_loc=index1(k)+back;
            if st_loc>=0 && e_loc<=zind
                [m,loc]=max(s(st_loc:e_loc));
                if m<mean_r1
                    ind=[ind;k];
                end
                index1(k)=st_loc+loc-1;
            end
        end
        index1(ind)=[];
        R_nums=length(index1);
        hold on
        scatter(index1,s(index1));
        for j=2: R_nums-1%丢弃一部分数据
            PreRR=index1(j)-index1(j-1);
            BackRR=index1(j+1)-index1(j);
            RRthr1=Fs/(45/60);
            RRthr2=floor(Fs/(145/60));
            if (RRthr2<=PreRR&&PreRR<=RRthr1)&&(BackRR<=RRthr1&&BackRR>=RRthr2)
                start_loc=index1(j-1);
                end_loc=index1(j+1);
                sig_3R=s(start_loc:end_loc);
                zeros_nums=find(abs(sig_3R)<0.0001);
                if length(zeros_nums)<=5
                    count=count+1;%count对样本个数进行计数
                    diff_3R=diff(sig_3R);
                    len_diff_3R=length(diff_3R);
                    for k=1:len_diff_3R
                        if diff_3R(k)>0
                            diff_3R(k)=1;
                        elseif diff_3R(k)<0
                            diff_3R(k)=0;
                        end
                    end
                    diff_3R_fluct=diff(diff_3R);
                    ecg_hk_3r(count,1:end_loc-start_loc+1+6)=[n,i,index1(j),PreRR,BackRR,...
                        sum(abs(diff_3R_fluct)),sig_3R'];
                end
            end
        end
    end
    zind=find(ecg_hk_3r(:,6),1,'last');
    len_volat=size(ecg_hk_3r,1);
    if zind<len_volat
        ecg_hk_3r=ecg_hk_3r(1:zind,:);
    end
    figure;
    h=histogram(ecg_hk_3r(:,6));
    if flag_s==1
        %% 保存fig图
        path1='..\ecg_hk_3r_wt_fir\';
        filename=[path1,matname,'_volat_hist.fig'];
        savefig(filename);
        %% 保存数据
        filename1=[path1,matname,'_3R_RR.mat'];
        save(filename1,'ecg_hk_3r');
        path2='..\ECG-HK-wt_fir\ECG-HK_Enh_SQ\';
        filename2=[path2,matname,'enh_sq.mat'];
        start_time=txt{n,2};
        save(filename2,'signal','start_time');
    end
end
toc


