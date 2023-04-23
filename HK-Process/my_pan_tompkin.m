function [qrs_amp_raw,qrs_i_raw,delay]=my_pan_tompkin(ecg,fs,gr)

%% function [qrs_amp_raw,qrs_i_raw,delay]=pan_tompkin(ecg,fs)
% 完成Pan-Tompkins算法的实现
%% Inputs
% ecg : raw ecg vector signal 1d signal
% fs : sampling frequency e.g. 200Hz, 400Hz and etc
% gr : flag to plot or not plot (set it 1 to have a plot or set it zero not
% to see any plots
%% Outputs
% qrs_amp_raw : amplitude of R waves amplitudes
% qrs_i_raw : index of R waves
% delay : number of samples which the signal is delayed due to the
% filtering
%% Method :

%% PreProcessing
% 1) Signal is preprocessed , if the sampling frequency is higher then it is downsampled
% and if it is lower upsampled to make the sampling frequency 200 Hz
% with the same filtering setups introduced in Pan
% tompkins paper (a combination of low pass and high pass filter 5-15 Hz)
% to get rid of the baseline wander and muscle noise.

% 2) The filtered signal
% is derivated using a derivating filter to high light the QRS complex.

% 3) Signal is squared.4)Signal is averaged with a moving window to get rid
% of noise (0.150 seconds length).

% 5) depending on the sampling frequency of your signal the filtering
% options are changed to best match the characteristics of your ecg signal

% 6) Unlike the other implementations in this implementation the desicion
% rule of the Pan tompkins is implemented completely.

%% Decision Rule
% At this point in the algorithm, the preceding stages have produced a roughly pulse-shaped
% waveform at the output of the MWI . The determination as to whether this pulse
% corresponds to a QRS complex (as opposed to a high-sloped T-wave or a noise artefact) is
% performed with an adaptive thresholding operation and other decision
% rules outlined below;

% a) FIDUCIAL MARK - The waveform is first processed to produce a set of weighted unit
% samples at the location of the MWI maxima. This is done in order to localize the QRS
% complex to a single instant of time. The w[k] weighting is the maxima value.

% b) THRESHOLDING - When analyzing the amplitude of the MWI output, the algorithm uses
% two threshold values (THR_SIG and THR_NOISE, appropriately initialized during a brief
% 2 second training phase) that continuously adapt to changing ECG signal quality. The
% first pass through y[n] uses these thresholds to classify the each non-zero sample
% (CURRENTPEAK) as either signal or noise:
% If CURRENTPEAK > THR_SIG, that location is identified as a QRS complex
% candidate?and the signal level (SIG_LEV) is updated:
% SIG _ LEV = 0.125 CURRENTPEAK + 0.875?SIG _ LEV

% If THR_NOISE < CURRENTPEAK < THR_SIG, then that location is identified as a
% Noise peak?and the noise level (NOISE_LEV) is updated:
% NOISE _ LEV = 0.125CURRENTPEAK + 0.875?NOISE _ LEV
% Based on new estimates of the signal and noise levels (SIG_LEV and NOISE_LEV,
% respectively) at that point in the ECG, the thresholds are adjusted as follows:
% THR _ SIG = NOISE _ LEV + 0.25 ?(SIG _ LEV-NOISE _ LEV )
% THR _ NOISE = 0.5?(THR _ SIG)
% These adjustments lower the threshold gradually in signal segments that are deemed to
% be of poorer quality.


% c) SEARCHBACK FOR MISSED QRS COMPLEXES - In the thresholding step above, if
% CURRENTPEAK < THR_SIG, the peak is deemed not to have resulted from a QRS
% complex. If however, an unreasonably long period has expired without an abovethreshold
% peak, the algorithm will assume a QRS has been missed and perform a
% searchback. This limits the number of false negatives. The minimum time used to trigger
% a searchback is 1.66 times the current R peak to R peak time period (called the RR
% interval). This value has a physiological origin - the time value between adjacent
% heartbeats cannot change more quickly than this. The missed QRS complex is assumed
% to occur at the location of the highest peak in the interval that lies between THR_SIG and
% THR_NOISE. In this algorithm, two average RR intervals are stored,the first RR interval is
% calculated as an average of the last eight QRS locations in order to adapt to changing heart
% rate and the second RR interval mean is the mean
% of the most regular RR intervals . The threshold is lowered if the heart rate is not regular
% to improve detection.

% d) ELIMINATION OF MULTIPLE DETECTIONS WITHIN REFRACTORY PERIOD - It is
% impossible for a legitimate QRS complex to occur if it lies within 200ms after a previously
% detected one. This constraint is a physiological one ?due to the refractory period during
% which ventricular depolarization cannot occur despite a stimulus[1]. As QRS complex
% candidates are generated, the algorithm eliminates such physically impossible events,
% thereby reducing false positives.

% e) T WAVE DISCRIMINATION - Finally, if a QRS candidate occurs after the 200ms
% refractory period but within 360ms of the previous QRS, the algorithm determines
% whether this is a genuine QRS complex of the next heartbeat or an abnormally prominent
% T wave. This decision is based on the mean slope of the waveform at that position. A slope of
% less than one half that of the previous QRS complex is consistent with the slower
% changing behaviour of a T wave ?otherwise, it becomes a QRS detection.
% Extra concept : beside the points mentioned in the paper, this code also
% checks if the occured peak which is less than 360 msec latency has also a
% latency less than 0,5*mean_RR if yes this is counted as noise

% f) In the final stage , the output of R waves detected in smoothed signal is analyzed and double
% checked with the help of the output of the bandpass signal to improve
% detection and find the original index of the real R waves on the raw ecg
% signal

%% References :

%[1]PAN.J, TOMPKINS. W.J,"A Real-Time QRS Detection Algorithm" IEEE
%TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. BME-32, NO. 3, MARCH 1985.

%% Author : Hooman Sedghamiz
% Linkoping university
% email : hoose792@student.liu.se
% hooman.sedghamiz@medel.com

% Any direct or indirect use of this code should be referenced
% Copyright march 2014
%%
if ~isvector(ecg)
    error('ecg must be a row or column vector');
end


if nargin < 3
    gr = 1;   % on default the function always plots
end
ecg = ecg(:); % vectorize

%% Initialize
qrs_c =[]; %amplitude of R/R波的幅值
qrs_i =[]; %index/索引
% SIG_LEV_M = 0;
nois_c =[];
nois_i =[];
delay = 0;
skip = 0; % becomes one when a T wave is detected/检测到T波，+1
not_nois = 0; % it is not noise when not_nois = 1
selected_RR =[]; % Selected RR intervals 选择RR间隔
m_selected_RR = 0;
mean_RR = 0;
qrs_i_raw =[];
qrs_amp_raw=[];
ser_back = 0;
test_m = 0;
SIGL_buf = [];
NOISL_buf = [];
THRS_buf = [];
SIGL_buf1 = [];
NOISL_buf1 = [];
THRS_buf1 = [];

%% 没有对信号进行归一化
%% Plot differently based on filtering settings/基于过滤设置的方式
if gr
    if fs == 200
        figure,  ax(1)=subplot(321);plot(ecg);axis tight;title('原始信号');%title('Raw ECG Signal');
    else
        figure,  ax(1)=subplot(3,2,[1 2]);plot(ecg);axis tight;title('原始信号');%title('Raw ECG Signal');
    end
end
%% Noise cancelation(Filtering) % Filters (Filter in between 5-15 Hz)/低通滤波
%怎么计算出其低通滤波的截止频率为30Hz
if fs == 200
    %% Low Pass Filter  H(z) = ((1 - z^(-6))^2)/(1 - z^(-1))^2
    b = [1 0 0 0 0 0 -2 0 0 0 0 0 1];
    a = [1 -2 1];
    h_l = filter(b,a,[1 zeros(1,12)]);
    ecg_l = conv (ecg ,h_l);%计算原始信号与滤波后信号的相关系数,有什么作用？
    %不是计算相关系数，而是进行卷积计算，就是相当于滤波
    ecg_l = ecg_l/ max( abs(ecg_l));
    delay = 6; %based on the paper
    if gr
        ax(2)=subplot(322);plot(ecg_l);axis tight;title('低通滤波');%title('Low pass filtered');
    end
    %% High Pass filter H(z) = (-1+32z^(-16)+z^(-32))/(1+z^(-1))/高通滤波
    %高通计算：0.7Hz以上
    b = [-1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 -32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1];
    a = [1 -1];
    h_h = filter(b,a,[1 zeros(1,32)]);
    ecg_h = conv (ecg_l ,h_h);
    ecg_h = ecg_h/ max( abs(ecg_h));
    delay = delay + 16; % 16 samples for highpass filtering
    if gr
        ax(3)=subplot(323);plot(ecg_h);axis tight;title('高通滤波');%title('High Pass Filtered');
    end
else
    %% bandpass filter for Noise cancelation of other sampling frequencies(Filtering)/带通滤波
    %滤波效果已验证，可以实现，freqz绘图时将参数a,b参数位置搞反了
    %这个也没有延迟
    f1=5; %cuttoff low frequency to get rid of baseline wander/消除基线漂移
    f2=15; %cuttoff frequency to discard high frequency noise/丢弃高频噪声
    Wn=[f1 f2]*2/fs; % cutt off based on fs
    N = 3; % order of 3 less processing
    [a,b] = butter(N,Wn); %bandpass filtering
    ecg_h = filtfilt(a,b,ecg);
    ecg_h = ecg_h/ max( abs(ecg_h));
    if gr
        ax(3)=subplot(323);plot(ecg_h);axis tight;title('带通滤波');%title('Band Pass Filtered');
    end
end
%% derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))/微分滤波器
%微分器：逻辑上，平滑使用的是积分，锐化则应使用微分，故使信号变得尖锐
% h_d = [-1 -2 0 2 1];%1/8*fs
h_d = [-1 -2 0 2 1];%1/8*fs%乘不乘0.125不影响
% h_d1=[1 0 -1];
ecg_d = conv (ecg_h ,h_d);%微分器使用的让我有些懵圈,不明白它为什么会呈现这样的效果
ecg_d = ecg_d/max(ecg_d);
delay = delay + 2; % delay of derivative filter 2 samples
if gr
    ax(4)=subplot(324);plot(ecg_d);axis tight;title('信号求导');%title('Filtered with the derivative filter');
end
%% Squaring nonlinearly enhance the dominant peaks/非线性平方增强了主导峰
ecg_s = ecg_d.^2;
if gr
    ax(5)=subplot(325);plot(ecg_s);axis tight;title('平方');%title('Squared');
end



%% Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
%n这个点向前再去N-1个点，进行均值处理
%避免过于尖锐
ecg_m = conv(ecg_s ,ones(1 ,round(0.150*fs))/round(0.150*fs));
% delay = delay + 15;%这个没有造成延迟

if gr
    ax(6)=subplot(326);plot(ecg_m);axis tight;
    title('黑色:噪声:绿色:自适应阈值;红色:信号电平;红圈:QRS自适应阈值');
    % title('Averaged with 30 samples length,Black noise,Green Adaptive Threshold,RED Sig Level,Red circles QRS adaptive threshold');
    axis tight;
end

%% Fiducial Mark
% Note : a minimum distance of 40 samples is considered between each R wave
% since in physiological point of view no RR wave can occur in less than
% 200 msec distance
%根据这个定义心率极限是300，而我查的资料表明心率极限是220
%不是0.2=60/300，应该是60/220,；正常人的心率范围为45-180
% [pks,locs] = findpeaks(ecg_m,'MINPEAKDISTANCE',round(0.2*fs));
[pks,locs] = findpeaks(ecg_m,'MINPEAKDISTANCE',round(60/240*fs));


%% initialize the training phase (2 seconds of the signal) to determine the THR_SIG and THR_NOISE
%初始化训练阶段(信号的2秒)，以确定THR_SIG和THR_NOISE
if max(ecg_m(1:2*fs))*1/3<0.05
    THR_SIG_M=0.05;
    THR_NOISE_M=0.005;
    SIG_LEV_M= THR_SIG_M;
    NOISE_LEV_M = THR_NOISE_M;
else
    THR_SIG_M = max(ecg_m(1:2*fs))*1/3; % 0.25 of the max amplitude/最大振幅的0.25
    THR_NOISE_M = mean(ecg_m(1:2*fs))*1/2; % 0.5 of the mean signal is considered to be noise/平均信号的0.5被认为是噪声
    SIG_LEV_M= THR_SIG_M;
    NOISE_LEV_M = THR_NOISE_M;
end


%% Initialize bandpath filter threshold(2 seconds of the bandpass signal)
%初始化带径滤波器阈值(带通信号的2秒)
THR_SIG_H = max(ecg_h(1:2*fs))*1/3; % 0.25 of the max amplitude
THR_NOISE_H = mean(ecg_h(1:2*fs))*1/2; %
SIG_LEV_H = THR_SIG_H; % Signal level in Bandpassed filter
NOISE_LEV_H = THR_NOISE_H; % Noise level in Bandpassed filter
%% Thresholding and online desicion rule
%阈值和在线决策规则

for i = 1 : length(pks)
    
    %% locate the corresponding peak in the filtered signal
    %在滤波信号中找到相应的峰值（一段信号的最大值）
    if locs(i)-round(0.150*fs)>= 1 && locs(i)<= length(ecg_h)
        [y_i,x_i] = max(ecg_h(locs(i)-round(0.150*fs):locs(i)));
    else
        if i == 1
            [y_i,x_i] = max(ecg_h(1:locs(i)));
            ser_back = 1;%初始值
        elseif locs(i)>= length(ecg_h)
            [y_i,x_i] = max(ecg_h(locs(i)-round(0.150*fs):end));
        end
        
    end
    
    
    %% update the heart_rate (Two heart rate means one the moste recent and the other selected)
    %更新heart_rate(两个心率意味着一个是最近的，另一个是已选的)
    if length(qrs_c) >= 9 %避免信号过小
        
        diffRR = diff(qrs_i(end-8:end)); %calculate RR interval/计算RR间隔
        %这里需要限定RR波的范围
        mean_RR = mean(diffRR); % calculate the mean of 8 previous R waves interval/计算之前8个R波间隔的平均值
        comp =qrs_i(end)-qrs_i(end-1); %latest RR/最新的RR间隔
        if comp <= 0.92*mean_RR || comp >= 1.16*mean_RR%超过这个范围，说明R波漏检
            % lower down thresholds to detect better in MVI/降低阈值以更好地检测MVI
            THR_SIG_M = 0.6*(THR_SIG_M);%其值减半%0.5改成0.6
            %THR_NOISE = 0.5*(THR_SIG);
            % lower down thresholds to detect better in Bandpass
            % filtered/降低阈值，以更好地检测
            THR_SIG_H = 0.6*(THR_SIG_H);
            %THR_NOISE1 = 0.5*(THR_SIG1);
            
        else
            m_selected_RR = mean_RR; %the latest regular beats mean/最新的常规平均节拍
        end
        
    end
    
    %% calculate the mean of the last 8 R waves to make sure that QRS is not
    % missing(If no R detected , trigger a search back) 1.66*mean
    %计算最后8个R波的平均值，以确保QRS不丢失(如果没有检测到R，触发搜索返回)1.66*平均值
    if m_selected_RR
        test_m = m_selected_RR; %if the regular RR availabe use it/如果常规RR可用，则使用它
    elseif mean_RR && m_selected_RR == 0
        test_m = mean_RR;
    else
        test_m = 0;
    end
    
    if test_m%带通滤波信号的8个R波的平均值
        if (locs(i) - qrs_i(end)) >= round(1.66*test_m)% it shows a QRS is missed /它表明一个QRS缺失
            [pks_temp,locs_temp] = max(ecg_m(qrs_i(end)+ round(0.200*fs):locs(i)-round(0.200*fs))); % search back and locate the max in this interval
            %搜索并在此间隔中找到最大值
            locs_temp = qrs_i(end)+ round(0.200*fs) + locs_temp -1; %location
            %为什么要减1？
            if pks_temp > THR_NOISE_M
                %添加幅值和索引
                qrs_c = [qrs_c pks_temp];
                qrs_i = [qrs_i locs_temp];
                
                % find the location in filtered sig/在滤波信号中找到位置
                if locs_temp <= length(ecg_h)
                    [y_i_t,x_i_t] = max(ecg_h(locs_temp-round(0.150*fs):locs_temp));
                else%不会超过locs_temp的范围
                    [y_i_t,x_i_t] = max(ecg_h(locs_temp-round(0.150*fs):end));
                end
                % take care of bandpass signal threshold/注意带通信号阈值
                if y_i_t > THR_NOISE_H
                    
                    qrs_i_raw = [qrs_i_raw locs_temp-round(0.150*fs)+ (x_i_t - 1)];% save index of bandpass /保存带宽索引
                    qrs_amp_raw =[qrs_amp_raw y_i_t]; %save amplitude of bandpass
                    SIG_LEV_H = 0.25*y_i_t + 0.75*SIG_LEV_H; %when found with the second thres %更改阈值
                end
                
                not_nois = 1; %是噪声
                SIG_LEV_M = 0.25*pks_temp + 0.75*SIG_LEV_M ;  %when found with the second threshold/当发现与第二个阈值
            end
            
        else %不是噪声
            not_nois = 0;
            
        end
    end
    
    
    
    
    %%  find noise and QRS peaks
    if pks(i) >= THR_SIG_M
        
        % if a QRS candidate occurs within 360ms of the previous QRS
        % ,the algorithm determines if its T wave or QRS
        %如果一个候选QRS出现在前一个QRS的360毫秒内，算法判断它是T波还是QRS
        if length(qrs_c) >= 3
            if (locs(i)-qrs_i(end)) <= round(0.3600*fs)
                Slope1 = mean(diff(ecg_m(locs(i)-round(0.075*fs):locs(i)))); %mean slope of the waveform at that position
                %波形在那个位置的平均斜率
                Slope2 = mean(diff(ecg_m(qrs_i(end)-round(0.075*fs):qrs_i(end)))); %mean slope of previous R wave
                %前R波的平均斜率
                if abs(Slope1) <= abs(0.5*(Slope2))  % slope less then 0.5 of previous R
                    %斜率小于之前R的0.5
                    nois_c = [nois_c pks(i)];
                    nois_i = [nois_i locs(i)];
                    skip = 1; % T wave identification/T波识别
                    % adjust noise level in both filtered and
                    % MVI/在滤波和MVI中调整噪声水平
                    NOISE_LEV_H = 0.125*y_i + 0.875*NOISE_LEV_H;
                    NOISE_LEV_M = 0.125*pks(i) + 0.875*NOISE_LEV_M;
                else
                    skip = 0;
                end
                
            end
        end
        
        if skip == 0  % skip is 1 when a T wave is detected /当检测到T波时，skip为1
            qrs_c = [qrs_c pks(i)];
            qrs_i = [qrs_i locs(i)];
            
            % bandpass filter check threshold/带通滤波器检查阈值
            if y_i >= THR_SIG_H%带通滤波信号
                if ser_back
                    qrs_i_raw = [qrs_i_raw x_i];  % save index of bandpass
                else
                    qrs_i_raw = [qrs_i_raw locs(i)-round(0.150*fs)+ (x_i - 1)];% save index of bandpass
                end
                qrs_amp_raw =[qrs_amp_raw y_i];% save amplitude of bandpass
                SIG_LEV_H = 0.125*y_i + 0.875*SIG_LEV_H;% adjust threshold for bandpass filtered sig/调整带通滤波信号的阈值
            end
            
            % adjust Signal level
            SIG_LEV_M = 0.125*pks(i) + 0.875*SIG_LEV_M ;
        end
        
        
    elseif THR_NOISE_M <= pks(i) && pks(i)<THR_SIG_M
        
        %adjust Noise level in filtered sig
        NOISE_LEV_H = 0.125*y_i + 0.875*NOISE_LEV_H;
        %adjust Noise level in MVI
        NOISE_LEV_M = 0.125*pks(i) + 0.875*NOISE_LEV_M;
        
        
        
    elseif pks(i) < THR_NOISE_M
        nois_c = [nois_c pks(i)];
        nois_i = [nois_i locs(i)];
        
        % noise level in filtered signal
        NOISE_LEV_H = 0.125*y_i + 0.875*NOISE_LEV_H;
        %end
        
        %adjust Noise level in MVI
        NOISE_LEV_M = 0.125*pks(i) + 0.875*NOISE_LEV_M;
        
        
    end
    
    
    
    
    
    %% adjust the threshold with SNR/根据信噪比调整阈值
    if NOISE_LEV_M ~= 0 || SIG_LEV_M ~= 0
        THR_SIG_M = NOISE_LEV_M + 0.25*(abs(SIG_LEV_M - NOISE_LEV_M));
        THR_NOISE_M = 0.5*(THR_SIG_M);
    end
    
    % adjust the threshold with SNR for bandpassed signal/根据信噪比调整带通信号的阈值
    if NOISE_LEV_H ~= 0 || SIG_LEV_H ~= 0
        THR_SIG_H = NOISE_LEV_H + 0.25*(abs(SIG_LEV_H - NOISE_LEV_H));
        THR_NOISE_H = 0.5*(THR_SIG_H);
    end
    
    
    % take a track of thresholds of smoothed signal/取平滑信号的阈值轨迹
    SIGL_buf = [SIGL_buf SIG_LEV_M];
    NOISL_buf = [NOISL_buf NOISE_LEV_M];
    THRS_buf = [THRS_buf THR_SIG_M];
    
    % take a track of thresholds of filtered signal/跟踪滤波信号的阈值
    SIGL_buf1 = [SIGL_buf1 SIG_LEV_H];
    NOISL_buf1 = [NOISL_buf1 NOISE_LEV_H];
    THRS_buf1 = [THRS_buf1 THR_SIG_H];
    
    
    
    
    skip = 0; %reset parameters
    not_nois = 0; %reset parameters
    ser_back = 0;  %reset bandpass param
end

if gr
    hold on,scatter(qrs_i,qrs_c,'m');
    hold on,plot(locs,NOISL_buf,'--k','LineWidth',2);
    hold on,plot(locs,SIGL_buf,'--r','LineWidth',2);
    hold on,plot(locs,THRS_buf,'--g','LineWidth',2);
    % if ax(:)%运行报错，注释掉
    linkaxes(ax,'x');
    zoom on;
    % end
end




%% overlay on the signals/信号叠加
if gr
    figure,az(1)=subplot(311);plot(ecg_h);
    title('滤波信号的QRS');%title('QRS on Filtered Signal');
    axis tight;
    hold on,scatter(qrs_i_raw,qrs_amp_raw,'m');
    hold on,plot(locs,NOISL_buf1,'LineWidth',2,'Linestyle','--','color','k');
    hold on,plot(locs,SIGL_buf1,'LineWidth',2,'Linestyle','-.','color','r');
    hold on,plot(locs,THRS_buf1,'LineWidth',2,'Linestyle','-.','color','g');
    az(2)=subplot(312);plot(ecg_m);
    title('MVI信号和噪声电平(黑色)、信号电平(红色)和自适应阈值(绿色)的QRS');
    %title('QRS on MVI signal and Noise level(black),Signal Level (red) and Adaptive Threshold(green)');axis tight;
    hold on,scatter(qrs_i,qrs_c,'m');
    hold on,plot(locs,NOISL_buf,'LineWidth',2,'Linestyle','--','color','k');
    hold on,plot(locs,SIGL_buf,'LineWidth',2,'Linestyle','-.','color','r');
    hold on,plot(locs,THRS_buf,'LineWidth',2,'Linestyle','-.','color','g');
    az(3)=subplot(313);plot(ecg-mean(ecg));
    title('ECG信号中定位的QRS脉冲序列');
    % title('Pulse train of the found QRS on ECG signal');
    axis tight;
    line(repmat(qrs_i_raw,[2 1]),repmat([min(ecg-mean(ecg))/2; max(ecg-mean(ecg))/2],size(qrs_i_raw)),'LineWidth',2.5,'LineStyle','-.','Color','r');
    linkaxes(az,'x');
    zoom on;
end
end










