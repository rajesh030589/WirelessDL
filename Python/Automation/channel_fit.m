function channel_fit()

clc;
clearvars;

currentFolder = pwd;
addpath(strcat(currentFolder, '/Imp_Files'));
addpath(strcat(currentFolder, '/Imp_Functions'));
run('Parameters_feedback.m');

% Channel 
H = open('Channel_Files/Channel_Output.mat');
H = H.Channel;

H = reshape(H, total_carriers, []);
Forward_Channel = zeros(no_of_subcarriers,2);

k = 1;
l = 1;
% h1 = figure;
% h2 = figure;
for i = subcarrier_locations
    if ~(any(pilot_carriers(:) == i))
        C = abs(H(l,:)).';
%         figure(h1)
%         scatter(C,i*ones(length(C),1));
%         hold on;
%         grid on;
        
        pd = fitdist(C,'Rician');
%         HH = random(pd,10000,1);  
        
%         figure(h2)
%         scatter(HH,i*ones(length(HH),1));
%         hold on;
%         grid on;
%         
        mean = pd.s;
        sig = pd.sigma;
        Forward_Channel(k,1) = mean^2/sig^2;
        Forward_Channel(k,2) = sqrt(mean^2 + sig^2);
        k = k+1;
    end
    l= l+1;
end
save(strcat('Channel_Files/Forward_Channel.mat'),'Forward_Channel');

% Noise
N = open('Channel_Files/Noise_Output.mat');
N = N.Noise;
N = real(N.');
Forward_Noise = zeros(no_of_subcarriers,2);
for i = 1:no_of_subcarriers
C = real(N(i,:)).';
pd = fitdist(C,'Normal');
Forward_Noise(i,1) = pd.mean;
Forward_Noise(i,2) = (pd.sigma)^2;
end
save(strcat('Channel_Files/Forward_Noise.mat'),'Forward_Noise');
end

% function channel_fit()
% 
% clc;
% 
% currentFolder = pwd;
% addpath(strcat(currentFolder, '/Imp_Files'));
% addpath(strcat(currentFolder, '/Imp_Functions'));
% run('Parameters_feedback.m');
% 
% H = open('Feedback_Files/Channel_Output.mat');
% H = H.Channel;
% 
% H = reshape(H, total_carriers, []);
% Forward_Channel = zeros(no_of_subcarriers,2);
% 
% k = 1;
% for i = 1:total_carriers
%     if ~(any(pilot_carriers(:) == i))
%         C = abs(H(i,:)).';
%         pd = fitdist(H,'Rician');
%         mean = pd.s;
%         sig = pd.sigma;
%         Forward_Channel(k,1) = mean^2/sig^2;
%         Forward_Channel(k,2) = msqrt(mean^2 + sig^2);
%     end
%     end
%         
% 
% % x_values = 0:.0001:.01;
% % y = pdf(pd,x_values);
% % plot(x_values,y)
% % hold on;
% % histogram(H,'normalization','pdf')
% 
% 
% mean = pd.s;
% sig = pd.sigma;
% 
% K = mean^2/sig^2
% 
% mean1 = sqrt(K/(K+1));
% sig1 = sqrt(1/(K+1));
% A = sqrt(mean^2 + sig^2)
% 
% X = A.*abs(mean1 + sig1.*(randn(50000,1) + 1j* randn(50000,1)));
% 
% x_values = 0:.0001:.01;
% y = pdf(pd,x_values);
% plot(x_values,y)
% hold on;
% histogram(X,'normalization','pdf')
% 
% 
% 
% end