function channel_fit()

clc;

currentFolder = pwd;
addpath(strcat(currentFolder, '/Imp_Files'));
addpath(strcat(currentFolder, '/Imp_Functions'));
run('Parameters_feedback.m');

% Channel 
H = open('Feedback_Files/Channel_Output.mat');
H = H.Channel;

H = reshape(H, total_carriers, []);
Forward_Channel = zeros(no_of_subcarriers,2);

k = 1;
l = 1;
for i = subcarrier_locations
    if ~(any(pilot_carriers(:) == i))
        C = abs(H(l,:)).';
        pd = fitdist(C,'Rician');
        mean = pd.s;
        sig = pd.sigma;
        Forward_Channel(k,1) = mean^2/sig^2;
        Forward_Channel(k,2) = sqrt(mean^2 + sig^2);
        k = k+1;
    end
    l= l+1;
end
save(strcat('Feedback_Files/Forward_Channel.mat'),'Forward_Channel');

% Noise
N = open('Feedback_Files/Noise_Output.mat');
N = N.Noise;
C = real(N);
pd = fitdist(C,'Normal');
Forward_Noise = zeros(1,2);

Forward_Noise(1,1) = pd.mean;
Forward_Noise(1,2) = (pd.sigma)^2;

save(strcat('Feedback_Files/Forward_Noise.mat'),'Forward_Noise');
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