% function channel_fit()
% 
%     clc;
%     clearvars;
%     close all;
%     currentFolder = pwd;
%     addpath(strcat(currentFolder, '/Imp_Files'));
%     addpath(strcat(currentFolder, '/Imp_Functions'));
%     run('Parameters_feedback.m');
% 
%     % Channel
%     H = open('Channel_Files/Channel_Output.mat');
%     H = H.Channel;
% 
%     H = reshape(H, total_carriers, []);
%     Forward_Channel = zeros(no_of_subcarriers, 2);
% 
%     k = 1;
%     l = 1;
%     % h1 = figure;
%     % h2 = figure;
%     for i = subcarrier_locations
% 
%         if ~(any(pilot_carriers(:) == i))
%             C = abs(H(l, :)).';
%             %         figure(h1)
%             %         scatter(C,i*ones(length(C),1));
%             %         hold on;
%             %         grid on;
% 
%             pd = fitdist(C, 'Rician');
% %             HH = random(pd, 10000, 1);
% %             figure;
% %             hold on;
% %             histogram(HH,50,'Normalization', 'pdf');
% %             histogram(C,50,'Normalization', 'pdf');
% 
%             %         figure(h2)
%             %         scatter(HH,i*ones(length(HH),1));
%             %         hold on;
%             %         grid on;
%             %
%             mean = pd.s;
%             sig = pd.sigma;
%             Forward_Channel(k, 1) = mean^2 / sig^2;
%             Forward_Channel(k, 2) = sqrt(mean^2 + sig^2);
%             k = k + 1;
%         end
% 
%         l = l + 1;
%     end
% 
%     save(strcat('Channel_Files/Forward_Channel.mat'), 'Forward_Channel');
% 
%     % Noise
%     N = open('Channel_Files/Noise_Output.mat');
%     N = N.Noise;
%     N = real(N.');
%     Forward_Noise = zeros(no_of_subcarriers, 2);
% 
%     for i = 1:no_of_subcarriers
%         C = real(N(i, :)).';
%         pd = fitdist(C, 'Normal');
%         Forward_Noise(i, 1) = pd.mean;
%         Forward_Noise(i, 2) = (pd.sigma)^2;
%     end
% 
%     save(strcat('Channel_Files/Forward_Noise.mat'), 'Forward_Noise');
% end

function channel_fit()

    clc;
    clearvars;
    close all;
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    % Channel
    H = open('Channel_Files/Channel_Output.mat');
    H = H.Channel;

    H = reshape(H, total_carriers, []);
    Backward_Channel = zeros(no_of_subcarriers, 2);

    k = 1;
    l = 1;
    % h1 = figure;
    % h2 = figure;
    for i = subcarrier_locations

        if ~(any(pilot_carriers(:) == i))
            C = abs(H(l, :)).';
            %         figure(h1)
            %         scatter(C,i*ones(length(C),1));
            %         hold on;
            %         grid on;

            pd = fitdist(C, 'Rician');
%             HH = random(pd, 10000, 1);
%             figure;
%             hold on;
%             histogram(HH,50,'Normalization', 'pdf');
%             histogram(C,50,'Normalization', 'pdf');

            %         figure(h2)
            %         scatter(HH,i*ones(length(HH),1));
            %         hold on;
            %         grid on;
            %
            mean = pd.s;
            sig = pd.sigma;
            Backward_Channel(k, 1) = mean^2 / sig^2;
            Backward_Channel(k, 2) = sqrt(mean^2 + sig^2);
            k = k + 1;
        end

        l = l + 1;
    end

    save(strcat('Channel_Files/Backward_Channel.mat'), 'Backward_Channel');

    % Noise
    N = open('Channel_Files/Noise_Output.mat');
    N = N.Noise;
    N = real(N.');
    Backward_Noise = zeros(no_of_subcarriers, 2);

    for i = 1:no_of_subcarriers
        C = real(N(i, :)).';
        pd = fitdist(C, 'Normal');
        Backward_Noise(i, 1) = pd.mean;
        Backward_Noise(i, 2) = (pd.sigma)^2;
    end

    save(strcat('Channel_Files/Backward_Noise.mat'), 'Backward_Noise');
end