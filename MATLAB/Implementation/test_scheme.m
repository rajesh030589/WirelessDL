clc;
clearvars;
close all;

Alpha = 1:1:18;
alpha_list = 10.^(-Alpha/10);

BER = zeros(length(alpha_list),1);
for i = 1:length(alpha_list)
    
    alpha = alpha_list(i);
    b1 = run_scheme(alpha);
    
    BER(i,1) = b1;
end
B = open('/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Datasets/Data_55048.mat');
Alpha1 = B.Alpha.';
Alpha1 = flipud(Alpha1);
B = B.BER;
B = flipud(B(:,1));
figure;
semilogy(Alpha, BER(:,1))
hold on;
semilogy(Alpha1, B(:,1))
grid on
xlabel('Attenuation')
ylabel('BER')
title('Theoretical')
legend('First Transmission','Last Transmission','location','best')
