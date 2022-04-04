function BER1 = run_scheme(alpha)

L = 200000;

% Data generation
D = randi([0 1], L, 1);

encoder_data = reshape(D, L, []);
save(strcat('Data_Files/TX_Encoded', num2str(1), '.mat'), 'encoder_data');
system(strcat('python3 /home/rajesh/ActiveFeedback/WirelessDL/MATLAB/Implementation/Imp_Functions/TX_NN_Encoder', num2str(1), '.py'));
mod_symbols = open(strcat('Data_Files/TX_Modulated', num2str(1), '.mat'));
C1 = double(mod_symbols.output);

[H1, N1] = gen_fading_noise(L);
Y1 = alpha*C1 .* H1 + N1;
d1 = (Y1 < 0);

[H1, N1] = gen_fading_noise(L);
Y1 = alpha*C1 .* H1 + N1;
d2 = (Y1 < 0);

[H1, N1] = gen_fading_noise(L);
Y1 = alpha*C1 .* H1 + N1;
d3 = (Y1 < 0);

decoded_data = ((d1 + d2 + d3) > 1.5);

data_to_encode = D;

BER1 = biterr(decoded_data, data_to_encode) / L;


end
function [H, Noise] = gen_fading_noise(N)
    fwd_stats = open('Channel_Files/Forward_Channel.mat');
    fwd_stats = fwd_stats.Forward_Channel;

    rand_subc = randi([1 48], N, 1);

    K = fwd_stats(rand_subc, 1);
    A = fwd_stats(rand_subc, 2);

    coeffLOS = sqrt(K ./ (K + 1));
    coeffNLOS = sqrt(1 ./ (K + 1));
    hLOSReal = ones(N, 1);
    hLOSImag = ones(N, 1);
    hNLOSReal = randn(N, 1);
    hNLOSImag = randn(N, 1);

    fading_h = coeffLOS .* (hLOSReal + 1j * hLOSImag) + coeffNLOS .* (hNLOSReal + 1j * hNLOSImag);
    H = abs(A .* fading_h);

    fwd_stats = open('Channel_Files/Forward_Noise.mat');
    fwd_stats = fwd_stats.Forward_Noise;

    Mean = fwd_stats(rand_subc, 1);
    Var = fwd_stats(rand_subc, 2);

    Noise = Mean + sqrt(Var) .* randn(N, 1);

end
