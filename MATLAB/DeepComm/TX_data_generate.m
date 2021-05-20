clc;
clearvars;
close all;

% Parameters
no_of_ofdm_symbols = 800;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_symbols = no_of_ofdm_symbols*no_of_subcarriers;
mod_order = 4;
bit_per_symbol = log2(mod_order);
total_no_bits = total_symbols*bit_per_symbol;
encoded_no_bits = (total_no_bits - 12)/3;

total_no_of_samples = no_of_ofdm_symbols*(size_of_FFT+ cp_length);
no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT,1);
pilot_values(pilot_carriers,1) = [1;1;1;-1];

% Encoding
turboEnc = comm.TurboEncoder('InterleaverIndicesSource','Input port');
dataIn = randi([0 1],encoded_no_bits,1);
save('dataIn.mat', 'dataIn');
intrlvrInd = randperm(encoded_no_bits);
save('Interleaver.mat','intrlvrInd');

encoded_data = step(turboEnc,dataIn, intrlvrInd);
release(turboEnc);

% Modulation
mod_symbols = qammod(encoded_data,mod_order,'InputType','bit','UnitAveragePower',true);
save('mod_symbols.mat','mod_symbols');

% Subcarrier Allocation
P = open('Pilot_matrix.mat');
P = P.PILOT;

Pilots = zeros(no_of_pilot_carriers, no_of_ofdm_symbols);

A = zeros(size_of_FFT, no_of_ofdm_symbols);
k = 1;
for i = 1:no_of_ofdm_symbols
    for j = subcarrier_locations
        if any(pilot_carriers(:) == j)
            A(j,i) = pilot_values(j,1)*P(1 + mod(i-1,127));
        else
        A(j,i) = mod_symbols(k);
        k = k+1;
        end
        
    end
    Pilots(:,i) = A(pilot_carriers,i);
end
save('Pilots.mat','Pilots');

% IFFT to generate tx symbols
for i = 1:no_of_ofdm_symbols
    IFFT_Data = ifft(fftshift(A(1:size_of_FFT,i)),size_of_FFT);
    A(1:size_of_FFT+cp_length,i) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
end
TX = A(:);

% Normalize the modulated data Power
TX = TX.*(.8/(max(max(abs(real(TX))),max(abs(imag(TX))))));


% Short Preamble Field
STS = open('STS.mat');
STS = STS.STS;
IFFT_Data = ifft(STS(1:size_of_FFT,1),size_of_FFT);
sts(1:size_of_FFT+cp_length,1) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
sts(1) = sts(1)*0.5;
sts(size_of_FFT+cp_length) = sts(size_of_FFT+cp_length)*0.5;
sts = [sts;sts];


% Long Preamble Field
LTS = open('LTS.mat');
LTS = LTS.LTS;
IFFT_Data = ifft(LTS(1:size_of_FFT,1),size_of_FFT);
lts(1:size_of_FFT+cp_length,1) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
lts(1) = lts(1)*0.5;
lts(size_of_FFT+cp_length) = lts(size_of_FFT+cp_length)*0.5;
lts = [lts;lts];

TX = [sts;lts;TX];


write_complex_binary(TX,'TX.bin');
