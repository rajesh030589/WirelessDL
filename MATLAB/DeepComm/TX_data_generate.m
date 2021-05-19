clc;
clearvars;
close all;

% Parameters
no_of_ofdm_symbols = 800;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 51;
total_symbols = no_of_ofdm_symbols*no_of_subcarriers;
mod_order = 2;
bit_per_symbol = log2(mod_order);
total_no_bits = total_symbols*bit_per_symbol;
encoded_no_bits = (total_no_bits - 12)/3;



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

A = zeros(size_of_FFT, no_of_ofdm_symbols);
k = 1;
for i = 1:no_of_ofdm_symbols
    for j = [7:32 34:58]
        A(j,i) = mod_symbols(k);
        k = k+1;
    end
end

% IFFT to generate tx symbols
for i = 1:no_of_ofdm_symbols
    IFFT_Data = ifft(fftshift(A(1:size_of_FFT,i)),size_of_FFT);
    A(1:size_of_FFT+cp_length,i) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
end
TX = A(:);

% Normalize the modulated data Power
TX = TX.*(.8/(max(max(abs(real(TX))),max(abs(imag(TX))))));


% Short Preamble Field
STS =  [0, 0, 0, 0, 0, 0, 0, 0, (1.472+1.472j), 0, 0, 0, (-1.472-1.472j),...
    0, 0, 0, (1.472+1.472j), 0, 0, 0, (-1.472-1.472j), 0, 0, 0, (-1.472-1.472j),...
    0, 0, 0, (1.472+1.472j), 0, 0, 0, 0, 0, 0, 0, (-1.472-1.472j), 0, 0, 0, (-1.472-1.472j),...
    0, 0, 0, (1.472+1.472j), 0, 0, 0, (1.472+1.472j), 0, 0, 0, (1.472+1.472j), 0, 0, 0, (1.472+1.472j),...
    0, 0, 0, 0, 0, 0, 0];

STS = STS.';
IFFT_Data = ifft(STS(1:size_of_FFT,1),size_of_FFT);
STS(1:size_of_FFT+cp_length,1) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
STS(1) = STS(1)*0.5;
STS(size_of_FFT+cp_length) = STS(size_of_FFT+cp_length)*0.5;
sts = [STS;STS];


% Long Preamble Field
LTS = [0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,...
    1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,...
    1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,...
    1,1,0,0,0,0,0,];
LTS = LTS.';
IFFT_Data = ifft(LTS(1:size_of_FFT,1),size_of_FFT);
LTS(1:size_of_FFT+cp_length,1) = [IFFT_Data(size_of_FFT - cp_length + 1: size_of_FFT);IFFT_Data];
LTS(1) = LTS(1)*0.5;
LTS(size_of_FFT+cp_length) = LTS(size_of_FFT+cp_length)*0.5;
lts = [LTS;LTS];

TX = [sts;lts;TX];

    
write_complex_binary(TX,'TX.bin');
