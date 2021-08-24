clearvars;
clc;
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_frames = 30;
enc_type = 'convolutional'; %'turbo'; %'convolutional'
dec_type = 'MAP'; %'convolutional' 'MAP'
block_len = 10; % Convolutional Code Parameter

term_bits = 0; % 4 for turbo encoder
rate = 1/2; %1/3
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
no_signal_symbols = 1;
no_preamble_symbols = 4;
mod_order = 4;

no_of_data_blocks = 2^block_len;
no_of_big_blocks = 10;
no_of_blocks = no_of_data_blocks * no_of_big_blocks;
encoded_no_bits = block_len * no_of_blocks;
coded_block_len = (block_len + term_bits) / rate;
no_encoder_out_bits = coded_block_len * no_of_blocks;
bit_per_symbol = log2(mod_order);
extra_bits = bit_per_symbol * no_of_subcarriers - rem(no_encoder_out_bits, bit_per_symbol * no_of_subcarriers);
total_no_bits = no_encoder_out_bits + extra_bits;
total_msg_symbols = total_no_bits / bit_per_symbol;
no_of_ofdm_symbols_per_frame = total_msg_symbols / no_of_subcarriers;
total_ofdm_symbols_per_frame = no_of_ofdm_symbols_per_frame + no_signal_symbols;
signal_field_symbols = no_signal_symbols * no_of_subcarriers;
preamble_len = no_preamble_symbols * (size_of_FFT + cp_length);
total_no_of_data_samples = total_ofdm_symbols_per_frame * (size_of_FFT + cp_length);
total_no_of_samples = total_no_of_data_samples + preamble_len;

no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
