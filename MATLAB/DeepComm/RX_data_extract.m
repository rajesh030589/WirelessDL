clc;
clearvars;
close all;

% Parameters
no_of_ofdm_symbols = 800;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 51;
total_symbols = no_of_ofdm_symbols*no_of_subcarriers;
mod_order = 4;
bit_per_symbol = log2(mod_order);
total_no_bits = total_symbols*bit_per_symbol;
encoded_no_bits = (total_no_bits - 12)/3;
total_no_of_samples = no_of_ofdm_symbols*(size_of_FFT+ cp_length);

% Extraction of the received data
stringTX = strcat("TX.bin");
stringRX = strcat("RX.bin");

TX = read_complex_binary(stringTX);
RX = read_complex_binary(stringRX);

k = 1;
pow = zeros(length(RX),1);
pkt_received = 0;
while length(RX) > total_symbols + 320
    
    
    % Power Trigger
    pow(k) = abs(RX(k)*conj(RX(k)));
    if  k<50000
        k = k+1;
        continue;
    elseif pow(k) - pow(k-1000) < 0.012
        k = k+1;
        continue;
    end
        
    
    % STS Packet Detection
    window_size = 16;
    count = 0;
    i = k;
    while count < 10
        corr = (sum(RX(i:i+window_size - 1).*conj(RX(i+16:i+16+window_size - 1))));
        corr = corr/(sum(RX(i:i+window_size - 1).*conj(RX(i:i+window_size - 1))));
        if corr > 1.001
            count = count + 1;
        else
            count = 0;
        end
        i = i + 1;
    end
    st_id = i;
    
    % LTS Symbol Alignment
    L = zeros(200,1);
    lts = open('LTS.mat');
    lts = lts.lts;
    lts = lts(cp_length+1:cp_length + size_of_FFT);
    for j = 1:200
        L(j) = sum(RX(st_id+j - 1:st_id+j - 1 +63).*conj(lts));
    end
    [~,lt_id1] = max(L);
    L(lt_id1) = 0;
    [~,lt_id2] = max(L);
    lt_id = min(lt_id1,lt_id2);
    
    lt_id = st_id + lt_id - 1 - 32;
    
    sts_start_id = lt_id-160;
    sts_end_id = lt_id - 1;
    
    lts1_start_id = lt_id;
    lts1_end_id = lt_id+ 79;
    
    
    lts2_start_id = lt_id + 80;
    lts2_end_id = lt_id+ 159;
    
    data_start_id = lt_id + 160;
    data_end_id = data_start_id + total_no_of_samples - 1;
    
    % Packet Extraction
    pkt_received = pkt_received + 1;
    packet.num = pkt_received;
    packet.STS = RX(sts_start_id:sts_end_id);
    packet.LTS1 = RX(lts1_start_id:lts1_end_id);
    packet.LTS2 = RX(lts2_start_id:lts2_end_id);
    packet.data = RX(data_start_id:data_end_id);
    
    fprintf('Packet Receieved %d\n',pkt_received);
    
    % Coarse Frequency offset
    sts = packet.STS;
    alpha = (1/16)*angle(sum(conj(sts(1:144)).*sts(17:160)));
    
    lts1 = packet.LTS1;
    lts1 = lts1.*exp(-1j.*[0:79]'*alpha);
    
    lts2 = packet.LTS2;
    lts2 = lts2.*exp(-1j.*[80:159]'*alpha);     
    
    y = packet.data;
    y = y.*exp(-1j.*[160:159+total_no_of_samples]'*alpha);
    
    fprintf('Frequency Corrected of Packet %d\n',pkt_received);
    
    y = reshape(y, size_of_FFT_cp_length , no_of_ofdm_symbols);
    Y = 
    
    RX(1:data_end_id) = [];
    
    
end


%
% %
% % % Plot the received data frequency response
% % figure();
% % plot(10*log10(abs(fftshift(pwelch(resample(RX(1:1e6),1,1),64)))))
% % xlabel('Samples')
% % ylabel('Mag Response');
% % title('Received Signal Frequency Response')
% % grid on;
%
%
% % Correlate and extract samples
% x = TX;
% y = acorr(TX,RX);
%
% % %Frequency Offset Correction
% % S = 0;
% % for i = 1:144
% %     S = S + y(i,1)'*y(i+16);
% % end
% % angleS = angle(S);
% % alpha = angleS/16;
% %
% % for i = 161:length(x)
% %     y(1) = y(161,1)*exp(-1j*(i - 160-1)*alpha);
% % end
% x = x(161:end);
% % y = y(1:length(x));
% y = y(161:end);gm
%
% % Channel Estimation
% h = Equalization_NLMS(x,y);
% H = fftshift(fft(h));
%
% % % Plot the time domain channel response
% % figure()
% % plot(abs(h))
% % xlabel('Samples')
% % ylabel('Mag Response');
% % title('Channel Impulse Response')
% % grid on;
% %
% %
% % % Plot the frequency domain channel response
% % figure()
% % plot(abs(H))
% % xlabel('Samples')
% % ylabel('Mag Response');
% % title('Channel Frequency Response')
% % grid on;
% %
%
%
%
% % Remove CP from the received samples
% y = reshape(y, size_of_FFT+cp_length, no_of_ofdm_symbols);
%
% % Take the FFT of the reveived time samples
% Y = zeros(size_of_FFT, no_of_ofdm_symbols);
% for i = 1:no_of_ofdm_symbols
%     Y(:,i) = fftshift(fft(y(cp_length+1:size_of_FFT+cp_length,i)));
% end
%
% % Extract from the subcarriers
% detected_symbols = zeros(total_symbols,1);
% k = 1;
% for i = 1:no_of_ofdm_symbols
%     for j = [7:32 34:58]
%         detected_symbols(k,1) = Y(j,i)/H(j,1);
%         k = k+1;
%     end
% end
%
% % TX Data
% mod_symbols = open('mod_symbols.mat');
% mod_symbols = mod_symbols.mod_symbols;
% %
% % figure()
% % subplot(1,2,1)
% % scatter(real(mod_symbols), imag(mod_symbols));
% % xlabel('Real')
% % ylabel('Imaginary')
% % title('TX Data')
% %
% % subplot(1,2,2)
% % scatter(real(detected_symbols), imag(detected_symbols));
% % xlabel('Real')
% % ylabel('Imaginary')
% % title('RX Data')
% %
% % sgtitle('Constellation')
%
%
% figure()
% hold on;
% % for i = 1: 10000
% %     if mod_symbols(i) == 1
% %         scatter(real(detected_symbols(i)), imag(detected_symbols(i)),'b');
% %     else
% %         scatter(real(detected_symbols(i)), imag(detected_symbols(i)),'r');
% %     end
% % end
%
% one_bit = detected_symbols(mod_symbols==1);
% zero_bit = detected_symbols(mod_symbols==-1);
% scatter(real(one_bit),imag(one_bit),'b')
% scatter(real(zero_bit),imag(zero_bit),'r')
% xlabel('Real')
% ylabel('Imaginary')
% title('RX Data')
% grid on;
% sgtitle('Constellation')