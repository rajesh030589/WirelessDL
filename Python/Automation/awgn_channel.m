function awgn_channel(ch_type)
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    clearvars -except ch_type
    SNR = 80;
    SNR = 10^(SNR / 10);
    TX = read_complex_binary("TX.bin");

    if strcmp(ch_type, "pass")
        RX = [TX; TX; TX];
    elseif strcmp(ch_type, "complex_noise")
        Z = (1 / sqrt(SNR)) * (randn(length(TX) * 3, 1) + 1j * randn(length(TX) * 3, 1));
        RX = [TX; TX; TX] +Z;
    end

    write_complex_binary(RX, 'RX.bin');
end
