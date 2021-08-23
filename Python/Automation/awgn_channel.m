function awgn_channel(ch_type)
    addpath('/home/rajesh/WirelessDL/Python/Automation/Imp_Files');
    addpath('/home/rajesh/WirelessDL/Python/Automation/Imp_Functions');
    clearvars -except ch_type
    SNR = 22;
    SNR = 10^(SNR / 10);
    TX = read_complex_binary("TX.bin");

    if strcmp(ch_type, "pass")
        RX = TX;
    elseif strcmp(ch_type, "complex_noise")
        Z = (1 / sqrt(SNR)) * (randn(length(TX), 1) + 1j * randn(length(TX), 1));
        RX = TX + Z;
    end

    write_complex_binary(RX, 'RX.bin');
end
