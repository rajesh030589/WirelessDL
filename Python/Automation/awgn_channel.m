function awgn_channel(SNR)

    run('Parameters.m');
    SNR = 25;
    SNR = 10^(SNR / 10);
    TX = read_complex_binary('TX.bin');
    Z = (1 / sqrt(SNR)) * (randn(length(TX), 1) + 1j * randn(length(TX), 1));
    % RX = TX + Z;
    RX = TX;

    write_complex_binary(RX, 'RX.bin');
end
