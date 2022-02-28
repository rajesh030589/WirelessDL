function RX_Feedback_Encoder1()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));

    run('Parameters_feedback.m');

    %Frame Data
    B1_Output = zeros(total_msg_symbols, no_of_frames);
    TX_Out = zeros(total_no_of_samples, no_of_frames);
    Y1_Input = open('Feedback_Files/Y1_Output.mat');
    Y1_Input = Y1_Input.Y1_Output;

    for n_frame = 1:no_of_frames

        sts = [];
        lts = [];

        % Modulation
        if strcmp(mod_type, "NN")
            encoder_data = Y1_Input(:, n_frame);
            save('Data_Files/RX_Encoded1.mat', 'encoder_data');
            system('python3 Imp_Functions/RX_NN_Encoder1.py');
            mod_symbols = open("Data_Files/RX_Modulated1.mat");
            mod_symbols = double(mod_symbols.output);
        else
            mod_symbols = ActiveDecoder(0, 0, Y1_Input(:, n_frame), 0, 0, 1);
        end

        % Signal Frame containing the frame number
        sig_symb = zeros(no_signal_symbols, 1);

        F = de2bi(n_frame - 1, 6);

        for i = 1:6
            sig_symb((i - 1) * 8 + 1:i * 8) = F(i) * ones(8, 1);
        end

        sig_symb = qammod(sig_symb, 2, 'InputType', 'bit', 'UnitAveragePower', true);
        % Subcarrier Allocation
        P = open('Pilot_matrix.mat');
        P = P.PILOT;
        Pilots = zeros(no_of_pilot_carriers, total_ofdm_symbols_per_frame);

        A = zeros(size_of_FFT, total_ofdm_symbols_per_frame);
        k = 1;
        l = 1;

        for i = 1:total_ofdm_symbols_per_frame

            for j = subcarrier_locations

                if any(pilot_carriers(:) == j)
                    A(j, i) = pilot_values(j, 1) * P(1 + mod(i - 1, 127));
                else

                    if i <= no_signal_symbols
                        A(j, i) = sig_symb(l);
                        l = l + 1;
                    else
                        A(j, i) = tx_gain * mod_symbols(k);
                        k = k + 1;
                    end

                end

            end

            Pilots(:, i) = A(pilot_carriers, i);
        end

        save('Imp_Files/Pilots.mat', 'Pilots');

        % IFFT to generate tx symbols
        for i = 1:total_ofdm_symbols_per_frame
            IFFT_Data = ifft(fftshift(A(1:size_of_FFT, i)), size_of_FFT);
            A(1:size_of_FFT + cp_length, i) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
        end

        TX = A(:);

        % Normalize the modulated data Power
        % TX = TX .* (.8 / (max(max(abs(real(TX))), max(abs(imag(TX))))));

        % Short Preamble Field
        STS = open('STS.mat');
        STS = STS.STS;
        IFFT_Data = ifft(fftshift(STS(1:size_of_FFT, 1)), size_of_FFT);
        sts(1:size_of_FFT + cp_length, 1) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
        sts(size_of_FFT + cp_length) = sts(size_of_FFT + cp_length);
        sts = [sts(1) * 0.5; sts(2:end); sts; sts(1) * 0.5];

        % Long Preamble Field
        LTS = open('LTS.mat');
        LTS = LTS.LTS;
        IFFT_Data = ifft(fftshift(LTS(1:size_of_FFT, 1)), size_of_FFT);
        lts(1:size_of_FFT + cp_length, 1) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
        lts = [lts(1) * 0.5; lts(2:end); lts; lts(1) * 0.5];

        % Concatenate the lts and sts to the transmit data
        TX = [sts(1:end - 1); sts(end) + lts(1); lts(2:end - 1); lts(end) + TX(1); TX(2:end)];

        % Frame Data

        B1_Output(:, n_frame) = mod_symbols;
        TX_Out(:, n_frame) = TX;
        Data_start(n_frame) = (n_frame - 1) * length(TX) + 320 + 1;
    end

    % Save Frame Data
    save('Feedback_Files/B1_Output.mat', 'B1_Output');
    save('Data_Files/TX_Out.mat', 'TX_Out');
    % Write to Transmitter
    TX = TX_Out(:);
    write_complex_binary(TX, 'TX.bin');
end
