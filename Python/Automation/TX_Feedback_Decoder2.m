function TX_Feedback_Decoder2()

    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    % Extraction of the received data
    RX = read_complex_binary('RX.bin');
    % Packet Detection

    % STS Packet Detection
    st_id_list = STS_detect(RX, total_no_of_samples);

    Z2_Output = open("Feedback_Files/Z2_Output.mat");
    Z2_Output = Z2_Output.Z2_Output;
    frame_capture = open("frame_capture.mat");
    frame_capture = frame_capture.frame_capture;
    err = 0;
    err_count = 0;

    for n_detect = 1:length(st_id_list)

        % Detected Packet
        st_id = st_id_list(n_detect);

        % Symbol Alignment
        lt_id = LTS_detect(RX, st_id, size_of_FFT);

        sts_start_id = lt_id - 160;
        sts_end_id = lt_id - 1;

        lts1_start_id = lt_id;
        lts1_end_id = lt_id + 79;

        lts2_start_id = lt_id + 80;
        lts2_end_id = lt_id + 159;

        data_start_id = lt_id + 160;
        data_end_id = data_start_id + total_no_of_data_samples - 1;

        % Packet Extraction
        sts = RX(sts_start_id:sts_end_id);
        lts1 = RX(lts1_start_id:lts1_end_id);
        lts2 = RX(lts2_start_id:lts2_end_id);
        y = RX(data_start_id:data_end_id);

        % Coarse Frequency offset
        alpha_ST = (1/16) * angle(sum(conj(sts(1:144)) .* sts(17:160)));

        lts1 = lts1 .* exp(-1j .* (0:79)' * alpha_ST);
        lts2 = lts2 .* exp(-1j .* (80:159)' * alpha_ST);

        % Data Arranged
        LTS = open('LTS.mat');
        LTS = LTS.LTS;
        LTS1 = fftshift(fft(lts1(cp_length + 1:size_of_FFT + cp_length, 1)));
        LTS2 = fftshift(fft(lts2(cp_length + 1:size_of_FFT + cp_length, 1)));

        y = reshape(y, size_of_FFT + cp_length, total_ofdm_symbols_per_frame);
        Y = zeros(size_of_FFT, total_ofdm_symbols_per_frame);

        for i = 1:total_ofdm_symbols_per_frame
            Y(:, i) = fftshift(fft(y(cp_length + 1:size_of_FFT + cp_length, i)));
        end

        Y = Y(:);
        % Fine Phase Offset Correction
        alpha_LT = (1/64) * angle(sum(conj(LTS1) .* LTS2));
        Y = Y .* exp(-1j .* ((1:length(Y))' * alpha_LT));
        Y = reshape(Y, size_of_FFT, []);

        % Channel Estimation
        H = zeros(size_of_FFT, 1);

        for j = subcarrier_locations
            H(j) = 0.5 * (LTS1(j) + LTS2(j)) * sign(LTS(j));
        end

        % Channel Equalization and Phase Offset Correction
        Pilots = open('Pilots.mat');
        Pilots = Pilots.Pilots;
        detected_signal_symbols = zeros(signal_field_symbols, 1);
        detected_symbols = zeros(total_msg_symbols, 1);
        l = 1;
        k = 1;

        for i = 1:total_ofdm_symbols_per_frame
            theta = angle(sum(conj(Y(pilot_carriers, i)) .* Pilots(:, i) .* H(pilot_carriers, 1)));

            for j = subcarrier_locations

                if ~(any(pilot_carriers(:) == j))

                    if i <= no_signal_symbols
                        detected_signal_symbols(l, 1) = (Y(j, i) / H(j)) * exp(1j * theta);
                        l = l + 1;
                    else
                        detected_symbols(k, 1) = (Y(j, i) / H(j)) * exp(1j * theta);
                        k = k + 1;
                    end

                end

            end

        end

        % SNR Estimate
        h = abs(ifft(fftshift(H)));
        snr_estimate = 20 * log10(abs(h(1))) - 20 * log10(mean(abs(h(10:end))));

        % Frame Detection
        demod_first_symbol = qamdemod(detected_signal_symbols, 2, 'OutputType', 'bit', 'UnitAveragePower', true);

        decoded_frame = zeros(6, 1);

        for i = 1:6
            decoded_frame(i, 1) = (sum(demod_first_symbol(8 * (i - 1) + 1:8 * i)) > 3);
        end

        frame_num = bi2de(decoded_frame.');

        if frame_num > no_of_frames - 1
            continue;
        end

        % Decoder
        decoded_data = qamdemod(detected_symbols, mod_order, 'OutputType', 'bit', 'UnitAveragePower', true);
        decoded_data = decoded_data(1:end - extra_bits);

        data_to_encode = open('Feedback_Files/Bit_Input.mat');
        data_to_encode = data_to_encode.Bit_Input;
        data_to_encode = data_to_encode(:, frame_num + 1);
        bit_err = biterr(decoded_data, data_to_encode) / encoded_no_bits;

        fprintf("Frame: %d  SNR:  %.2f  BER: %1.4f\n", frame_num, snr_estimate, bit_err);

        if snr_estimate > 14
            Z2_Output(:, frame_num + 1) = real(detected_symbols/1.7);
            frame_capture(frame_num + 1, 1) = 1;
            save("frame_capture.mat", "frame_capture")

            if nnz(frame_capture) == no_of_frames
                fprintf("All frames captured\n");
                break;
            end

        end

    end
    save('Feedback_Files/Z2_Output.mat', 'Z2_Output')
end

function st_id = STS_detect(RX, total_no_of_samples)

    window_size = 64;
    mean_size = 64;
    corr_th = 0.32;
    count_cn = 90;

    L = length(RX) - total_no_of_samples;

    CORR = zeros(L, 1);

    for k = 1:L
        CORR(k) = (sum(RX(k:k + window_size - 1) .* conj(RX(k + 16:k + 16 + window_size - 1))));
        CORR(k) = CORR(k) / (sum(RX(k:k + window_size - 1) .* conj(RX(k:k + window_size - 1))));
    end

    M = movmean(CORR, mean_size);

    st_id = [];
    count = 0;
    trigger = 0;

    for i = 1:length(M)

        if (abs(M(i)) > 0.4 && trigger == 0) || i < 546119
            continue;
        else
            trigger = 1;
        end

        if abs(M(i)) > corr_th
            count = count + 1;
        else
            count = 0;
        end

        if count > count_cn
            st_id = [st_id i];
            count = 0;
        end

        if length(st_id) > 34
            break;
        end

    end

end

function lt_id = LTS_detect(RX, st_id, size_of_FFT)

    lt_id = 0;

    % LTS Symbol Alignment
    L = zeros(200, 1);

    LTS = open('LTS.mat');
    LTS = LTS.LTS;
    lts = ifft(fftshift(LTS(1:size_of_FFT, 1)), size_of_FFT);

    for j = 1:200
        L(j) = sum(RX(st_id + j - 1:st_id + j - 1 +63) .* conj(lts));
    end

    [~, lt_id1] = max(abs(L));
    L(lt_id1) = 0;
    [~, lt_id2] = max(abs(L));
    lt_id = min(lt_id1, lt_id2);

    lt_id = st_id + lt_id - 1 - 16;
end
