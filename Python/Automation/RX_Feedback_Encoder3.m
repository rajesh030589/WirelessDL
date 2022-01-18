function RX_Feedback_Encoder3()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));

    run('Parameters_feedback.m');

    Bit_Input = open('Feedback_Files/Bit_Input.mat');
    Bit_Input = Bit_Input.Bit_Input;
    Y3_Input = open('Feedback_Files/Y3_Output.mat');
    Y3_Input = Y3_Input.Y3_Output;
    B2_Input = open('Feedback_Files/B2_Output.mat');
    B2_Input = B2_Input.B2_Output;
    B3_Output = zeros(total_msg_symbols, no_of_frames);
    BB3_Output = zeros(total_msg_symbols, no_of_frames);

    for n_frame = 1:no_of_frames

        % Modulation
        B3 = ActiveDecoder(0, B2_Input(:, n_frame), 0, 0, Y3_Input(:, n_frame), 3);

        BB3 = 1 ./ (1 + exp(-B3));
        decoded_data = (BB3 > 0.5);
        data_to_encode = Bit_Input(:, n_frame);
        bit_err = biterr(decoded_data, data_to_encode) / encoded_no_bits;

        fprintf("Frame: %d  BER: %1.4f\n", n_frame, bit_err);

        B3_Output(:, n_frame) = B3;
        BB3_Output(:, n_frame) = BB3;
    end

    save('Feedback_Files/B3_Output.mat', 'B3_Output');
    save('Feedback_Files/BB3_Output.mat', 'BB3_Output');
end
