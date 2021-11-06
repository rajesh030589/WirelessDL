function Encoded_data = Encoder(data_input, type, no_of_blocks, block_length, code_rate)

    if strcmp(type, 'turbo')
        TRELLIS = poly2trellis(4, [13, 15], 13);
        turboEnc = comm.TurboEncoder('TrellisStructure', TRELLIS, 'InterleaverIndicesSource', 'Input port');
        coded_block_length = block_length / code_rate + 12;
        Encoded_data = zeros(coded_block_length, no_of_blocks);

        for i = 1:no_of_blocks
            X = data_input(:, i);
            % intrlvrInd = [22, 20, 25, 4, 10, 15, 28, 11, 18, 29, 27, ...
            %             35, 37, 2, 39, 30, 34, 16, 36, 8, 13, 5, 17, 14, 33, 7, ...
            %             32, 1, 26, 12, 31, 24, 6, 23, 21, 19, 9, 38, 3, 0] + 1;
            intrlvrInd = [0, 33, 14, 47, 28, 61, 42, 75, 56, 89, 70, 103, 84, 13, 98, 27, 8, 41, 22, 55, 36, 69, 50, 83, 64, 97, 78, 7, 92, 21, ...
                    2, 35, 16, 49, 30, 63, 44, 77, 58, 91, 72, 1, 86, 15, 100, 29, 10, 43, 24, 57, 38, 71, 52, 85, 66, 99, 80, 9, 94, 23, 4, 37, 18, 51, ...
                        32, 65, 46, 79, 60, 93, 74, 3, 88, 17, 102, 31, 12, 45, 26, 59, 40, 73, 54, 87, 68, 101, 82, 11, 96, 25, 6, 39, 20, 53, 34, 67, 48, ...
                        81, 62, 95, 76, 5, 90, 19] +1;
            encoded_data = step(turboEnc, X, intrlvrInd);
            Encoded_data(:, i) = encoded_data;
        end

    elseif strcmp(type, 'convolutional')
        coded_block_length = block_length / code_rate;
        Encoded_data = zeros(coded_block_length, no_of_blocks);
        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % Convolutional Encoder
        hConEnc = comm.ConvolutionalEncoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated');

        for i = 1:no_of_blocks
            X = data_input(:, i);
            encoded_data = hConEnc(X);
            Encoded_data(:, i) = encoded_data;
        end

    end

end
