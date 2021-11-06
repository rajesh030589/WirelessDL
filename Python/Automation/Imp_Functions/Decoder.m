function Decoded_data = Decoder(Demod_data, type, no_of_blocks, block_length)

    if strcmp(type, 'turbo')
        TRELLIS = poly2trellis(4, [13, 15], 13);
        turboDec = comm.TurboDecoder('TrellisStructure', TRELLIS, 'InterleaverIndicesSource', 'Input port', 'NumIterations', 6);
        Interleaver = open('Interleaver.mat');
        Interleaver = Interleaver.Interleaver;

        Decoded_data = zeros(block_length, no_of_blocks);

        for i = 1:no_of_blocks
            %             intrlvrInd = Interleaver(:, i);
            % intrlvrInd = [22, 20, 25, 4, 10, 15, 28, 11, 18, 29, 27, ...
            %             35, 37, 2, 39, 30, 34, 16, 36, 8, 13, 5, 17, 14, 33, 7, ...
            %             32, 1, 26, 12, 31, 24, 6, 23, 21, 19, 9, 38, 3, 0] + 1;
            intrlvrInd = [0, 33, 14, 47, 28, 61, 42, 75, 56, 89, 70, 103, 84, 13, 98, 27, 8, 41, 22, 55, 36, 69, 50, 83, 64, 97, 78, 7, 92, 21, ...
                    2, 35, 16, 49, 30, 63, 44, 77, 58, 91, 72, 1, 86, 15, 100, 29, 10, 43, 24, 57, 38, 71, 52, 85, 66, 99, 80, 9, 94, 23, 4, 37, 18, 51, ...
                        32, 65, 46, 79, 60, 93, 74, 3, 88, 17, 102, 31, 12, 45, 26, 59, 40, 73, 54, 87, 68, 101, 82, 11, 96, 25, 6, 39, 20, 53, 34, 67, 48, ...
                        81, 62, 95, 76, 5, 90, 19] +1;
            demod_data = Demod_data(:, i);
            decoded_data = step(turboDec, demod_data, intrlvrInd);
            Decoded_data(:, i) = decoded_data;
        end

    elseif strcmp(type, 'MAP')

        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % BCJR Decoder
        hAPPDec = comm.APPDecoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated', ...
        'Algorithm', 'True APP', 'CodedBitLLROutputPort', true);

        Decoded_data = zeros(block_length, no_of_blocks);

        for i = 1:no_of_blocks
            demod_data = Demod_data(:, i);
            ll0 = zeros(block_length, 1);
            llr = step(hAPPDec, ll0, demod_data);
            data = (llr > 0); % MAP decoded bits
            decoded_data = data(1:block_length);
            Decoded_data(:, i) = decoded_data;
        end

    end
