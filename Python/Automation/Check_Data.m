function Check_Data()
    addpath('/home/rajesh/WirelessDL/Python/Automation/Imp_Files');
    addpath('/home/rajesh/WirelessDL/Python/Automation/Imp_Functions');
    run('Parameters.m');

    DataIn = load('Data_Files/DataIn.mat');
    DataIn = DataIn.DataIn;

    DataOut = load('Data_Files/DataOut.mat');
    DataOut = DataOut.DataOut;

    [m, n, p] = size(DataOut);
    bit_err = 0;

    for i = 1:m
        decoded_data = Decoder(squeeze(DataOut(i, :, :)), dec_type, no_of_blocks, block_len);
        bit_err = bit_err + biterr(decoded_data, squeeze(DataIn(i, :, :))) / encoded_no_bits;
    end

    fprintf('The average biterr of data %d x %d x %d is %.4f\n', m, n, p, bit_err / m)
end
