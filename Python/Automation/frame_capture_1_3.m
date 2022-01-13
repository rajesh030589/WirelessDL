function frame_capture_1_3()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    frame_capture = zeros(no_of_frames, 1);
    save('frame_capture.mat', 'frame_capture');

    Z2_Output = zeros(total_msg_symbols, no_of_frames);
    save('Feedback_Files/Z2_Output.mat', 'Z2_Output');
end
