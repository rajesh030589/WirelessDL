function frame_capture_1_2()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    frame_capture = zeros(no_of_frames, 1);
    save('frame_capture.mat', 'frame_capture');

    Z1_Output = zeros(total_msg_symbols, no_of_frames);
    save('Feedback_Files/Z1_Output.mat', 'Z1_Output');
end
