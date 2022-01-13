function frame_capture_2_1()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_feedback.m');

    frame_capture = zeros(no_of_frames, 1);
    save('frame_capture.mat', 'frame_capture');

    Y1_Output = zeros(total_msg_symbols, no_of_frames);
    save('Feedback_Files/Y1_Output.mat', 'Y1_Output');
end
