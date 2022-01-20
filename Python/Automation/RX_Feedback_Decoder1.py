import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time

eng = matlab.engine.start_matlab()
N_captures = 3
eng.frame_capture_2_1(nargout=0)
for i in range(N_captures):
    frame_capture = sio.loadmat("frame_capture.mat")
    frame_capture = frame_capture["frame_capture"]
    if np.count_nonzero(frame_capture) == len(frame_capture):
        break
    os.system("python3 flow_graph_RX.py")
    print("Capture done")
    eng.RX_Feedback_Decoder1(nargout=0)
