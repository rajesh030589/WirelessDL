import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time

eng = matlab.engine.start_matlab()
N_captures = 10

eng.frame_capture_1_3(nargout=0)
for i in range(N_captures):
    frame_capture = sio.loadmat("frame_capture.mat")
    frame_capture = frame_capture["frame_capture"]
    if np.count_nonzero(frame_capture) == len(frame_capture):
        break
    os.system("python3 flow_graph_RX.py")
    eng.TX_Feedback_Decoder2(nargout=0)
