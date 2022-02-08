import numpy as np
import matlab.engine
import subprocess
from asyncio.subprocess import DEVNULL
import signal
import time
import scipy.io as sio
import time

eng = matlab.engine.start_matlab()
N_captures = 3
print("\nTX Reception 2 starts")
eng.frame_capture_1_3(nargout=0)
for i in range(N_captures):
    frame_capture = sio.loadmat("frame_capture.mat")
    frame_capture = frame_capture["frame_capture"]
    if np.count_nonzero(frame_capture) == len(frame_capture):
        break
    print('\nCapture : ',i+1,' starts ...\n')
    subprocess.run("python3 flow_graph_RX.py", shell=True, stdout=DEVNULL,stderr=DEVNULL)
    print('Capture Done\n')
    eng.TX_Feedback_Decoder2(nargout=0)
