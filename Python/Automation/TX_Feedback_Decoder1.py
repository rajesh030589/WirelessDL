from sys import stderr
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
print("\nTX Reception 1 starts")
eng.frame_capture_1_2(nargout=0)
for i in range(N_captures):
    frame_capture = sio.loadmat("frame_capture.mat")
    frame_capture = frame_capture["frame_capture"]
    if np.count_nonzero(frame_capture) == len(frame_capture):
        break
    print('\nCapture : ',i+1,' starts ...\n')
    subprocess.run("python3 flow_graph_RX.py", shell=True, stdout=DEVNULL,stderr=DEVNULL)
    time.sleep(3)
    print('Capture Done\n')
    eng.TX_Feedback_Decoder1(nargout=0)
