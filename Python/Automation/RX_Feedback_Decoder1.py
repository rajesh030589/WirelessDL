import numpy as np
import matlab.engine
import subprocess
import signal
import time
import scipy.io as sio
import time
from asyncio.subprocess import DEVNULL

eng = matlab.engine.start_matlab()
N_captures = 3
eng.frame_capture_2_1(nargout=0)
for i in range(N_captures):
    frame_capture = sio.loadmat("frame_capture.mat")
    frame_capture = frame_capture["frame_capture"]
    if np.count_nonzero(frame_capture) == len(frame_capture):
        break
    print("capture :", i + 1, "...\n")
    subprocess.run(
        "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flow_graph_RX.py",
        shell=True,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    print("Capture done\n")
    eng.RX_Feedback_Decoder1(nargout=0)
