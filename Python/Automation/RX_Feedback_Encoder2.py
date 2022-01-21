import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time
import subprocess
from asyncio.subprocess import DEVNULL

eng = matlab.engine.start_matlab()
eng.RX_Feedback_Encoder2(nargout=0)
print("RX Encoder 2 Output generated\n")
subprocess.Popen(
    "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flow_graph_TX.py",
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)
print("RX Transmission 2 started\n")
