import argparse
import numpy as np
import matlab.engine
import time
import scipy.io as sio
import time
import subprocess
from asyncio.subprocess import DEVNULL


eng = matlab.engine.start_matlab()
forward_rx_gain = 12

cmd_string = (
    "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/RX_flow_graph_RX.py -rx_gain "
    + str(forward_rx_gain)
    + " -file_path "
    + "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/RX.bin"
)
subprocess.Popen(
    cmd_string,
    shell=True,
    stdout=DEVNULL,
    stderr=DEVNULL,
)

time.sleep(3)
print("Capture done")
eng.RX_Measure_Noise(nargout=0)