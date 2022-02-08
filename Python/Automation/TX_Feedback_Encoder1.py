from asyncio.subprocess import DEVNULL
import numpy as np
import matlab.engine
import subprocess
import time

eng = matlab.engine.start_matlab()

# TX Transmission 1
eng.TX_Feedback_Encoder1(nargout=0)
print('\nPROCESS BEGINS\n')
print('TX Encoder 1 generated\n')
subprocess.Popen("python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flow_graph_TX.py", shell=True,stdout=DEVNULL,stderr=DEVNULL)
time.sleep(3)
print('TX Transmission 1 starts')