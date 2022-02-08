from asyncio.subprocess import DEVNULL
import numpy as np
import matlab.engine
import subprocess

eng = matlab.engine.start_matlab()

# TX Transmission 2
eng.TX_Feedback_Encoder2(nargout=0)
print('\nTX Encoder 2 generated\n')
subprocess.Popen("python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/flow_graph_TX.py", shell=True,stdout=DEVNULL,stderr=DEVNULL)
print('TX Transmission 2 starts')
