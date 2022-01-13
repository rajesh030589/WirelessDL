import numpy as np
import matlab.engine
import os

eng = matlab.engine.start_matlab()

# TX Transmission 2
print("TX -> RX 2")
eng.TX_Feedback_Encoder3(nargout=0)
os.system("python3 flow_graph_TX.py")
