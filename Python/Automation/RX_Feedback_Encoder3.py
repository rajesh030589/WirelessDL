import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time

eng = matlab.engine.start_matlab()
print("Final Decoding starts\n")
eng.RX_Feedback_Encoder3(nargout=0)
print("\nProcess Ended\n")
