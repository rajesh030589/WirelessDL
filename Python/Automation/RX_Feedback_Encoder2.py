import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time

eng = matlab.engine.start_matlab()
eng.TX_Feedback_Encoder2(nargout=0)
