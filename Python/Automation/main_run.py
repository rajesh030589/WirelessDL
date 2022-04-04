import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time
import subprocess
import matplotlib.pyplot as plt

forward_tx_gain = 5
forward_rx_gain = 10

string1 = "python3 TX_Feedback.py -tx_gain " + str(forward_tx_gain)
string2 = "python3 RX_Feedback.py -rx_gain " + str(forward_rx_gain)


def run_scheme(alpha):

    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    subprocess.call(
        string1 + " -dev_type encoder -num 1 -scale " + str(alpha), shell=True
    )
    time.sleep(3)
    subprocess.call(string2 + " -dev_type decoder -num 1", shell=True)

    Y1_Output = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Y1_Output.mat"
    )
    Y1_Output = Y1_Output["YL"]

    D1 = np.zeros((len(Y1_Output), 1))
    D1 = np.where(Y1_Output < 0, D1, 1)
    time.sleep(3)

    subprocess.call(string2 + " -dev_type decoder -num 1", shell=True)

    Y1_Output = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Y1_Output.mat"
    )
    Y1_Output = Y1_Output["YL"]

    D2 = np.zeros((len(Y1_Output), 1))
    D2 = np.where(Y1_Output < 0, D2, 1)

    time.sleep(3)
    subprocess.call(string2 + " -dev_type decoder -num 1", shell=True)

    Y1_Output = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Y1_Output.mat"
    )
    Y1_Output = Y1_Output["YL"]

    D3 = np.zeros((len(Y1_Output), 1))
    D3 = np.where(Y1_Output < 0, D3, 1)

    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    D = np.zeros((len(D1), 1))
    D = np.where((D1 + D2 + D3) < 1.5, D, 1)

    Bit_Input = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Bit_Input.mat"
    )

    Bit_Input = Bit_Input["Bit_Input"]

    BER1 = 1 - sum(abs(D - Bit_Input)) / len(Bit_Input)

    print(BER1)

    return BER1


alpha_list = np.linspace(18, 6, 7, endpoint=True)
BER = np.zeros((len(alpha_list), 1))

for i in range(len(alpha_list)):
    alpha = alpha_list[i]

    BER1 = run_scheme(alpha)

    BER[i, 0] = BER1

a = np.random.randint(0, 100000)
Data = {"BER": BER, "Alpha": alpha_list}
sio.savemat("Datasets/Data_" + str(a) + ".mat", Data)
