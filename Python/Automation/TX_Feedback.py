import argparse
import numpy as np
import matlab.engine
import time
import scipy.io as sio
import time
import subprocess
from asyncio.subprocess import DEVNULL


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dev_type", type=str, default="encoder")
    parser.add_argument("-num", type=int, default=3)
    parser.add_argument("-rx_gain", type=int, default=15)
    parser.add_argument("-tx_gain", type=int, default=15)
    parser.add_argument("-n_captures", type=int, default=3)
    parser.add_argument(
        "-rx_filename",
        type=str,
        default="/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/RX.bin",
    )
    parser.add_argument(
        "-tx_filename",
        type=str,
        default="/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/TX.bin",
    )
    args = parser.parse_args()

    return args


args = get_args()

if args.dev_type == "encoder":
    eng = matlab.engine.start_matlab()

    # TX Transmission 1
    eng.TX_Feedback_Encoder(args.num,nargout=0)
    print('\nPROCESS BEGINS\n')
    print('TX Encoder 1 generated')
    cmd_string = (
        "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/TX_flow_graph_TX.py -tx_gain "
        + str(args.tx_gain)
        + " -file_path "
        + args.tx_filename
    )
    subprocess.Popen(
        cmd_string,
        shell=True,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    time.sleep(3)
    print('TX Transmission 1 starts')

elif args.dev_type == "decoder":
    eng = matlab.engine.start_matlab()
    N_captures = args.n_captures
    print("TX Reception 1 starts")
    eng.frame_capture(nargout=0)
    for i in range(N_captures):
        frame_capture = sio.loadmat("frame_capture.mat")
        frame_capture = frame_capture["frame_capture"]
        if np.count_nonzero(frame_capture) == len(frame_capture):
            break
        print("Capture :", i + 1, "...")
        cmd_string = (
            "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/TX_flow_graph_RX.py -rx_gain "
            + str(args.rx_gain)
            + " -file_path "
            + args.rx_filename
        )
        subprocess.Popen(
            cmd_string,
            shell=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        time.sleep(3)
        print("Capture done")
        eng.TX_Feedback_Decoder(args.num, nargout=0)

