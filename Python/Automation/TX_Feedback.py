import argparse
import numpy as np
import matlab.engine
import time
import scipy.io as sio
import time
import subprocess
from asyncio.subprocess import DEVNULL
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dev_type", type=str, default="encoder")
    parser.add_argument("-num", type=int, default=3)
    parser.add_argument("-rx_gain", type=int, default=15)
    parser.add_argument("-tx_gain", type=int, default=15)
    parser.add_argument("-n_captures", type=int, default=3)
    parser.add_argument("-fdbk", type=str, default="noisy")
    parser.add_argument("-scale", type=float, default=10.0)
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
    eng.TX_Feedback_Encoder(args.num, args.scale, nargout=0)
    if args.num == 1:
        print("\nPROCESS BEGINS\n")

    print("TX Encoder " + str(args.num) + " generated")
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
    print("TX Transmission " + str(args.num) + " starts")

elif args.dev_type == "decoder":

    eng = matlab.engine.start_matlab()

    if args.fdbk == "noisy":
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

    else:
        eng.TX_Feedback_Noiseless(args.num, nargout=0)
        print("Noiseless feedback received at TX")

elif args.dev_type == "encoder_noise":
    eng = matlab.engine.start_matlab()

    # TX Transmission 1
    eng.TX_Send_Noise(nargout=0)
    print("TX Noise generated")
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
    print("TX Transmission starts")

elif args.dev_type == "encoder_channel":
    eng = matlab.engine.start_matlab()

    # TX Transmission 1
    eng.TX_Send_Channel(nargout=0)
    print("TX Noise generated")
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
    print("TX Transmission starts")
elif args.dev_type == "decoder_channel":
    eng = matlab.engine.start_matlab()
    N_captures = args.n_captures
    if os.path.exists(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Channel_Files/Channel_Output.mat"
    ):
        os.remove(
            "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Channel_Files/Channel_Output.mat"
        )
    else:
        print("The file does not exist")

    print("RX Reception " + str(args.num) + "1 starts")
    for i in range(N_captures):
        print("Capture :", i + 1, "...")
        cmd_string = (
            "python3 /home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/TX_flow_graph_RX.py -nsamples 10000000 -rx_gain "
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
        eng.RX_Measure_Channel(nargout=0)
elif args.dev_type == "decoder_noise":
    eng = matlab.engine.start_matlab()
    N_captures = args.n_captures
    if os.path.exists(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Channel_Files/Noise_Output.mat"
    ):
        os.remove(
            "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Channel_Files/Noise_Output.mat"
        )
    else:
        print("The file does not exist")

    print("RX Reception " + str(args.num) + "1 starts")
    for i in range(N_captures):
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
        eng.RX_Measure_Noise(nargout=0)
