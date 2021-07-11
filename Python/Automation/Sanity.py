import numpy as np
import pickle as pkl
import matlab.engine
import scipy.io as sio


def check_dataset(Eng, Dataset):
    Data = pkl.load(open(Dataset, "rb"))
    DataIn = Data["Data_Input"]
    DataOut = Data["Data_Output"]

    DataIn = {"DataIn": DataIn}
    DataOut = {"DataOut": DataOut}

    sio.savemat("Data_Files/DataIn.mat", DataIn)
    sio.savemat("Data_Files/DataOut.mat", DataOut)

    Eng.Check_Data(nargout=0)


def main():
    eng = matlab.engine.start_matlab()
    check_dataset(eng, "Dataset_OTA_BPSK_2021-07-07/Data_0.10_0.05.pkl")


if __name__ == "__main__":
    main()
