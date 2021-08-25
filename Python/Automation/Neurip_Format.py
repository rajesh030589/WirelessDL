import numpy as np
import pickle as pkl
import scipy.io as sio
from datetime import date


def format_dataset(Dataset, MatIn, MatOut):

    n_b = 60
    rate = 1 / 2

    r = int(1 / rate)

    Data = pkl.load(open(Dataset, "rb"))
    DataIn = Data["Data_Input"]
    DataOut = Data["Data_Output"]
    DataIn = np.squeeze(DataIn)

    a = DataIn.shape[0]
    b = DataIn.shape[1]
    c = int(DataIn.shape[2] / n_b)
    d = n_b

    NewIn = np.zeros((a, b, c, d))
    for i in range(d):
        NewIn[:, :, :, i] = DataIn[:, :, i * c : (i + 1) * c]
    NewNewIn = np.zeros((b, c, a * d))
    for i in range(a):
        NewNewIn[:, :, i * d : (i + 1) * d] = NewIn[i, :, :, :]

    DataOut = np.squeeze(DataOut)
    NewOut = np.zeros((a, r * b, c, d))
    for i in range(d):
        NewOut[:, :, :, i] = DataOut[:, :, i * c : (i + 1) * c]
    NewNewOut = np.zeros((r * b, c, a * d))
    for i in range(a):
        NewNewOut[:, :, i * d : (i + 1) * d] = NewOut[i, :, :, :]

    NewNewNewOut = np.zeros((r, b, c, a * d))
    for i in range(r):
        NewNewNewOut[i, :, :, :] = NewNewOut[i::r, :, :]

    DataIn = {"DataIn": NewNewIn}
    DataOut = {"DataOut": NewNewNewOut}

    sio.savemat(MatIn, DataIn)
    sio.savemat(MatOut, DataOut)


def main():
    current_date = date.today()
    file_name = "NEURIP_Datasets/Dataset_64_QAM_FRAMED_OTA_2021-08-23"
    format_dataset(
        file_name + "/Data_0.40_0.10.pkl",
        file_name + "/DataIn_0.40_0.10.mat",
        file_name + "/DataOut_0.40_0.10.mat",
    )


if __name__ == "__main__":
    main()
