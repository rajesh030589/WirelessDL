import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


def save_datafiles(
    file_name,
    N_captures,
    Frame_Error,
    Data_Input,
    Encoder_Output,
    Receiver_Output,
    Data_Output,
    berHi,
    berLo,
):
    W, X, Y, Z = [], [], [], []
    PP = 0

    for n in range(N_captures):
        for i in range(Frame_Error.shape[1]):
            a = int(Frame_Error[n, i, 1])
            if Frame_Error[n, i, 0] > berLo and Frame_Error[n, i, 0] < berHi:
                W.append(Data_Input[n, :, :, a])
                X.append(Encoder_Output[n, :, :, a])
                Y.append(Receiver_Output[n, :, i])
                Z.append(Data_Output[n, :, :, i])

    Data = {
        "Data_Input": np.array(W),
        "Encoder_Output": np.array(X),
        "Receiver_Output": np.array(Y),
        "Data_Output": np.array(Z),
    }
    pkl.dump(
        Data, open(file_name + "/Data_" + str(berHi) + "_" + str(berLo) + ".pkl", "wb")
    )

    return W, X, Y, Z


def plot_histogram(file_name, X, Z, berHi, berLo):
    plt.figure()
    Z = np.array(Z).flatten()
    E = np.array(X).flatten()
    plt.hist(Z[E == 1], 1000, color="red", alpha=0.3)
    plt.hist(Z[E == 0], 1000, color="blue", alpha=0.3)
    plt.grid(True)
    plt.xlim([-50, 50])
    plt.savefig(file_name + "/Figures/Data_" + str(berHi) + "_" + str(berLo) + ".png")
