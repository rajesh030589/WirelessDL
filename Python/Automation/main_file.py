import numpy as np
import matlab.engine
from datetime import date
import os
import scipy.io as sio
import pickle as pkl
import matplotlib.pyplot as plt
import os
from Sanity import check_dataset
from tqdm import tqdm

data_generate = True
sanity_check = True
OTA_data = True
Frame_802_11 = True
N_captures = 10


id = "QPSK_FRAMED_OTA_1"

if OTA_data == True and Frame_802_11 == False:
    raise Exception("OTA Data must be Framed")


current_date = date.today()
file_name = "Datasets/Dataset_" + str(id) + "_" + str(current_date)

# make a folder for today's data
try:
    try:
        os.mkdir("Datasets")
    except:
        pass
    os.mkdir(file_name)
    os.mkdir(file_name + "/Figures")
except:
    pass

print(" Capture Process Starts")
print("=========================\n")
print("Starting MATLAB")

eng = matlab.engine.start_matlab()

Data_Input, Encoder_Output, Receiver_Output, Data_Output, Frame_Error = (
    [],
    [],
    [],
    [],
    [],
)

for i in tqdm(range(N_captures)):
    print("Capture ", i)

    # Generate TX
    if Frame_802_11:
        eng.TX_data_generate("802_11 Framed", nargout=0)
    else:
        eng.TX_data_generate("Raw Data", nargout=0)
    # Pass through Channel

    if OTA_data:
        os.system("python3 flow_graph.py")
    else:
        eng.awgn_channel("complex_noise", nargout=0)

    # Decode RX
    if Frame_802_11:
        eng.RX_data_extract("802_11 Framed", nargout=0)
    else:
        eng.RX_data_extract("Raw Data", nargout=0)

    # Prepare Data
    # Input Data
    Data = sio.loadmat("Data_Files/Data_Input.mat")
    Data_Input.append(Data["Data_Input"])

    # Encoder Output
    Data = sio.loadmat("Data_Files/Encoded_Output.mat")
    Encoder_Output.append(Data["Encoder_Output"])

    # Receiver Output
    Data = sio.loadmat("Data_Files/Receiver_Output.mat")
    Receiver_Output.append(Data["Receiver_Output"])

    # Output LLR
    Data = sio.loadmat("Data_Files/Data_Output.mat")
    Data_Output.append(Data["Data_Output"])

    # Frame Error
    Data = sio.loadmat("Data_Files/Frame_Error.mat")
    Frame_Error.append(Data["Frame_Error"])

Data_Input = np.array(Data_Input)
Data_Output = np.array(Data_Output)
Receiver_Output = np.array(Receiver_Output)
Encoder_Output = np.array(Encoder_Output)
Frame_Error = np.array(Frame_Error)


if ~Frame_802_11:
    Data_Input = np.expand_dims(Data_Input, -1)
    Encoder_Output = np.expand_dims(Encoder_Output, -1)
    Receiver_Output = np.expand_dims(Receiver_Output, -1)
    Data_Output = np.expand_dims(Data_Output, -1)

W1, W2, W3, W4 = [], [], [], []
X1, X2, X3, X4 = [], [], [], []
Y1, Y2, Y3, Y4 = [], [], [], []
Z1, Z2, Z3, Z4 = [], [], [], []
PP = 0
for n in range(N_captures):
    for i in range(Frame_Error.shape[1]):

        a = int(Frame_Error[n, i, 1])

        if Frame_Error[n, i, 0] > 0.1 and Frame_Error[n, i, 0] < 0.4:
            W1.append(Data_Input[n, :, :, a])
            X1.append(Encoder_Output[n, :, :, a])
            Y1.append(Receiver_Output[n, :, i])
            Z1.append(Data_Output[n, :, :, i])
        elif Frame_Error[n, i, 0] > 0.05 and Frame_Error[n, i, 0] < 0.1:
            W2.append(Data_Input[n, :, :, a])
            X2.append(Encoder_Output[n, :, :, a])
            Y2.append(Receiver_Output[n, :, i])
            Z2.append(Data_Output[n, :, :, i])
        elif Frame_Error[n, i, 0] > 0.01 and Frame_Error[n, i, 0] < 0.05:
            W3.append(Data_Input[n, :, :, a])
            X3.append(Encoder_Output[n, :, :, a])
            Y3.append(Receiver_Output[n, :, i])
            Z3.append(Data_Output[n, :, :, i])
        elif Frame_Error[n, i, 0] < 0.01 and Frame_Error[n, i, 0] > 0.001:
            W4.append(Data_Input[n, :, :, a])
            X4.append(Encoder_Output[n, :, :, a])
            Y4.append(Receiver_Output[n, :, i])
            Z4.append(Data_Output[n, :, :, i])

Data1 = {
    "Data_Input": np.array(W1),
    "Encoder_Output": np.array(X1),
    "Receiver_Output": np.array(Y1),
    "Data_Output": np.array(Z1),
}
Data2 = {
    "Data_Input": np.array(W2),
    "Encoder_Output": np.array(X2),
    "Receiver_Output": np.array(Y2),
    "Data_Output": np.array(Z2),
}
Data3 = {
    "Data_Input": np.array(W3),
    "Encoder_Output": np.array(X3),
    "Receiver_Output": np.array(Y3),
    "Data_Output": np.array(Z3),
}
Data4 = {
    "Data_Input": np.array(W4),
    "Encoder_Output": np.array(X4),
    "Receiver_Output": np.array(Y4),
    "Data_Output": np.array(Z4),
}

pkl.dump(Data1, open(file_name + "/Data_0.40_0.10.pkl", "wb"))
pkl.dump(Data2, open(file_name + "/Data_0.10_0.05.pkl", "wb"))
pkl.dump(Data3, open(file_name + "/Data_0.05_0.01.pkl", "wb"))
pkl.dump(Data4, open(file_name + "/Data_0.01_0.001.pkl", "wb"))


# Plot histogram
plt.figure()
Z = np.array(Z1).flatten()
E = np.array(X1).flatten()
plt.hist(Z[E == 1], 1000, color="red", alpha=0.3)
plt.hist(Z[E == 0], 1000, color="blue", alpha=0.3)
plt.grid(True)
plt.xlim([-50, 50])
plt.savefig(file_name + "/Figures/Data_0.40_0.10.png")

plt.figure()
Z = np.array(Z2).flatten()
E = np.array(X2).flatten()
plt.hist(Z[E == 1], 800, color="red", alpha=0.3)
plt.hist(Z[E == 0], 800, color="blue", alpha=0.3)
plt.grid(True)
plt.xlim([-50, 50])
plt.savefig(file_name + "/Figures/Data_0.10_0.05.png")

plt.figure()
Z = np.array(Z3).flatten()
E = np.array(X3).flatten()
plt.hist(Z[E == 1], 600, color="red", alpha=0.3)
plt.hist(Z[E == 0], 600, color="blue", alpha=0.3)
plt.grid(True)
plt.xlim([-50, 50])
plt.savefig(file_name + "/Figures/Data_0.05_0.01.png")

plt.figure()
Z = np.array(Z4).flatten()
E = np.array(X4).flatten()
plt.hist(Z[E == 1], 500, color="red", alpha=0.3)
plt.hist(Z[E == 0], 500, color="blue", alpha=0.3)
plt.grid(True)
plt.xlim([-50, 50])
plt.savefig(file_name + "/Figures/Data_0.01_0.001.png")


# Sanity Checks
check_dataset(eng, file_name + "/Data_0.40_0.10.pkl")
check_dataset(eng, file_name + "/Data_0.10_0.05.pkl")
check_dataset(eng, file_name + "/Data_0.05_0.01.pkl")
check_dataset(eng, file_name + "/Data_0.01_0.001.pkl")
