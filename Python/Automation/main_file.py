import numpy as np
import matlab.engine
from datetime import date
import os
import scipy.io as sio
import pickle as pkl
import matplotlib.pyplot as plt
import os
from Sanity import check_dataset

id = "OTA_BPSK"
N_captures = 20

current_date = date.today()
file_name = "Dataset_" + str(id) + "_" + str(current_date)

# make a folder for today's data
try:
    os.mkdir(file_name)
    os.mkdir(file_name + "/Figures")
except:
    pass

print(" Capture Process Starts\n")
print("=========================\n")
print("Starting MATLAB")

eng = matlab.engine.start_matlab()

# Data_Input, Encoder_Output, Data_Output, Frame_Error = [], [], [], []

# for i in range(N_captures):
#     print("Capture ", i)
#     # # Generate TX
#     eng.TX_data_generate(nargout=0)

#     # # Pass through Channel
#     # eng.awgn_channel(nargout=0)
#     os.system("python3 flow_graph.py")

#     # # Decode RX
#     eng.RX_data_extract(nargout=0)

#     # Prepare Data
#     # Input Data
#     Data = sio.loadmat("Data_Files/Data_Input.mat")
#     Data_Input.append(Data["Data_Input"])

#     # Encoder Output
#     Data = sio.loadmat("Data_Files/Encoded_Output.mat")
#     Encoder_Output.append(Data["Encoder_Output"])

#     # Output LLR
#     Data = sio.loadmat("Data_Files/Data_Output.mat")
#     Data_Output.append(Data["Data_Output"])

#     # Frame Error
#     Data = sio.loadmat("Data_Files/Frame_Error.mat")
#     Frame_Error.append(Data["Frame_Error"])

# Data_Input = np.array(Data_Input)
# Data_Output = np.array(Data_Output)
# Encoder_Output = np.array(Encoder_Output)
# Frame_Error = np.array(Frame_Error)

# X1, X2, X3, X4 = [], [], [], []
# Y1, Y2, Y3, Y4 = [], [], [], []
# Z1, Z2, Z3, Z4 = [], [], [], []
# PP = 0
# for n in range(N_captures):
#     for i in range(Frame_Error.shape[1]):

#         a = int(Frame_Error[n, i, 1])

#         if Frame_Error[n, i, 0] > 0.1 and Frame_Error[n, i, 0] < 0.4:
#             X1.append(Data_Input[n, :, :, a])
#             Y1.append(Encoder_Output[n, :, :, a])
#             Z1.append(Data_Output[n, :, :, i])
#         elif Frame_Error[n, i, 0] > 0.05 and Frame_Error[n, i, 0] < 0.1:
#             X2.append(Data_Input[n, :, :, a])
#             Y2.append(Encoder_Output[n, :, :, a])
#             Z2.append(Data_Output[n, :, :, i])
#         elif Frame_Error[n, i, 0] > 0.01 and Frame_Error[n, i, 0] < 0.05:
#             X3.append(Data_Input[n, :, :, a])
#             Y3.append(Encoder_Output[n, :, :, a])
#             Z3.append(Data_Output[n, :, :, i])
#         elif Frame_Error[n, i, 0] < 0.01 and Frame_Error[n, i, 0] > 0.001:
#             X4.append(Data_Input[n, :, :, a])
#             Y4.append(Encoder_Output[n, :, :, a])
#             Z4.append(Data_Output[n, :, :, i])

# Data1 = {
#     "Data_Input": np.array(X1),
#     "Encoder_Output": np.array(Y1),
#     "Data_Output": np.array(Z1),
# }
# Data2 = {
#     "Data_Input": np.array(X2),
#     "Encoder_Output": np.array(Y2),
#     "Data_Output": np.array(Z2),
# }
# Data3 = {
#     "Data_Input": np.array(X3),
#     "Encoder_Output": np.array(Y3),
#     "Data_Output": np.array(Z3),
# }
# Data4 = {
#     "Data_Input": np.array(X4),
#     "Encoder_Output": np.array(Y4),
#     "Data_Output": np.array(Z4),
# }

# pkl.dump(Data1, open(file_name + "/Data_0.40_0.10.pkl", "wb"))
# pkl.dump(Data2, open(file_name + "/Data_0.10_0.05.pkl", "wb"))
# pkl.dump(Data3, open(file_name + "/Data_0.05_0.01.pkl", "wb"))
# pkl.dump(Data4, open(file_name + "/Data_0.01_0.001.pkl", "wb"))


# # Plot histogram
# plt.figure()
# plt.hist(np.array(Z1).flatten(), 200)
# plt.savefig(file_name + "/Figures/Data_0.40_0.10.png")

# plt.figure()
# plt.hist(np.array(Z2).flatten(), 200)
# plt.savefig(file_name + "/Figures/Data_0.10_0.05.png")

# plt.figure()
# plt.hist(np.array(Z3).flatten(), 200)
# plt.savefig(file_name + "/Figures/Data_0.05_0.01.png")

# plt.figure()
# plt.hist(np.array(Z4).flatten(), 200)
# plt.savefig(file_name + "/Figures/Data_0.01_0.001.png")

# Sanity Checks
check_dataset(eng, file_name + "/Data_0.40_0.10.pkl")
check_dataset(eng, file_name + "/Data_0.10_0.05.pkl")
check_dataset(eng, file_name + "/Data_0.05_0.01.pkl")
check_dataset(eng, file_name + "/Data_0.01_0.001.pkl")
