import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def gen_rician_fading_examples(data_shape,type):

    fwd_stats = sio.loadmat('Feedback_Files/Forward_Channel.mat')
    fwd_stats = fwd_stats['Forward_Channel']

    fwd_stats = torch.tensor(fwd_stats)
    rand_subc = torch.randint(high=48, size=data_shape)


    K = fwd_stats[rand_subc,0] #Rician Fading coefficient (Ratio of LOS to NLOS paths)
    A = fwd_stats[rand_subc,1] #Rician Fading coefficient (Ratio of LOS to NLOS paths)
    
    coeffLOS = torch.sqrt(K/(K+1))
    coeffNLOS = torch.sqrt(1/(K+1))
    if type == 'fast':
        hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn(data_shape)
        hNLOSImag = torch.randn(data_shape)
    else: #Slow fading case
        hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])
        hNLOSImag = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])
    fading_h = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal,hNLOSImag)
    #Assuming phase information at the receiver
    return A*fading_h

def gen_gaussian_noise_examples(data_shape):

    fwd_stats = sio.loadmat('Feedback_Files/Forward_Noise.mat')
    fwd_stats = fwd_stats['Forward_Noise']

    fwd_stats = torch.tensor(fwd_stats)
    

    Mean = fwd_stats[0,0] #Rician Fading coefficient (Ratio of LOS to NLOS paths)
    STD = fwd_stats[0,1] #Rician Fading coefficient (Ratio of LOS to NLOS paths)
    
    Real = torch.randn(data_shape)
    Imag = torch.randn(data_shape)
    Noise = Mean + STD*torch.complex(Real,Imag)
    #Assuming phase information at the receiver
    return Noise   

data_shape = (10000,1)
X = torch.randint(low=0,high=2,size=data_shape)
H = gen_rician_fading_examples(data_shape,'fast')
N = gen_gaussian_noise_examples(data_shape)
Y = torch.abs(H)*(2*X - 1) + N

X = X.numpy()
Y = Y.numpy()

YR = np.real(Y)
YI = np.imag(Y)

plt.figure()
plt.scatter(YR[np.where(X==0)], YI[np.where(X==0)], color = 'r')
plt.scatter(YR[np.where(X==1)], YI[np.where(X==1)], color = 'b')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.savefig('Figures/Constellation_Created.png')


# X = gen_rician_fading_examples((100000,1),'fast')


# XM = np.abs(np.array(X))
# XP = np.angle(np.array(X))

# plt.figure()
# plt.hist(XM,200)
# plt.savefig('example1.png')

# plt.figure()
# plt.hist(XP,200)
# plt.savefig('example2.png')
