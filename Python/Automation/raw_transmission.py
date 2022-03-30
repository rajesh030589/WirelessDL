##Power Norm where we fix the mean and var
#################################
#######  Libraries Used  ########
#################################
import os
import sys
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio


def snr_db2sigma(train_snr):
    return 10 ** (-train_snr * 1.0 / 20)


#################################
#######  Parameters  ########
#################################
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-init_nw_weight", type=str, default="default")
    parser.add_argument("-code_rate", type=int, default=3)
    parser.add_argument("-precompute_stats", type=bool, default=True)  ##########

    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-batch_size", type=int, default=500)
    parser.add_argument("-num_epoch", type=int, default=600)

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument("-block_len", type=int, default=1)
    parser.add_argument("-num_block", type=int, default=50000)
    parser.add_argument("-num_rounds", type=int, default=3)
    # parser.add_argument('-delta_snr', type=int, default=15)  ##SNR_FB - SNR_FW

    parser.add_argument("-enc_num_layer", type=int, default=2)
    parser.add_argument("-dec_num_layer", type=int, default=2)
    parser.add_argument("-enc_num_unit", type=int, default=50)
    parser.add_argument("-dec_num_unit", type=int, default=50)

    parser.add_argument("-frwd_snr", type=float, default=0)
    parser.add_argument("-bckwd_snr", type=float, default=16)

    parser.add_argument("-snr_test_start", type=float, default=0.0)
    parser.add_argument("-snr_test_end", type=float, default=5.0)
    parser.add_argument("-snr_points", type=int, default=6)

    parser.add_argument(
        "-channel_mode",
        choices=["normalize", "lazy_normalize", "tanh"],
        default="lazy_normalize",
    )

    parser.add_argument(
        "-enc_act",
        choices=["tanh", "selu", "relu", "elu", "sigmoid", "none"],
        default="elu",
    )
    parser.add_argument(
        "-dec_act",
        choices=["tanh", "selu", "relu", "elu", "sigmoid", "none"],
        default="none",
    )

    args = parser.parse_args()

    return args


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args = args

        # Encoder
        self.enc_rnn_fwd = torch.nn.GRU(
            3,
            self.args.enc_num_unit,
            num_layers=self.args.enc_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.enc_linear = torch.nn.Linear(
            self.args.enc_num_unit,
            int((self.args.code_rate) * (self.args.block_len) / (self.args.num_rounds)),
        )

        # Decoder
        self.dec_rnn = torch.nn.GRU(
            1,
            self.args.dec_num_unit,
            num_layers=self.args.dec_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.dec_output = torch.nn.Linear(self.args.dec_num_unit, self.args.block_len)

    ##Power Constraint
    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs)
        this_std = torch.std(inputs)
        outputs = (inputs - this_mean) * 1.0 / this_std
        return outputs

    ##Encoder Activation
    def enc_act(self, inputs):
        if self.args.enc_act == "tanh":
            return F.tanh(inputs)
        elif self.args.enc_act == "elu":
            return F.elu(inputs)
        elif self.args.enc_act == "relu":
            return F.relu(inputs)
        elif self.args.enc_act == "selu":
            return F.selu(inputs)
        elif self.args.enc_act == "sigmoid":
            return F.sigmoid(inputs)
        else:
            return inputs

    ##Decoder Activation
    def dec_act(self, inputs):
        if self.args.dec_act == "tanh":
            return F.tanh(inputs)
        elif self.args.dec_act == "elu":
            return F.elu(inputs)
        elif self.args.dec_act == "relu":
            return F.relu(inputs)
        elif self.args.dec_act == "selu":
            return F.selu(inputs)
        elif self.args.dec_act == "sigmoid":
            return F.sigmoid(inputs)
        else:
            return inputs

    def forward(
        self,
        input,
        fwd_noise,
        fb_noise,
        fwd_fading_h,
        fb_fading_h,
        eval=False,
        precomp=False,
        use_precomp=False,
        mu_1_enc=0.0,
        v_1_enc=0.0,
        mu_2_enc=0.0,
        v_2_enc=0.0,
        mu_3_enc=0.0,
        v_3_enc=0.0,
        mu_1_dec=0.0,
        v_1_dec=0.0,
        mu_2_dec=0.0,
        v_2_dec=0.0,
    ):

        input_tmp_1 = torch.cat(
            [
                input.view(self.args.batch_size, 1, self.args.block_len),
                torch.zeros((self.args.batch_size, 1, 2 * self.args.block_len)).to(
                    device
                )
                + 0.5,
            ],
            dim=2,
        )

        x_fwd_1, h_enc_tmp_1 = self.enc_rnn_fwd(input_tmp_1)
        x_tmp_1 = self.enc_act(self.enc_linear(x_fwd_1))

        enc1 = x_tmp_1
        if use_precomp:
            x_tmp_1 = (x_tmp_1 - mu_1_enc) * 1.0 / v_1_enc

        else:
            x_tmp_1 = self.power_constraint(x_tmp_1)

        # x_tmp_1 = np.sqrt(3)*(self.p1_enc*x_tmp_1)/den_enc

        x_rec_1 = fwd_fading_h[:, :, :, 0] * x_tmp_1 + fwd_noise[:, :, :, 0].view(
            self.args.batch_size, 1, self.args.block_len
        )

        x_dec_1, h_dec_tmp_1 = self.dec_rnn(x_rec_1)
        x_dec_1 = self.dec_act(self.dec_output(x_dec_1))
        dec1 = x_dec_1

        if use_precomp:
            x_dec_1 = (x_dec_1 - mu_1_dec) * 1.0 / v_1_dec

        else:
            x_dec_1 = self.power_constraint(x_dec_1)

        # x_dec_1 = np.sqrt(2)*(self.p1_dec*x_dec_1)/den_dec

        noisy_x_dec_1 = fb_fading_h[:, :, :, 0] * x_dec_1 + fb_noise[:, :, :, 0].view(
            self.args.batch_size, 1, self.args.block_len
        )
        input_tmp_2 = torch.cat(
            [
                input.view(self.args.batch_size, 1, self.args.block_len),
                noisy_x_dec_1.view(self.args.batch_size, 1, self.args.block_len),
                torch.zeros((self.args.batch_size, 1, self.args.block_len)).to(device)
                + 0.5,
            ],
            dim=2,
        )

        x_fwd_2, h_enc_tmp_2 = self.enc_rnn_fwd(input_tmp_2, h_enc_tmp_1)
        x_tmp_2 = self.enc_act(self.enc_linear(x_fwd_2))

        enc2 = x_tmp_2

        if use_precomp:
            x_tmp_2 = (x_tmp_2 - mu_2_enc) * 1.0 / v_2_enc

        else:
            x_tmp_2 = self.power_constraint(x_tmp_2)
        # x_tmp_2 = np.sqrt(3)*(self.p2_enc*x_tmp_2)/den_enc

        x_rec_2 = fwd_fading_h[:, :, :, 1] * x_tmp_2 + fwd_noise[:, :, :, 1].view(
            self.args.batch_size, 1, self.args.block_len
        )

        x_dec_2, h_dec_tmp_2 = self.dec_rnn(x_rec_2, h_dec_tmp_1)

        ##before power norm
        x_dec_2_before = self.dec_output(x_dec_2)

        x_dec_2 = self.dec_act(self.dec_output(x_dec_2))
        dec2 = x_dec_2
        if use_precomp:
            x_dec_2 = (x_dec_2 - mu_2_dec) * 1.0 / v_2_dec

        else:
            x_dec_2 = self.power_constraint(x_dec_2)

        # x_dec_2 = np.sqrt(2)*(self.p2_dec*x_dec_2)/den_dec

        noisy_x_dec_2 = fb_fading_h[:, :, :, 1] * x_dec_2 + fb_noise[:, :, :, 1].view(
            self.args.batch_size, 1, self.args.block_len
        )
        input_tmp_3 = torch.cat(
            [
                input.view(self.args.batch_size, 1, self.args.block_len),
                noisy_x_dec_1.view(self.args.batch_size, 1, self.args.block_len),
                noisy_x_dec_2.view(self.args.batch_size, 1, self.args.block_len),
            ],
            dim=2,
        )

        x_fwd_3, h_enc_tmp_3 = self.enc_rnn_fwd(input_tmp_3, h_enc_tmp_2)
        x_tmp_3 = self.enc_act(self.enc_linear(x_fwd_3))

        enc3 = x_tmp_3

        if use_precomp:
            x_tmp_3 = (x_tmp_3 - mu_3_enc) * 1.0 / v_3_enc

        else:
            x_tmp_3 = self.power_constraint(x_tmp_3)
        # x_tmp_3 = np.sqrt(3)*(self.p3_enc*x_tmp_3)/den_enc

        x_rec_3 = fwd_fading_h[:, :, :, 2] * x_tmp_3 + fwd_noise[:, :, :, 2].view(
            self.args.batch_size, 1, self.args.block_len
        )

        x_dec_3, h_dec_tmp_3 = self.dec_rnn(x_rec_3, h_dec_tmp_2)
        x_dec_3 = self.dec_output(x_dec_3)
        # x_dec_3 = self.power_constraint(x_dec_3)

        # final_x=x_dec
        final_x = F.sigmoid(x_dec_3)

        if eval == True:
            return (
                input_tmp_1,
                x_tmp_1,
                fwd_noise[:, :, :, 0],
                fwd_fading_h[:, :, :, 0],
                x_rec_1,
                x_dec_1,
                fb_noise[:, :, :, 0],
                fb_fading_h[:, :, :, 0],
                input_tmp_2,
                x_tmp_2,
                fwd_noise[:, :, :, 1],
                fwd_fading_h[:, :, :, 1],
                x_rec_2,
                x_dec_2,
                x_dec_2_before,
                fb_noise[:, :, :, 1],
                fb_fading_h[:, :, :, 1],
                input_tmp_3,
                x_tmp_3,
                fwd_noise[:, :, :, 2],
                fwd_fading_h[:, :, :, 2],
                x_rec_3,
                final_x,
                x_dec_3,
            )

        if precomp:
            return enc1, enc2, enc3, dec1, dec2

        return final_x


###### MAIN
args = get_args()
print(args)
alpha = 0.1


def errors_ber(y_true, y_pred):

    t1 = np.round(y_true[:, :, :])
    t2 = np.round(y_pred[:, :, :])

    myOtherTensor = np.not_equal(t1, t2).float()
    k = sum(sum(sum(myOtherTensor))) / (
        myOtherTensor.shape[0] * myOtherTensor.shape[1] * myOtherTensor.shape[2]
    )
    return k


# use_cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = AE(args).to(device)
else:
    model = AE(args)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
)

test_ratio = 1
num_train_block, num_test_block = args.num_block, args.num_block / test_ratio

frwd_snr = args.frwd_snr
frwd_sigma = 10 ** (-frwd_snr * 1.0 / 20)

bckwd_snr = args.bckwd_snr
bckwd_sigma = 10 ** (-bckwd_snr * 1.0 / 20)


def gen_rician_fading_noise_examples_rajesh(data_shape, type, alpha):
    fwd_stats = sio.loadmat("Channel_Files/Forward_Channel.mat")
    fwd_stats = fwd_stats["Forward_Channel"]
    fwd_stats = torch.tensor(fwd_stats)
    rand_subc = torch.randint(high=48, size=data_shape)
    K = fwd_stats[
        rand_subc, 0
    ]  # Rician Fading coefficient (Ratio of LOS to NLOS paths)
    A = fwd_stats[
        rand_subc, 1
    ]  # Rician Fading coefficient (Ratio of LOS to NLOS paths)
    A = A.type(torch.FloatTensor).to(device)

    coeffLOS = torch.sqrt(K / (K + 1))
    coeffNLOS = torch.sqrt(1 / (K + 1))
    if type == "fast":
        hLOSReal = torch.ones(
            data_shape
        )  # Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn(data_shape)
        hNLOSImag = torch.randn(data_shape)
    else:  # Slow fading case
        hLOSReal = torch.ones(
            data_shape
        )  # Assuming SISO see page 3.108 in Heath and Lazano
        hLOSImag = torch.ones(data_shape)
        hNLOSReal = torch.randn((data_shape[0], 1, 1)).repeat(
            1, data_shape[1], data_shape[2]
        )
        hNLOSImag = torch.randn((data_shape[0], 1, 1)).repeat(
            1, data_shape[1], data_shape[2]
        )
    fading_h = coeffLOS * torch.complex(hLOSReal, hLOSImag) + coeffNLOS * torch.complex(
        hNLOSReal, hNLOSImag
    )
    fading_h = fading_h.type(torch.FloatTensor).to(device)
    # Assuming phase information at the receiver

    fwd_stats = sio.loadmat("Channel_Files/Forward_Noise.mat")
    fwd_stats = fwd_stats["Forward_Noise"]
    fwd_stats = torch.tensor(fwd_stats)

    Mean = fwd_stats[
        rand_subc, 0
    ]  # Rician Fading coefficient (Ratio of LOS to NLOS paths)
    VAR = fwd_stats[
        rand_subc, 1
    ]  # Rician Fading coefficient (Ratio of LOS to NLOS paths)
    STD = torch.sqrt(VAR)

    Real = torch.randn(data_shape)
    Imag = torch.randn(data_shape)
    Noise = Mean + STD * torch.complex(Real, Imag)
    # Assuming phase information at the receiver
    return alpha * torch.abs(A * fading_h), Noise


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx in range(int(num_train_block / args.batch_size)):

        X_train = torch.randint(
            0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float
        )
        # fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
        # fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
        shape = (args.batch_size, 1, args.block_len, args.num_rounds)
        # fwd_noise = gen_gaussian_noise_examples_rajesh(shape).float()
        fb_noise = torch.zeros(shape).float()

        shape = (args.batch_size, 1, args.block_len, args.num_rounds)
        # fwd_fading_h = gen_rician_fading_examples(shape,'fast')
        # fb_fading_h = gen_rician_fading_examples(shape,'fast')

        # fwd_fading_h = gen_rician_fading_examples_rajesh(shape, "fast").float()
        # fb_fading_h = gen_rician_fading_examples_rajesh(shape,'fast').float()
        fb_fading_h = torch.ones(shape).float()
        fwd_fading_h, fwd_noise = gen_rician_fading_noise_examples_rajesh(
            shape, "fast", alpha
        )
        fwd_fading_h = fwd_fading_h.float()
        fwd_noise = fwd_noise.float()
        # use GPU
        X_train, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h = (
            X_train.to(device),
            fwd_noise.to(device),
            fb_noise.to(device),
            fwd_fading_h.to(device),
            fb_fading_h.to(device),
        )

        optimizer.zero_grad()
        output = model(X_train, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h)

        loss = F.binary_cross_entropy(output, X_train)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # if batch_idx % 1000 == 0:
        # print('Train Epoch: {} [{}/{} Loss: {:.6f}'.format(
        #    epoch, batch_idx, num_train_block/args.batch_size, loss.item()))

    # print(output[0,:,:])
    # print(X_train[0,:,:])
    print(
        "====> Epoch: {}, Average BCE loss: {:.4f}".format(
            epoch, train_loss / (num_train_block / args.batch_size)
        )
    )


def test_2(model):
    model.eval()
    # torch.manual_seed(random.randint(0,1000))

    frwd_snr = args.frwd_snr
    frwd_sigma = 10 ** (-frwd_snr * 1.0 / 20)

    bckwd_snr = args.bckwd_snr
    bckwd_sigma = 10 ** (-bckwd_snr * 1.0 / 20)

    num_train_block = args.num_block

    test_ber = 0.0

    with torch.no_grad():
        mu_1_enc = 0.0
        v_1_enc = 0.0
        mu_2_enc = 0.0
        v_2_enc = 0.0
        mu_3_enc = 0.0
        v_3_enc = 0.0

        mu_1_dec = 0.0
        v_1_dec = 0.0
        mu_2_dec = 0.0
        v_2_dec = 0.0

        if args.precompute_stats:
            ##Step 1: save mean and var
            num_test_batch = 1000
            for batch_idx in range(num_test_batch):

                X_test = torch.randint(
                    0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float
                )
                # fwd_noise  = frwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
                # fb_noise   = bckwd_sigma * torch.randn((args.batch_size, 1 , args.block_len, args.num_rounds), dtype=torch.float)
                shape = (args.batch_size, 1, args.block_len, args.num_rounds)
                # fwd_noise = gen_gaussian_noise_examples_rajesh(shape).float()
                fb_noise = torch.zeros(shape).float()

                shape = (args.batch_size, 1, args.block_len, args.num_rounds)
                # fwd_fading_h = gen_rician_fading_examples(shape,'fast')
                # fb_fading_h = gen_rician_fading_examples(shape,'fast')

                # fwd_fading_h = gen_rician_fading_examples_rajesh(shape, "fast").float()
                fb_fading_h = torch.ones(shape).float()
                # fb_fading_h = gen_rician_fading_examples_rajesh(shape,'fast').float()
                # use GPU

                fwd_fading_h, fwd_noise = gen_rician_fading_noise_examples_rajesh(
                    shape, "fast", alpha
                )
                fwd_fading_h = fwd_fading_h.float()
                fwd_noise = fwd_noise.float()
                X_test, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h = (
                    X_test.to(device),
                    fwd_noise.to(device),
                    fb_noise.to(device),
                    fwd_fading_h.to(device),
                    fb_fading_h.to(device),
                )

                enc_1, enc_2, enc_3, dec_1, dec_2 = model(
                    X_test,
                    fwd_noise,
                    fb_noise,
                    fwd_fading_h,
                    fb_fading_h,
                    False,
                    True,
                    False,
                )

                mu_1_enc += torch.mean(enc_1)
                v_1_enc += torch.std(enc_1)
                mu_2_enc += torch.mean(enc_2)
                v_2_enc += torch.std(enc_2)
                mu_3_enc += torch.mean(enc_3)
                v_3_enc += torch.std(enc_3)

                mu_1_dec += torch.mean(dec_1)
                v_1_dec += torch.std(dec_1)
                mu_2_dec += torch.mean(dec_2)
                v_2_dec += torch.std(dec_2)

            mu_1_enc /= num_test_batch
            v_1_enc /= num_test_batch
            mu_2_enc /= num_test_batch
            v_2_enc /= num_test_batch
            mu_3_enc /= num_test_batch
            v_3_enc /= num_test_batch

            mu_1_dec /= num_test_batch
            v_1_dec /= num_test_batch
            mu_2_dec /= num_test_batch
            v_2_dec /= num_test_batch

        ##Step 2: compute ber
        num_test_batch = 1000
        for batch_idx in range(num_test_batch):

            X_test = torch.randint(
                0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float
            )
            fwd_noise = frwd_sigma * torch.randn(
                (args.batch_size, 1, args.block_len, args.num_rounds), dtype=torch.float
            )
            # fb_noise = bckwd_sigma * torch.randn(
            #     (args.batch_size, 1, args.block_len, args.num_rounds), dtype=torch.float
            # )

            shape = (args.batch_size, 1, args.block_len, args.num_rounds)
            fb_noise = torch.zeros(shape).float()
            fb_fading_h = torch.ones(shape).float()
            fwd_fading_h, fwd_noise = gen_rician_fading_noise_examples_rajesh(
                shape, "fast", alpha
            )
            fwd_fading_h = fwd_fading_h.float()
            fwd_noise = fwd_noise.float()
            # use GPU
            X_test, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h = (
                X_test.to(device),
                fwd_noise.to(device),
                fb_noise.to(device),
                fwd_fading_h.to(device),
                fb_fading_h.to(device),
            )

            X_hat_test = model(
                X_test,
                fwd_noise,
                fb_noise,
                fwd_fading_h,
                fb_fading_h,
                False,
                False,
                args.precompute_stats,
                mu_1_enc,
                v_1_enc,
                mu_2_enc,
                v_2_enc,
                mu_3_enc,
                v_3_enc,
                mu_1_dec,
                v_1_dec,
                mu_2_dec,
                v_2_dec,
            )

            test_ber += errors_ber(X_hat_test.cpu(), X_test.cpu())

    test_ber /= 1.0 * num_test_batch
    print("Test SNR", frwd_snr, "with ber ", float(test_ber))


train_model = True
test_ber = False
eval_scheme = False

##Training
if train_model:

    for epoch in range(1, args.num_epoch + 1):
        train(epoch)

        if epoch % 10 == 0:
            # torch.save(model.state_dict(), "test_8.pt")
            # torch.save(model.state_dict(), "./ric_no_trainable_weights.pt")
            print("Model is saved")
            test_2(model)

##Testing
elif test_ber:

    pretrained_model = torch.load(
        "./models/power_norm_models_fading_csi_bl1_snr0/trainable_weights_12.pt"
    )
    model.load_state_dict(pretrained_model)
    model.args = args
    test_2(model)

##Scheme analysis
elif eval_scheme:
    pretrained_model = torch.load(
        "./models/power_norm_models_fading_csi_bl1_snr0/trainable_weights_8.pt"
    )
    model.load_state_dict(pretrained_model)
    model.args = args

    model.eval()
    frwd_snr = args.frwd_snr
    frwd_sigma = 10 ** (-frwd_snr * 1.0 / 20)

    bckwd_snr = args.bckwd_snr
    bckwd_sigma = 10 ** (-bckwd_snr * 1.0 / 20)

    num_train_block = args.num_block

    with torch.no_grad():
        mu_1_enc = 0.0
        v_1_enc = 0.0
        mu_2_enc = 0.0
        v_2_enc = 0.0
        mu_3_enc = 0.0
        v_3_enc = 0.0

        mu_1_dec = 0.0
        v_1_dec = 0.0
        mu_2_dec = 0.0
        v_2_dec = 0.0

        if args.precompute_stats:
            ##Step 1: save mean and var
            num_test_batch = 5000
            for batch_idx in range(num_test_batch):

                X_test = torch.randint(
                    0, 2, (args.batch_size, 1, args.block_len), dtype=torch.float
                )
                fwd_noise = frwd_sigma * torch.randn(
                    (args.batch_size, 1, args.block_len, args.num_rounds),
                    dtype=torch.float,
                )
                fb_noise = bckwd_sigma * torch.randn(
                    (args.batch_size, 1, args.block_len, args.num_rounds),
                    dtype=torch.float,
                )
                fwd_fading_h = torch.sqrt(
                    torch.randn(args.batch_size, 1, args.block_len, args.num_rounds)
                    ** 2
                    + torch.randn(args.batch_size, 1, args.block_len, args.num_rounds)
                    ** 2
                ) / torch.sqrt(torch.tensor(3.14 / 2.0))
                fb_fading_h = torch.sqrt(
                    torch.randn(args.batch_size, 1, args.block_len, args.num_rounds)
                    ** 2
                    + torch.randn(args.batch_size, 1, args.block_len, args.num_rounds)
                    ** 2
                ) / torch.sqrt(torch.tensor(3.14 / 2.0))

                # use GPU
                X_test, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h = (
                    X_test.to(device),
                    fwd_noise.to(device),
                    fb_noise.to(device),
                    fwd_fading_h.to(device),
                    fb_fading_h.to(device),
                )

                enc_1, enc_2, enc_3, dec_1, dec_2 = model(
                    X_test,
                    fwd_noise,
                    fb_noise,
                    fwd_fading_h,
                    fb_fading_h,
                    False,
                    True,
                    False,
                )

                mu_1_enc += torch.mean(enc_1)
                v_1_enc += torch.std(enc_1)
                mu_2_enc += torch.mean(enc_2)
                v_2_enc += torch.std(enc_2)
                mu_3_enc += torch.mean(enc_3)
                v_3_enc += torch.std(enc_3)

                mu_1_dec += torch.mean(dec_1)
                v_1_dec += torch.std(dec_1)
                mu_2_dec += torch.mean(dec_2)
                v_2_dec += torch.std(dec_2)

            mu_1_enc /= num_test_batch
            v_1_enc /= num_test_batch
            mu_2_enc /= num_test_batch
            v_2_enc /= num_test_batch
            mu_3_enc /= num_test_batch
            v_3_enc /= num_test_batch

            mu_1_dec /= num_test_batch
            v_1_dec /= num_test_batch
            mu_2_dec /= num_test_batch
            v_2_dec /= num_test_batch

        ##Step 2: generate examples
        num_test_batch = 1
        for batch_idx in range(num_test_batch):

            X_zeros = torch.zeros(
                (args.batch_size // 2, 1, args.block_len), dtype=torch.float
            )
            X_ones = torch.ones(
                (args.batch_size // 2, 1, args.block_len), dtype=torch.float
            )
            X_test = torch.cat([X_zeros, X_ones], dim=0)
            fwd_noise = frwd_sigma * torch.randn(
                (args.batch_size, 1, args.block_len, args.num_rounds), dtype=torch.float
            )
            fb_noise = bckwd_sigma * torch.randn(
                (args.batch_size, 1, args.block_len, args.num_rounds), dtype=torch.float
            )

            fwd_fading_h = torch.sqrt(
                torch.randn(args.batch_size, 1, args.block_len, args.num_rounds) ** 2
                + torch.randn(args.batch_size, 1, args.block_len, args.num_rounds) ** 2
            ) / torch.sqrt(torch.tensor(3.14 / 2.0))
            fb_fading_h = torch.sqrt(
                torch.randn(args.batch_size, 1, args.block_len, args.num_rounds) ** 2
                + torch.randn(args.batch_size, 1, args.block_len, args.num_rounds) ** 2
            ) / torch.sqrt(torch.tensor(3.14 / 2.0))

            # use GPU

            X_test, fwd_noise, fb_noise, fwd_fading_h, fb_fading_h = (
                X_test.to(device),
                fwd_noise.to(device),
                fb_noise.to(device),
                fwd_fading_h.to(device),
                fb_fading_h.to(device),
            )

            (
                msg1,
                code1,
                fwnoise1,
                fwh1,
                x_rec_1,
                fb1,
                fbnoise1,
                fbh1,
                msg2,
                code2,
                fwnoise2,
                fwh2,
                x_rec_2,
                fb2,
                fb2_before,
                fbnoise2,
                fbh2,
                msg3,
                code3,
                fwnoise3,
                fwh3,
                x_rec_3,
                decoded_post_sigmoid,
                decoded_pre_sigmoid,
            ) = model(
                X_test,
                fwd_noise,
                fb_noise,
                fwd_fading_h,
                fb_fading_h,
                True,
                False,
                args.precompute_stats,
                mu_1_enc,
                v_1_enc,
                mu_2_enc,
                v_2_enc,
                mu_3_enc,
                v_3_enc,
                mu_1_dec,
                v_1_dec,
                mu_2_dec,
                v_2_dec,
            )

        ##Round 1
        msg1 = np.reshape(msg1.cpu().numpy(), (args.batch_size, 2))
        code1 = np.reshape(code1.cpu().numpy(), (args.batch_size, 1))
        fwnoise1 = np.reshape(fwnoise1.cpu().numpy(), (args.batch_size, 1))
        fwh1 = np.reshape(fwh1.cpu().numpy(), (args.batch_size, 1))
        fb1 = np.reshape(fb1.cpu().numpy(), (args.batch_size, 1))
        x_rec_1 = np.reshape(x_rec_1.cpu().numpy(), (args.batch_size, 2))
        fbnoise1 = np.reshape(fbnoise1.cpu().numpy(), (args.batch_size, 1))
        fbh1 = np.reshape(fbh1.cpu().numpy(), (args.batch_size, 1))

        ##Round 2
        msg2 = np.reshape(msg2.cpu().numpy(), (args.batch_size, 2))
        code2 = np.reshape(code2.cpu().numpy(), (args.batch_size, 1))
        fwnoise2 = np.reshape(fwnoise2.cpu().numpy(), (args.batch_size, 1))
        fwh2 = np.reshape(fwh2.cpu().numpy(), (args.batch_size, 1))
        fb2 = np.reshape(fb2.cpu().numpy(), (args.batch_size, 1))
        fb2_before = np.reshape(fb2_before.cpu().numpy(), (args.batch_size, 1))
        x_rec_2 = np.reshape(x_rec_2.cpu().numpy(), (args.batch_size, 2))
        fbnoise2 = fbnoise2.cpu().numpy()
        fbh2 = np.reshape(fbh2.cpu().numpy(), (args.batch_size, 1))

        ##Round 3
        msg3 = np.reshape(msg3.cpu().numpy(), (args.batch_size, 2))
        code3 = np.reshape(code3.cpu().numpy(), (args.batch_size, 1))
        fwnoise3 = fwnoise3.cpu().numpy()
        fwh3 = np.reshape(fwh3.cpu().numpy(), (args.batch_size, 1))
        x_rec_3 = np.reshape(x_rec_3.cpu().numpy(), (args.batch_size, 2))
        mdecoded_post_sigmoidsg1 = np.reshape(
            decoded_post_sigmoid.cpu().numpy(), (args.batch_size, 1)
        )
        decoded_pre_sigmoid = np.reshape(
            decoded_pre_sigmoid.cpu().numpy(), (args.batch_size, 1)
        )

    fig = plt.figure()

    ##Transmission 1
    ##code vs input
    plt.figure(1)
    plt.plot(
        msg1[: args.batch_size // 2, 0],
        code1[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        msg1[args.batch_size // 2 :, 0],
        code1[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$b$", fontsize=12)
    plt.ylabel("$x^1$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis1.png")

    ##code vs h_fw
    plt.figure(2)
    plt.plot(
        fwh1[: args.batch_size // 2],
        code1[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh1[args.batch_size // 2 :],
        code1[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^1$", fontsize=12)
    plt.ylabel("$x^1$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis2.png")

    ##Feedback 1
    ##code vs input
    plt.figure(3)
    plt.plot(
        x_rec_1[: args.batch_size // 2, 0],
        fb1[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        x_rec_1[args.batch_size // 2 :, 0],
        fb1[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$y^1$", fontsize=12)
    plt.ylabel("$c^1$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis3.png")

    ##code vs h_fb
    plt.figure(4)
    plt.plot(
        fbh1[: args.batch_size // 2],
        fb1[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fbh1[args.batch_size // 2 :, 0],
        fb1[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fb}^1$", fontsize=12)
    plt.ylabel("$c^1$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis4.png")

    ##code vs h_fb and input

    ax = plt.axes(projection="3d")
    ax.scatter3D(
        fbh1[: args.batch_size // 2],
        x_rec_1[: args.batch_size // 2, 0],
        fb1[: args.batch_size // 2],
        "red",
        label="initial msg=0",
        cmap="hsv",
    )
    ax.scatter3D(
        fbh1[args.batch_size // 2 :],
        x_rec_1[args.batch_size // 2 :, 0],
        fb1[args.batch_size // 2 :],
        "blue",
        label="initial msg=1",
        cmap="hsv",
    )
    ax.set_xlabel("$h_{fb}^1$")
    ax.set_ylabel("$y^1$")
    ax.set_zlabel("$c^1$")
    plt.legend(loc="best")
    plt.savefig("./analysis_figs/3d_1.png")

    ##Transmission 2
    ##code vs input
    plt.figure(5)
    plt.plot(
        msg2[: args.batch_size // 2, 0],
        code2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        msg2[args.batch_size // 2 :, 0],
        code2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$c^1 + w^1$", fontsize=12)
    plt.ylabel("$x^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis5.png")

    ##code vs h_fw_1
    plt.figure(6)
    plt.plot(
        fwh1[: args.batch_size // 2],
        code2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh1[args.batch_size // 2 :],
        code2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^1$", fontsize=12)
    plt.ylabel("$x^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis6.png")

    ##code vs h_fw_2
    plt.figure(7)
    plt.plot(
        fwh2[: args.batch_size // 2],
        code2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh2[args.batch_size // 2 :],
        code2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^2$", fontsize=12)
    plt.ylabel("$x^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis7.png")

    ax = plt.axes(projection="3d")
    ax.scatter3D(
        fwh1[: args.batch_size // 2],
        fwh2[: args.batch_size // 2, 0],
        code2[: args.batch_size // 2],
        "red",
        label="initial msg=0",
        cmap="hsv",
    )
    ax.scatter3D(
        fwh1[args.batch_size // 2 :],
        fwh2[args.batch_size // 2 :, 0],
        code2[args.batch_size // 2 :],
        "blue",
        label="initial msg=1",
        cmap="hsv",
    )
    ax.set_xlabel("$h_{fw}^1$")
    ax.set_ylabel("$h_{fw}^2$")
    ax.set_zlabel("$x^2$")
    plt.legend(loc="best")
    plt.savefig("./analysis_figs/3d_2.png")

    ##Feedback 2
    ##code vs input
    plt.figure(8)
    plt.plot(
        x_rec_2[: args.batch_size // 2, 0],
        fb2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        x_rec_2[args.batch_size // 2 :, 0],
        fb2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$y^2$", fontsize=12)
    plt.ylabel("$c^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis8.png")

    ##code vs h_fb_1
    plt.figure(9)
    plt.plot(
        fbh1[: args.batch_size // 2],
        fb2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fbh1[args.batch_size // 2 :, 0],
        fb2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fb}^1$", fontsize=12)
    plt.ylabel("$c^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis9.png")

    ##code vs h_fb_2
    plt.figure(10)
    plt.plot(
        fbh2[: args.batch_size // 2],
        fb2[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fbh2[args.batch_size // 2 :, 0],
        fb2[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fb}^2$", fontsize=12)
    plt.ylabel("$c^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis10.png")

    ##Transmission 3
    ##code vs input
    plt.figure(11)
    plt.plot(
        msg3[: args.batch_size // 2, 0],
        code3[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        msg3[args.batch_size // 2 :, 0],
        code3[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$c^2 + w^2$", fontsize=12)
    plt.ylabel("$x^3$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis11.png")

    ##code vs h_fw_1
    plt.figure(12)
    plt.plot(
        fwh1[: args.batch_size // 2],
        code3[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh1[args.batch_size // 2 :],
        code3[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^1$", fontsize=12)
    plt.ylabel("$x^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis12.png")

    ##code vs h_fw_2
    plt.figure(13)
    plt.plot(
        fwh2[: args.batch_size // 2],
        code3[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh2[args.batch_size // 2 :],
        code3[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^2$", fontsize=12)
    plt.ylabel("$x^2$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis13.png")

    ##code vs h_fw_3
    plt.figure(14)
    plt.plot(
        fwh3[: args.batch_size // 2],
        code3[: args.batch_size // 2],
        "bo",
        label="initial msg=0",
    )
    plt.plot(
        fwh3[args.batch_size // 2 :],
        code3[args.batch_size // 2 :],
        "r+",
        label="initial msg=1",
    )
    plt.xlabel("$h_{fw}^3$", fontsize=12)
    plt.ylabel("$x^3$", fontsize=12)
    plt.legend(loc="best")
    plt.grid()
    plt.savefig("./analysis_figs/analysis14.png")

    ##Round 1
    msg1 = np.reshape(msg1.cpu().numpy(), (args.batch_size, 6))
    code1 = np.reshape(code1.cpu().numpy(), (args.batch_size, 1))
    fwnoise1 = np.reshape(fwnoise1.cpu().numpy(), (args.batch_size, 1))
    fwh1 = np.reshape(fwh1.cpu().numpy(), (args.batch_size, 1))
    fb1 = np.reshape(fb1.cpu().numpy(), (args.batch_size, 1))
    x_rec_1 = np.reshape(x_rec_1.cpu().numpy(), (args.batch_size, 3))
    fbnoise1 = np.reshape(fbnoise1.cpu().numpy(), (args.batch_size, 1))
    fbh1 = np.reshape(fbh1.cpu().numpy(), (args.batch_size, 1))

    ##Round 2
    msg2 = np.reshape(msg2.cpu().numpy(), (args.batch_size, 6))
    code2 = np.reshape(code2.cpu().numpy(), (args.batch_size, 1))
    fwnoise2 = np.reshape(fwnoise2.cpu().numpy(), (args.batch_size, 1))
    fwh2 = np.reshape(fwh2.cpu().numpy(), (args.batch_size, 1))
    fb2 = np.reshape(fb2.cpu().numpy(), (args.batch_size, 1))
    fb2_before = np.reshape(fb2_before.cpu().numpy(), (args.batch_size, 1))
    x_rec_2 = np.reshape(x_rec_2.cpu().numpy(), (args.batch_size, 3))
    fbnoise2 = fbnoise2.cpu().numpy()
    fbh2 = np.reshape(fbh2.cpu().numpy(), (args.batch_size, 1))

    ##Round 3
    msg3 = np.reshape(msg3.cpu().numpy(), (args.batch_size, 6))
    code3 = np.reshape(code3.cpu().numpy(), (args.batch_size, 1))
    fwnoise3 = fwnoise3.cpu().numpy()
    fwh3 = np.reshape(fwh3.cpu().numpy(), (args.batch_size, 1))
    x_rec_3 = np.reshape(x_rec_3.cpu().numpy(), (args.batch_size, 3))
    mdecoded_post_sigmoidsg1 = np.reshape(
        decoded_post_sigmoid.cpu().numpy(), (args.batch_size, 1)
    )
    decoded_pre_sigmoid = np.reshape(
        decoded_pre_sigmoid.cpu().numpy(), (args.batch_size, 1)
    )

    ##case 1
    ##case 2
    ##case 3

    ##Color code last step
    # x_rec_3_temp1 = []
    # mdecoded_post_sigmoidsg1_temp1 = []

    # x_rec_3_temp2 = []
    # mdecoded_post_sigmoidsg1_temp2 = []

    # for i in range(args.batch_size):
    #     if fb2[i,:] <= 0:
    #         x_rec_3_temp1.append(x_rec_3[i])
    #         mdecoded_post_sigmoidsg1_temp1.append(mdecoded_post_sigmoidsg1[i])
    #     else:
    #         x_rec_3_temp2.append(x_rec_3[i])
    #         mdecoded_post_sigmoidsg1_temp2.append(mdecoded_post_sigmoidsg1[i])

    # plt.figure(7)
    # plt.plot(x_rec_3_temp1,mdecoded_post_sigmoidsg1_temp1,'bo',label='$\hat{b}^2$<=0')
    # plt.plot(x_rec_3_temp2,mdecoded_post_sigmoidsg1_temp2,'r+',label='$\hat{b}^2$>0')
    # plt.xlabel('$y^3$',fontsize=12)
    # plt.ylabel('$\hat{b}^3$',fontsize=12)
    # plt.legend(loc='best')
    # plt.grid()
    # plt.savefig('analysis7.png')
