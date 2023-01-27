#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:23:08 2023

@author: rodolfofreitas
"""
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import numpy as np
import h5py
import os
import sys
import argparse
import time
import json
import matplotlib.pyplot as plt
from models import FCNN
from data_utils import read_data, load_data
from data_plot import conditional
# Reproducibility
np.random.seed(12345)
torch.manual_seed(12345)

# default to use cuda
parser = argparse.ArgumentParser(description='Fully connected NN for the cell-reacting fraction')
parser.add_argument('--case-name', type=str, default='3D_DNS_R2_premCH4', help='experiment name')
parser.add_argument('--data-dir', type=str, default="../../Filtered", help='data directory')
parser.add_argument('--n-sp', type=int, default=16, help='number of species')
parser.add_argument('--dimz', type=int, default=40, help='number of points in z-direction') # Original 3D DNS data dimension
parser.add_argument('--dimx', type=int, default=962, help='number of points in x-direction')
parser.add_argument('--dimy', type=int, default=512, help='number of points in y-direction')
parser.add_argument('--filter-name', type=str, default='9', help='filter width [2, 3, 6, 9, 12, 18]') # Delta_plus  --> [0.3 0.5 1.0 1.5 2.0 3.0]
parser.add_argument('--taum-name', type=str, default='kolmo_EDC', help='mixing timescale [kolmo, kolmo_EDC, velS, dynC]')
parser.add_argument('--tauc-name', type=str, default='Ch', help='chemical timescale [Sabel, Ch, SFR, FFR]')
parser.add_argument('--num-layers', type=int, default=8, help='number of FC layers')
parser.add_argument('--neurons-fc', type=int, default=100, help='number of neurons in the fully-connected layer')
parser.add_argument('--n-iterations', type=int, default=1000, help='number of iterations to train (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learnign rate')
parser.add_argument('--beta', type=float, default=5e-1, help='penalization factor (Physics-Aware, mass balance)')
parser.add_argument('--batch-size', type=int, default=1000, help='input batch size for training')
parser.add_argument('--train-size', type=float, default=0.7, help="% of data selected to train")
parser.add_argument('--activation', type=str, default='relu', help='Hidden layer activation, [relu, elu, gelu, tanh, None=linear]')
parser.add_argument('--output-activation', type=str, default=None, help='Output layer activation, sigmoid ~ [0,1], None=linear')
parser.add_argument('--beta1', type=float, default=1.0, help="regularization parameter for mse")
parser.add_argument('--beta2', type=float, default=5e-1, help="regularization parameter for mass balance loss term")
args = parser.parse_args()

# Check if cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('------------ Arguments -------------')
print("Torch device:{}".format(device))
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
''' 
 %% Kwargs for Dataloader 
    num_workers (int, optional): how many subprocesses to use for data loading. 
                                 0 means that the data will be loaded in the main process. (default: 0)
    
    pin_memory (bool, optional): If True, the data loader will copy Tensors 
                                into device/CUDA pinned memory before returning them. 
                                If your data elements are a custom type, or your collate_fn 
                                returns a batch that is a custom type, see the example below.
'''
kwargs = {'num_workers': 0,'pin_memory': True} if torch.cuda.is_available() else {}

# load data
fdimz, fdimx, fdimy, Da, T, C, C_var, Chi_C, R_DNS, R_batch, R_PaSR, DH_, HRR_batch, HRR_DNS, HRR_PaSR  = read_data(args.case_name, 
                                                                                                                    args.filter_name, 
                                                                                                                    args.taum_name,
                                                                                                                    args.tauc_name,
                                                                                                                    args.n_sp,
                                                                                                                    args.data_dir)

# load training data
Xtrain, Xtest, YtrainDNS, YtestDNS, Ytrainbatch, Ytestbatch, RtrainDNS, RtestDNS, Rtrainbatch, Rtestbatch = load_data(Da, 
                                                                                                                      T, 
                                                                                                                      C, 
                                                                                                                      C_var, 
                                                                                                                      Chi_C, 
                                                                                                                      fdimz, 
                                                                                                                      fdimx, 
                                                                                                                      fdimy, 
                                                                                                                      R_DNS, 
                                                                                                                      HRR_DNS, 
                                                                                                                      R_batch, 
                                                                                                                      HRR_batch, 
                                                                                                                      args, 
                                                                                                                      kwargs) 
# Input and output dimensions
X_dim   = Xtrain.shape[1]
Y_dim   = YtrainDNS.shape[1]
f_dim   = 1

# Load the model
model = FCNN(inp_dim=X_dim,
             out_dim=Y_dim,
             n_layers=args.num_layers,
             neurons_fc=args.neurons_fc,
             hidden_activation=args.activation,
             out_layer_activation=args.output_activation)
print(model)
print("number of parameters {} of layers {}".format(*model.num_parameters()))

# Save diretory
model_dir = "Models/Model_gamma_fcnn{}x{}_{}_taum_{}_tauc_{}_filter_{}".\
    format(args.num_layers, args.neurons_fc, args.output_activation, 
           args.taum_name, args.tauc_name,args.filter_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
#                     verbose=True, threshold=0.0001, threshold_mode='rel',
#                     cooldown=0, min_lr=0, eps=1e-08)

#================================= Network training ===========================
# Fetches a mini-batch of data
def fetch_minibatch(x, y_H, y_L, R_H, R_L, N_batch):
    N = x.shape[0]
    idx = np.random.choice(N, N_batch, replace=False)
    X_batch = x[idx,:]
    Y_L_batch = y_L[idx,:]
    Y_H_batch = y_H[idx,:]
    R_L_batch = R_L[idx,:]
    R_H_batch = R_H[idx,:]
    return X_batch, Y_H_batch, Y_L_batch, R_H_batch, R_L_batch


def mse_loss(y_true, y_pred):
    return torch.mean(torch.square(y_true - y_pred))

def loss_regularization(R_true, R_pred):
    return torch.sum(torch.square(torch.sum(R_true, dim=1) - torch.sum(R_pred, dim=1)))

def test(Xtest, YtestDNS, Ytestbatch, RtestDNS, Rtestbatch ,args):
    x, y_H, y_L, R_H, R_L = fetch_minibatch(Xtest, YtestDNS, Ytestbatch, RtestDNS, Rtestbatch, args.batch_size)
    # y_H = Y_DNS, Y_L = Y_batch, R_H = R_DNS, R_L = R_batch
    x, y_H, y_L, R_H, R_L = x.to(device), y_H.to(device), y_L.to(device), R_H.to(device), R_L.to(device)
    model.eval()
    with torch.no_grad():
        gamma = model(x)
    y_pred = gamma * y_L
    mse = mse_loss(y_H, y_pred)
    rmse = np.sqrt(mse.item())
    R_pred = gamma * R_L
    reg = loss_regularization(R_H, R_pred)
    loss = args.beta1 * mse + args.beta2 * reg
    return loss, rmse
        

# Scaling the Heat release rate dividing by the maximum value ()
Ymax = YtrainDNS.max()
YtrainDNS = YtrainDNS / Ymax
Ytrainbatch = Ytrainbatch / Ymax
YtestDNS = YtestDNS / Ymax
Ytestbatch = Ytestbatch / Ymax

print("Start training network")
tic = time.time()
start_time = tic
save_loss, save_loss_test = [], []
rmse_train, rmse_test = [], []
for it in range(1,args.n_iterations+1):
    model.train()
    
    # sample a batch of data 
    x, y_H, y_L, R_H, R_L = fetch_minibatch(Xtrain, YtrainDNS, Ytrainbatch, RtrainDNS, Rtrainbatch, args.batch_size)
    
    # y_H = Y_DNS, Y_L = Y_batch, R_H = R_DNS, R_L = R_batch
    x, y_H, y_L, R_H, R_L = x.to(device), y_H.to(device), y_L.to(device), R_H.to(device), R_L.to(device) 
    
    #forward propagation 
    model.zero_grad()
    gamma = model(x)
    y_pred = gamma * y_L 
    
    # Compute mse loss (HRR normalized) 
    mse = mse_loss(y_H, y_pred)
    
    # regularization (Chemical source terms) 
    R_pred = gamma * R_L
    reg = loss_regularization(R_H, R_pred)
    
    # Physics-aware constrain loss function
    loss = args.beta1 * mse + args.beta2 * reg
    
    # Backward Step 
    loss.backward()
    optimizer.step()
    
    # Check the model in the test data
    loss_test, rmse_t = test(Xtest, YtestDNS, Ytestbatch, RtestDNS, Rtestbatch ,args)
    
    #  Save the losses
    save_loss.append(loss.item())
    save_loss_test.append(loss_test.item())
    
    # Disp the training process
    if it % 100 == 0:
        # Compute accuracy metrics
        rmse_train.append(np.sqrt(mse.item()))
        rmse_test.append(rmse_t)
        # Scheduler the learning rate
        scheduler.step()
        elapsed = time.time() - start_time
        print('It: %d, Recons loss:%.4e, Reg loss:%.4e' % (it, mse, reg))
        print('Training RMSE:%.4e, Testing RMSE:%.4e, Time: %.2f' % (np.sqrt(mse.item()), rmse_t, elapsed))
        start_time = time.time()
        
    # save the model
    if it == args.n_iterations:
        torch.save(model.state_dict(), model_dir + "/gamma_fcnn_model_iter{}.pth".format(it))


tic2 = time.time()
print("Done training {} iteratiosn in {} seconds"
      .format(args.n_iterations, tic2 - tic))

# Save the training details
np.savetxt(model_dir + "/loss_train.txt", save_loss)
np.savetxt(model_dir + "/loss_test.txt", save_loss_test)
np.savetxt(model_dir + "/rmse_train.txt", rmse_train)
np.savetxt(model_dir + "/rmse_test.txt", rmse_test)

# Plot training and testing loss function
plt.figure(11, dpi=150)
plt.plot(save_loss,'b-',lw=2, label='Train')
plt.plot(save_loss_test,'r--',lw=2, label='Test')
plt.box('True')
plt.grid('True')
plt.yscale("log")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel(r'Loss',fontsize=12)
plt.xlabel(r'Number of Iterations',fontsize=12)
plt.legend(loc='best',fontsize=12)
plt.savefig(model_dir+'/losses_filter{}.jpg'.format(args.filter_name), bbox_inches='tight',dpi=150)

    
######################## TEST MODEL ON WHOLE SET #####################################
# reshape data  
C_              = C.reshape(fdimz*fdimx*fdimy,1)
C_var_          = C_var.reshape(fdimz*fdimx*fdimy,1)
Chi_C_          = Chi_C.reshape(fdimz*fdimx*fdimy,1)
Da_             = Da.reshape(fdimz*fdimx*fdimy,1)
T_              = T.reshape(fdimz*fdimx*fdimy,1)
HRR_batch_      = HRR_batch.reshape(fdimz*fdimx*fdimy,1)

# Scaling the data
T_              = (T_ - T_.mean(0)) / T_.std(0)
Da_             = (np.log10(Da_) - np.log10(Da_).mean(0)) / np.log10(Da_).std(0)
C_var_          = (np.log10(C_var_) - np.log10(C_var_).mean(0)) / np.log10(C_var_).std(0)
Chi_C_          = (np.log10(Chi_C_) - np.log10(Chi_C_).mean(0)) / np.log10(Chi_C_).std(0)

# Input data
X               = np.hstack([Da_, T_, C_, C_var_, Chi_C_])
X = torch.FloatTensor(X) 
start_time      = time.time()
model.eval()
with torch.no_grad():
    gamma = model(X)
gamma_fcnn      = gamma.data.cpu().numpy()
HRR_FCNN        = gamma_fcnn * HRR_batch_
elapsed         = time.time() - start_time
print('Prediction time: %.4f' % (elapsed))

# Reshape to save gamma field
gamma_fcnn = gamma_fcnn.reshape(fdimz,fdimx,fdimy) 
HRR_FCNN = HRR_FCNN.reshape(fdimz,fdimx,fdimy) 

f = h5py.File(model_dir+'/gamma_fcnn_taum_{}_tauc_{}_filter_{}.h5'.format(args.taum_name, args.tauc_name, args.filter_name), 'w')
f.create_dataset('gamma', data=gamma_fcnn)
f.close()

# index to be plot (middle field)
idx = fdimz//2

'Compare the conditional means'
pts = np.linspace(C.min(),C.max(),51)
C_  = C[idx].reshape(fdimx*fdimy)

'Heat release rates'
HRR_DNS_        = HRR_DNS[idx].reshape(fdimx*fdimy)
HRR_batch_      = HRR_batch[idx].reshape(fdimx*fdimy)
HRR_PaSR_       = HRR_PaSR[idx].reshape(fdimx*fdimy)
HRR_FCNN_       = HRR_FCNN[idx].reshape(fdimx*fdimy)
#
cpts, q_dns         = conditional(C_, HRR_DNS_, pts)
cpts, q_batch       = conditional(C_, HRR_batch_, pts)
cpts, q_pasr        = conditional(C_, HRR_PaSR_, pts)
cpts, q_fcnn        = conditional(C_, HRR_FCNN_, pts)


# Plot the HRR conditioned to C
plt.figure(122, figsize=(8,6), dpi=150)
plt.scatter(np.reshape(C,-1),np.reshape(HRR_DNS,-1), s=10, marker='.',c='grey')
plt.plot(cpts, q_dns,'b-o',lw=2, label='DNS')
plt.plot(cpts, q_batch,'r-^',lw=2, label='PaSR$^*$')
plt.plot(cpts, q_pasr,'k-v',lw=2, label='PaSR')
plt.plot(cpts, q_fcnn,'m->',lw=2, label='FCNN')
plt.box('True')
plt.grid('True')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel(r'$\langle \bar{\dot{Q}} \ \| \ \bar{C} \rangle$  [$J.m^{-3}.s^{-1}$]',fontsize=22)
plt.xlabel(r'$\bar{C}$',fontsize=22)
plt.legend(loc='best',fontsize=22)
plt.savefig(model_dir+'/HRR_gamma_fcnn_taum_{}_tauc_{}_filter_{}.jpg'.format(args.taum_name, args.tauc_name, args.filter_name), bbox_inches='tight',dpi=150)

# save args and time taken
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
args_dict['time'] = tic2 - tic
n_params, n_layers = model.num_parameters()
args_dict['num_layers'] = n_layers
args_dict['num_params'] = n_params
with open(model_dir + "/args.txt", 'w') as file:
    file.write(json.dumps(args_dict))

