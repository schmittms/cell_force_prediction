import argparse
from itertools import product as iterprod
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import time
from datetime import datetime

import utils.utils_data_processing as dp
import matplotlib.pyplot as plt

from utils.UNeXt import UNet
from utils.utils_loss import loss_function_dict

import pprint

np.random.seed(11)
torch.manual_seed(11)


if __name__=='__main__':

    ################### Build dataset #######################
    batch_size = 4
    num_workers = 0#4
    n_epochs = 100

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if torch.cuda.is_available() else False
    print("DEVICE AND PIN MEMORY\t:", device, pin_memory)

    # Dataset
    directory  = '/project/vitelli/cell_stress/TractionData_All_16kpa_new_HighForce'
    test_cells = "pax_cell_5,pax_cell_2,myo_cell_6,myo_cell_1,08_cell_1,11_cell_1,myo_cell_4,17_cell_3,myo_cell_5,11_cell_4,10_cell_4,11_cell_2,10_cell_1,myo_cell_3,17_cell_4"


    in_channels  = [[6],] # Example: [[4], [4,6], [4,6,7]]
    out_channels = (2,3) 
    transform_kwargs = {'crop_size': 128,
                        'output_channels': out_channels, 
                        'vector_components': [out_channels, (0,1)],
                        'magnitude_only': False,
                        'angmag': True,
                        'norm_output': {'rescale': 0.25, 'threshold': 0.4},
                        }

    dataset_kwargs = { 
                        'root': directory,
                        'force_load': False,
                        'test_split': 'bycell',
                        'test_cells': test_cells,
                        'in_channels': in_channels, 
                        'out_channels': out_channels, 
                        'transform_kwargs': transform_kwargs,
                        'frames_to_keep': 128,
                        'input_baseline_normalization': 'outside_inside', # Comment on what these do
                        'output_baseline_normalization': 'mean_dataset',
                         }
                        


    dataset = dp.CellDataset( **dataset_kwargs )
    
    train_loader = dataset.get_loader(dataset.train_indices, 
                        batch_size, 
                        num_workers, 
                        pin_memory)
    
    validation_loader = dataset.get_loader(dataset.test_indices, 
                        batch_size, 
                        num_workers, 
                        pin_memory)



    
    ################### Build model #######################
    n_lyr  = 3 # number of downsampling layers
    ds_krnl= 4 # downsample kernel
    n_ch   = 4 # number of channels in the beginning of the network
    n_blocks = 4 # number of ConvNext blocks, wherever ConvNext blocks are used

    prepend_hparams = {'start_channel': 1, 'resnet_channel': n_ch, 'end_channel': n_ch, 'N_blocks': n_blocks,                                         # Args for architecture
                        'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1} # Args for ConvNext blocks
    encoder_hparams = {'n_ch': n_ch, 'n_layers': n_lyr, 'N_node_blocks': n_blocks, 'N_skip_blocks': n_blocks,
                        'downsample_kwargs': {'kernel': ds_krnl, 'activation': 'gelu', 'batchnorm': 1},
                        'interlayer_kwargs': {'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1}
                        }
    decoder_hparams = {'n_layers': n_lyr, 'N_node_blocks': n_blocks, 'upsample_kernel': ds_krnl,
                        'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1}
    append_hparams = {'start_channel': n_ch, 'resnet_channel': n_ch, 'end_channel': 2, 'N_blocks': n_blocks,
                        'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 8, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1} 
    optimizer_hparams = {'LR': 0.001, 'schedule_rate': 0.99}
    loss_hparams = {'loss_type': 'am',
                    'exp_weight': 0.0,
                    'strainenergy_regularization': 0.0,
                    'exp_schedule': {'type': 'linear', 'width': 310, 'e_crit': 30},
                    'reg_schedule': {'type': 'linear', 'width': 310, 'e_crit': 30},
                    'loss_kwargs': {'max_force': 8.}
                   }

    modelname = 'model_0'
    
    logger_params = {'log_dir': f'./tensorboard_logs/${modelname}', 
                     'image_epoch_freq': 10,
                     'image_callbacks': 'vectorfield,hists',
                     'save_model_freq': 20}


    # Actually build model:
    model_kwargs={
                    'input_type':  dp.channel_to_protein_dict['6'], # will use channel 6=zyxin
                    'prepend_hparams': prepend_hparams, 
                    'encoder_hparams': encoder_hparams, 
                    'decoder_hparams': decoder_hparams, 
                    'append_hparams': append_hparams, 
                    'optimizer_hparams': optimizer_hparams,
                    'loss_hparams': loss_hparams,
                    'logger_params': logger_params,
                    'name': 'model_0'}

        
    model = UNet( **model_kwargs, model_idx=0)
    model.to(device)

    ################### TRAINING #######################
    t0 = time.time()
    for e in range(n_epochs):
        loss_values_train = {}
        loss_values_val = {}

        model.reset_running_train_loss()
        model.reset_running_val_loss()

        for sample in train_loader: 
            for key in sample:
                sample[key] = sample[key].to(device)

            model.training_step(sample, epoch=e) # loss.backward() and optimizer step occurs in here

        for sample in validation_loader:
            for key in sample:
                sample[key] = sample[key].to(device)
                
            model.validation_step(sample, epoch=e)
        
        model.scheduler.step()
        model.log_images(epoch=e)
        model.log_scalars(epoch=e) 
    
        print("Epoch %u:\t Time: %0.2f \t(per epoch: %0.2f)"%(e, time.time()-t0, (time.time()-t0)/(e+1)))
    
        # SAVE
        if e%(logger_params['save_model_freq'])==0 or e==n_epochs-1: 
            print("SAVING MODEL")
            torch.save({'model': model.state_dict(),
                        'model_kwargs': model_kwargs,
                        'model_name': model.name,
                        'model_idx': model.index,
                        'dataset_kwargs': dataset_kwargs,
                        'test_cells': dataset.test_cells,
                        }, 
                       os.path.join( model.logdir, 'model.pt') )

    


