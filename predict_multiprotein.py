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
from utils.utils_data_processing_base import SubsetSampler

import matplotlib.pyplot as plt

from utils.UNeXt import UNet

from utils.utils_loss import loss_function_dict
from utils.utils_callbacks import CallbackPlot, scalar_callback_dict

import pprint

np.random.seed(11)
torch.manual_seed(11)


if __name__=='__main__':
    ap = ArgumentParser()
    ap.add_argument('-v', '--verbose', type=int, default=0)
    ap.add_argument('--save_model_freq', type=int, default=10)

# Data
    ap.add_argument('--directory', type=str, default='')
    ap.add_argument('--out_dir', type=str, default=r'./')
    ap.add_argument('--testtrain_split', type=str, default='') # choices: bycell, bycell and time, etc.
    # Sample formatting
    ap.add_argument('--in_channels', type=str, default='')
    ap.add_argument('--out_channels', type=str, default='')
    ap.add_argument('--test_cell', type=str, default='')
    ap.add_argument('--frames_to_keep', type=int, default=-1)
    # Transforms
    ap.add_argument('--crop_size', type=int, default=200)
    ap.add_argument('--magnitude_only', type=int, default = 0) # bool
    ap.add_argument('--angmag', type=int, default = 0) # bool
    ap.add_argument('--normalization_output', type=str, default='rescale,1./threshold,0.0') # will be transformed to dict
    ap.add_argument('--perturb_input', type=str, default='') # will be transformed to list
    ap.add_argument('--perturb_output', type=str, default='') # ditto
    ap.add_argument('--add_noise', type=str, default='') # will be transformed to dict. should be key,item/key2,item2/...
    # Normalization that occurs via lookup, i.e. non-transforms which are applied nonuniformly to images
    ap.add_argument('--input_baseline_normalization', type=str, choices=['none', 'outside_inside', 'outside_max', 'outside_inside_actin', 'totally_normalize'], default='')
    ap.add_argument('--output_baseline_normalization', type=str, choices=['mean','mean_dataset','none'], default='')


# Training and loss
    """

    optimization hparams: 'kernel_LR', 'LR'
    """
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--loss_hparams', type=str, default='') # List of params separated by colon. Keys: 'loss_type', 'kernel_regularization'. 
    ap.add_argument('--logger_params', type=str, default='') # key,item pairs separated by /. keys: 'image_epoch_freq','image_callbacks','figscale'(default 4),'predfig_cscheme'(defaulted) 
    # Optimization params
    ap.add_argument('--optim_hparams', type=str, default='') # This will be list of kernelLR,LR,schedule_rate/kernelLR2,LR2,schedulerate2 (etc.). So must have at least three comma separated values

# Model
    """
    Each argument is a string delimited list of arguments to be passed to the model __init__. Note that you can have multiple models by passing list separated by /. 
    """
    ap.add_argument('--prepend_struct', type=str, default='1,8') # # This is string with 4 ints: start_channel, resnet_channel, end_channel, N_blocks 
    ap.add_argument('--prepend_layer_args', type=str, default='7,1,4,1,0.0,gelu,0' ) # ConvNext
                    # ['kernel','stride','inv_bottleneck_factor','dilation','dropout_rate','activation','batchnorm']

    ap.add_argument('--encoder_struct', type=str, default='8,3,3,3') # n_ch, n_layers, n_node_blocks, n_skip_blocks 
    ap.add_argument('--decoder_struct', type=str, default='3,3,4') # n_layers, n_node_blocks, upsample_kernel 

    ap.add_argument('--encoder_dwnsmpl_args', type=str, default='4,gelu,1' ) # kernel, activation, batchnorm (bool) 
    ap.add_argument('--encoder_intrlyr_args', type=str, default='7,1,4,1,0.0,gelu,0' ) # ConvNext
    ap.add_argument('--decoder_layer_args', type=str, default='7,1,4,1,0.0,gelu,0' ) # ConvNext

    ap.add_argument('--append_struct', type=str, default='8,8,5') # This is string with 5 integers, 'n_layers', 'n_channels_in', 'n_channels_firstlayer', n_channels_out, channel_factor 
    ap.add_argument('--append_layer_args', type=str, default='7,1,4,1,0.0,gelu,0' ) # ConvNext


    args = ap.parse_args()

    # Remove spaces from args that are strings
    for a in vars(args):
        if isinstance( getattr(args, a), str):
            setattr(args, a, getattr(args, a).replace(' ', ''))
            
    
# Process model hyperparams
    prepend_hparams=[]
    encoder_hparams=[]
    decoder_hparams=[]
    append_hparams = []
    optimizer_hparams = []
    loss_hparams = []
    modelname_strings = []
    
    for (p_strct,e_strct,d_strct),a_strct,(p_lyr,e_dwn,e_int),(d_lyr,a_lyr), optprm,lossprm in iterprod(
                                                                                        zip(args.prepend_struct.split('/'),args.encoder_struct.split('/'),args.decoder_struct.split('/')), 
                                                                                        args.append_struct.split('/'), 
                                                                                        zip(args.prepend_layer_args.split('/'),args.encoder_dwnsmpl_args.split('/'),args.encoder_intrlyr_args.split('/')),
                                                                                        zip(args.decoder_layer_args.split('/'),args.append_layer_args.split('/')),
                                                                                        args.optim_hparams.split('/'), args.loss_hparams.split('/') ):
            
    
        prepend_hparams.append( { **UNet.str_to_dict(p_strct, 'pre_struct'), **UNet.str_to_dict(p_lyr, 'convnext_layer')} )
        encoder_hparams.append( { **UNet.str_to_dict(e_strct, 'enc_struct'), 'downsample_kwargs': UNet.str_to_dict(e_dwn, 'downsample_layer'), 'interlayer_kwargs': UNet.str_to_dict(e_int, 'convnext_layer')} )
        decoder_hparams.append( { **UNet.str_to_dict(d_strct, 'dec_struct'), **UNet.str_to_dict(d_lyr, 'convnext_layer')} )
        append_hparams.append( { **UNet.str_to_dict(a_strct, 'pre_struct'), **UNet.str_to_dict(a_lyr, 'convnext_layer')} )
        optimizer_hparams.append( {s.split(',')[0]: float(s.split(',')[1]) for s in optprm.split(':') })
        loss_hparams.append( UNet.str_to_dict(lossprm, 'loss') )

        modelname_strings.append( 'p-S'+p_strct+'-L'+p_lyr+'_e-S'+e_strct+'-L'+e_dwn+'_d-L'+d_lyr+'_a-S'+a_strct+'-L'+a_lyr+'_'+optprm+'_'+lossprm )
    
    logger_params = UNet.str_to_dict( args.logger_params, 'logger' )
    
       
################### SETUP #######################
# Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pin_memory = True if torch.cuda.is_available() else False
    print("DEVICE AND PIN MEMORY\t:", device, pin_memory)


# Dataset
    in_channels=[[int(c) for c in in_chs.split(',')] for in_chs in args.in_channels.split(r'/')]  # Can be multiple in channels 
    out_channels=tuple([int(c) for c in args.out_channels.split(',')]) # Can only be one set of out channels
    transform_kwargs = {'crop_size': int(args.crop_size),
                        'output_channels': out_channels, 
                        'vector_components': [out_channels, (0,1)],
                        'magnitude_only': bool(int(args.magnitude_only)),
                        'angmag': bool(int(args.angmag)),
                        **dp.args_to_transform_kwargs(norm_output=args.normalization_output, 
                                                                        perturb_input=args.perturb_input, 
                                                                        perturb_output=args.perturb_output, add_noise=args.add_noise)
                        }
    dataset_kwargs = { 
                        'root': args.directory,
                        'force_load': False,
                        'test_split': args.testtrain_split,
                        'test_cells': args.test_cell,
                        'in_channels': in_channels, 
                        'out_channels': out_channels, 
                        'transform_kwargs': transform_kwargs, # was args
                        'frames_to_keep': args.frames_to_keep,
                        'input_baseline_normalization': args.input_baseline_normalization,
                        'output_baseline_normalization': args.output_baseline_normalization,
                        
                         }
                        

    if args.verbose: pprint.pprint(dataset_kwargs, width=1)

    dataset = dp.CellDataset( **dataset_kwargs )
    
    train_loader = dataset.get_loader(dataset.train_indices, 
                        args.batch_size, 
                        args.num_workers, 
                        pin_memory)
    
    validation_loader = dataset.get_loader(dataset.test_indices, 
                        args.batch_size, 
                        args.num_workers, 
                        pin_memory)
    
# Model
    models = []
    model_kwargs_all = []

    model_idx = 0
    for input_type, (pp, ep, dep, ap, optp, lossp, name) in iterprod(args.in_channels.split('/'), zip(prepend_hparams, 
                                                                                                    encoder_hparams, 
                                                                                                    decoder_hparams, 
                                                                                                    append_hparams, 
                                                                                                    optimizer_hparams, loss_hparams, modelname_strings)):


        pp['start_channel'] = len(input_type.split(','))
        model_kwargs={
                        'input_type':  dp.channel_to_protein_dict[input_type],
                        'prepend_hparams': pp, 
                        'encoder_hparams': ep, 
                        'decoder_hparams': dep, 
                        'append_hparams': ap, 
                        'optimizer_hparams': optp,
                        'loss_hparams': lossp,
                        'logger_params': logger_params,
                        'name': name}

            
        if args.verbose: pprint.pprint(model_kwargs, width=1)
        models.append( UNet( **model_kwargs, model_idx=model_idx))
        model_kwargs_all.append( model_kwargs)
        model_idx += 1

    for m in models:
        m.to(device)


   # logger = SummaryWriter( logger_params['log_dir'] ) 

################### TRAINING #######################
    t0 = time.time()
    for e in range(args.epochs):

        loss_values_train = {}
        loss_values_val = {}

        for m in models:
            m.reset_running_train_loss()
            m.reset_running_val_loss()

        t1 = time.time()
        for nt, sample in enumerate(train_loader): # This loop is critical: we want to load each sample only once, then pass to models (rather than vice versa)
            for key in sample:
                sample[key] = sample[key].to(device)

            for m in models:
                m.training_step(sample, epoch=e)
        t2 = time.time()

        for nv, sample in enumerate(validation_loader): # This loop is critical: we want to load each sample only once, then pass to models (rather than vice versa)
            for key in sample:
                sample[key] = sample[key].to(device)
            for m in models:
                m.validation_step(sample, epoch=e)
        t3 = time.time()
        
        for m_idx, m in enumerate(models):
            m.scheduler.step()
            m.log_images(epoch=e)
            m.log_scalars(epoch=e) # losstrain is dict {'loss1': value, 'loss2': value}, same for lossval
    
        print("Epoch %u:\t Time: %0.2f \t(per epoch: %0.2f)\t(train/sample: %0.2f)\t(val/sample: %0.2f)"%(e, 
                                                                                                    time.time()-t0, 
                                                                                                    (time.time()-t0)/(e+1), 
                                                                                                    (t2-t1)/(nt+1)/len(models),
                                                                                                    (t3-t2)/(nv+1)/len(models) ))#, trainlog_sorted_by_losstype, vallog_sorted_by_losstype)
    

# Save

    for m in range(len(models)):
        #models[m].logger.add_hparams(models[m].hparam_dict, models[m].metrics_dict, hparam_domain_discrete=None, run_name='model_%u'%models[m].index)
        torch.save({'model': models[m].state_dict(),
                    'model_kwargs': model_kwargs_all[m],
                    'model_name': models[m].name,
                    'model_idx': models[m].index,
                    'dataset_kwargs': dataset_kwargs,
                    'all_args': vars(args)}, 
                    os.path.join( models[m].logdir, 'model.pt') )


# Evaluate Model
        
    print(dataset.test_cells)
    print(np.unique( dataset.info.folder))

    test_cells = dataset.test_cells['test_cells']
    index_train = np.asarray( dataset.info[ ~dataset.info.folder.isin(test_cells) ].index)
    index_test = np.asarray( dataset.info[ dataset.info.folder.isin(test_cells) ].index)

    print("Train index", index_train)
    print("Test index", index_test)

    sampler_train = SubsetSampler(index_train)
    sampler_test = SubsetSampler(index_test)
    loader_train = torch.utils.data.DataLoader(dataset, 
        batch_size=16,
        shuffle=False,
        sampler=sampler_train,
        pin_memory=True)
    loader_test = torch.utils.data.DataLoader(dataset, 
        batch_size=16,
        shuffle=False,
        sampler=sampler_test,
        pin_memory=True)

    dict_of_lossLists_train = {}
    dict_of_lossLists_test = {}
    for m, model in enumerate(models):
        model.eval()
        dict_of_lossLists_train[m] = []
        dict_of_lossLists_test[m] = []

    t0 = time.time()
    for t, sample in enumerate(loader_test):
        for key in sample:
            sample[key] = sample[key].to(device)

        for m,model in enumerate(models):
            with torch.no_grad():
                prediction = model(model.select_inputs(model.input_type, sample))

                #print(sample['mask'].shape)

            # Calculate loss
                loss_dict = model.loss_function(prediction, sample['output'], expweight=0., batch_avg=False) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

                loss_dict = {**loss_dict, 'model': model.logdir}

                dict_of_lossLists_test[m].append(loss_dict)

    print(time.time()-t0)

    #print(dict_of_lossLists_test.keys())

    t0 = time.time()
    for t, sample in enumerate(loader_train):
        for key in sample:
            sample[key] = sample[key].to(device)

        for m,model in enumerate(models):
            with torch.no_grad():
                prediction = model(model.select_inputs(model.input_type, sample))

                #print(sample['mask'].shape)

            # Calculate loss
                loss_dict = model.loss_function(prediction, sample['output'], expweight=0., batch_avg=False) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

                loss_dict = {**loss_dict, 'model': model.logdir}

                dict_of_lossLists_train[m].append(loss_dict)

    print(time.time()-t0)


    


# SAVE

    print("SAVING MODEL")
    for m in range(len(models)):
        #models[m].logger.add_hparams(models[m].hparam_dict, models[m].metrics_dict, hparam_domain_discrete=None, run_name='model_%u'%models[m].index)
        torch.save({'model': models[m].state_dict(),
                    'model_kwargs': model_kwargs_all[m],
                    'model_name': models[m].name,
                    'model_idx': models[m].index,
                    'dataset_kwargs': dataset_kwargs,
                    'test_cells': dataset.test_cells,
                    'all_args': vars(args),
                    'lossList_test': dict_of_lossLists_test[m], 
                    'lossList_train': dict_of_lossLists_train[m]}, 
                   os.path.join( models[m].logdir, 'model.pt') )

    
    # At very end, print model summary. 
    print(torch.cuda.memory_summary())


