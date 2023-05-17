import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_base_layers import CnnCell, DeCnnCell
from utils.utils_loss import loss_function_dict 
import utils.utils_plot as utils_plot


from torch.utils.tensorboard import SummaryWriter


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape((x.shape[0], -1))

#class VanillaCNN(nn.Module): # This will have


class CnnChain(nn.Module):
    """
    Chains convolutions (or deconvolutions) together. Intentionally modular, so an AE could be composed of two chains (one downsampling, one upsampling)
    """

    def __init__(self,
                n_layers,
                n_channels_in,
                n_channels_firstlayer,
                n_channels_out=0,
                channel_factor=1, # This tells how the channels can change through the chain. I.e. CF=2 means channels will double in each layer
                stride=1,
                kernel=3,
                dilation=1,
                dropout_rate=0.0,
                activation_function='relu',
                batchnorm=True,
                split=False): # Whether to split the channels into two disjoint groups.

        super(CnnChain, self).__init__()


    # Set chain attributes
        self.layers = nn.ModuleList()
        self.channels = [    [n_channels_in, n_channels_firstlayer] if L==0 
                        else [int(n_channels_firstlayer*channel_factor**(L-1)), int(n_channels_firstlayer*channel_factor**L)]
                         for L in range(n_layers)]

        if n_channels_out: self.channels[-1][-1] = n_channels_out

        self.dropout_rate=dropout_rate
        
    # Make network

        for channel_pair in self.channels:
            self.layers.append(CnnCell( *channel_pair, 
                                            stride=stride, 
                                            kernel=kernel, 
                                            dilation=dilation,
                                            activation=activation_function, 
                                            batchnorm=batchnorm, 
                                            split_conv=split))


    def forward(self, x):
        for layer in self.layers:
            #print("\tx shape\t", x.shape)
            #print("\tx dtype\t", x.dtype)
            #print("\tx device\t", x.device)
            x = layer(x)
            x = nn.Dropout(p=self.dropout_rate)(x)
        return x

    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params


        

class Approximated_UNet(nn.Module):
    r"""
    Rough approximation of a UNet, which is a linear combination of terms of the form f1 * \nabla (G \star f2)

    Currently, f1 and f2 networks can only take one activation function (respectively). If this is an issue, will try 2-step chains in the future. 

    """

    mtwopii = -2.0j * np.pi
    verbose = False

    def __init__(self,
                net1_hparams, 
                net2_hparams, # For computing function that is convolved 
                kernel_hparams, # For network that learns kernel
                optimizer_hparams, # Learning rate, scheduling, etc.
                loss_hparams, # Learning rate, scheduling, etc.
                logger_params,
                name,
                model_idx):
        
        super(Approximated_UNet, self).__init__()

        self.name = name

        self.input_type = net1_hparams['input_type']

        net1_input_type = net1_hparams.pop('input_type') # Should be string. Some combination (or individual) mask, zyxin, actin
        net2_input_type = net2_hparams.pop('input_type') # Ditto

        self.loss_hparams = loss_hparams 

        self.kernel_hparams = kernel_hparams

        self.net1 = CnnChain(**net1_hparams)
        self.net2 = CnnChain(**net2_hparams)
        self.init_kernels()
        self.final_layer = CnnCell( net2_hparams['n_channels_out'], 1, kernel=1, bias=False, batchnorm=False, activation='none') # This is just a linear combination of any layers at the end 


        self.optimizer = torch.optim.Adam([{'params': self.net1.named_grad_parameters()},
                                        {'params': self.net2.named_grad_parameters()},
                                        {'params': self.final_layer.named_grad_parameters()},
                                        {'params': self.kernels, 'lr': optimizer_hparams['kernel_LR']}],  lr=optimizer_hparams['LR'])

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate']) #gamma decay rate

        self.logger_params = logger_params
        self.logger = SummaryWriter( logger_params['log_dir']+'_%u'%model_idx ) 
        self.index = model_idx
        self.logger.add_text('Name', self.name, global_step=0) 
        self.metrics_dict = {}

        self.hparam_dict = {**optimizer_hparams, 
                            'input_type': self.input_type, 
                            **{k: i for k,i in net1_hparams.items() if k in ['n_layers', 'dropout_rate', 'n_channels_firstlayer', 'kernel', 'activation_function']},
                            **loss_hparams,
                            }

        self.loss_function = loss_function_dict[loss_hparams['loss_type']](**loss_hparams)

        self.verbose = False
    
        self.track_f1f2 = False
        

    @classmethod
    def str_to_dict(cls, arg_string, arg_type):
        """
        Used to initialize model
        
        Struct string is of form: '2,1,32,4,1'
            This is string with 5 integers, 'n_layers', 'n_channels_in', 'n_channels_firstlayer', n_channels_out, channel_factor 
        Layer string is of form: '1,1,1,0.2,relu,1,0'
            String with 7 csvs: 'stride','kernel','dilation','dropout_rate','activation_function','batchnorm','split' 
        logger string is / separated list of key,item pairs: k1,i1/k2,i2...
        """
        if arg_type=='struct':
            keys = ['n_layers', 'n_channels_in', 'n_channels_firstlayer', 'n_channels_out', 'channel_factor']
            paramtypes = [int, int, int, int, int]
            return {key: typemap(param) for key,typemap, param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='layers':
            keys = ['stride','kernel','dilation','dropout_rate','activation_function','batchnorm','split']
            paramtypes = [int, int, int, float, str, int, int] # Last two are actually bools, but beware that mapping bool(str) is true even if str='0'
            return {key: typemap(param) for key,typemap,param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='loss':
            keys = ['loss_type', 'kernel_regularization']
            paramtypes = [str, float]
            
            rawdict = {s.split(',')[0]: p( s.split(',')[1] ) for s,p in zip(arg_string.split(':'), paramtypes)}
            return rawdict        
            
        if arg_type=='logger':
            keys = ['log_dir', 'image_epoch_freq','image_callbacks','figscale','predfig_cscheme']
            paramtypes = [str, int, str, float, str] # The first two are actually lists
            rawdict = {s.split(',')[0]: p( ','.join(s.split(',')[1:]) ) for s,p in zip(arg_string.split(':'), paramtypes)}

            rawdict['image_callbacks'] = [x for x in rawdict['image_callbacks'].split(',')]
            return rawdict
        
            

    def select_inputs(self, str_, dict_):
        """
        String is either mask ; or zyxin ; or mask,zyxin ; etc.
        """
        keys = str_.split(',')
        
        inputs = []
        for key in keys:
            inputs.append(dict_[key])
        
        inputs = torch.cat(inputs, axis=0)

        if self.verbose: print(" Models: Inputs shape: \t %s, dict entry shape\t%s"%(inputs.shape, dict_['mask'].shape))

        return inputs 
            

    def forward(self, x):
    # Make f1 term
        f1 = self.net1(x) # Shape: [B, C_end, H, W]
        f2 = self.net2(x)

        if self.track_f1f2: self.f1_tracked = f1.detach().cpu().numpy().copy()
        if self.track_f1f2: self.f2_tracked = f2.detach().cpu().numpy().copy()

    # Make f2 term
        f2 = torch.fft.fft2( f2, dim=(-2, -1) )
        f2 = f2 * self.kernels.exp()[None, ...]

    # Make convolution terms, \nabla (G\star f2)
        f2 = torch.fft.ifft2(self.mtwopii * self.q[None, None, :, :, :] * f2[:, :, None, :, :], dim=(-2, -1)).real  

        f = f2*f1[:,:,None,:,:] # Shape [B, C2, 2, H, W]

    # Pass through final (linear combo) layer
        fx = f[:, :, 0, :, :] # now shape [B, C2, H, W]
        fy = f[:, :, 1, :, :] # now shape [B, C2, H, W]

        fx = self.final_layer(fx) # Will reduce to shape [B, 1, H, W]
        fy = self.final_layer(fy) # ditto

        f = torch.cat([fx, fy], dim=1) # Shape [B, 2, H, W]

        return f
        
        
    def init_kernels(self):
        """
        Initializes kernel, and q (which is used later for forming the vector)


        kernel shape: [nkernels, L, L]
        q shape: [L, L]
        """
        self.kernels = torch.nn.Parameter(
            torch.empty((self.kernel_hparams['num_kernels'], self.kernel_hparams['crop_size'], self.kernel_hparams['crop_size']),
                        dtype=torch.cfloat),
            requires_grad=True)

        q = torch.fft.fftfreq(self.kernel_hparams['crop_size'])
        self.q = torch.nn.Parameter(
            torch.stack(torch.meshgrid(q, q, indexing='ij'), dim=0),
            requires_grad=False)
        
        torch.nn.init.xavier_uniform_(self.kernels)
        
        return 

    

         
    def draw_vectorfields_figure(self, epoch, input, output, prediction, logger):

        colorscheme_dict = {'none': {'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                                     'output': utils_plot.SymmetricNorm(vmax=None),
                                     'prediction': utils_plot.SymmetricNorm(vmax=None)},
             'individually_normed': {'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                                     'output': utils_plot.SymmetricNorm(vmax='individual'),
                                     'prediction': utils_plot.SymmetricNorm(vmax='individual')},
                            }
        
        figscale= self.logger_params.get('figscale', 2)
        cscheme=colorscheme_dict[ self.logger_params.get('predfig_cscheme', 'individually_normed') ]

        nrows = input.shape[0] # Batch size
        ncols = input.shape[1] + 2

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale*ncols, figscale*nrows), squeeze=False)


        with torch.no_grad():
        
            for b in range(nrows): # b for batch
                if torch.is_tensor(input): input = input.cpu()
                if torch.is_tensor(output): output = output.cpu().numpy()
                if torch.is_tensor(prediction): prediction = prediction.cpu().numpy()

                ax[b][0].imshow(input[b][0]/input[b][0].max(), origin='lower', **cscheme['input'](input, b))

                ax[b][1].imshow(np.linalg.norm(output[b], axis=0), origin='lower', vmax=4, cmap='inferno')
                ax[b][1].quiver(*utils_plot.make_vector_field(*output[b], downsample=20, threshold=0.1), color='w', width=0.003, scale=20)

                ax[b][2].imshow(np.linalg.norm(prediction[b], axis=0), origin='lower', vmax=4, cmap='inferno')
                ax[b][2].quiver(*utils_plot.make_vector_field(*prediction[b], downsample=20, threshold=0.1), color='w', width=0.003, scale=20)

    
        for a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)
        ax[0][1].text(s='Target', **utils_plot.texttop, transform=ax[0][1].transAxes)
        ax[0][2].text(s='Prediction', **utils_plot.texttop, transform=ax[0][2].transAxes)

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("vectorfields/model_%u"%self.index, fig, close=True, global_step=epoch)
        return

       
    def draw_prediction_figure(self, epoch, input, output, prediction, logger):

        colorscheme_dict = {'none': {'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                                     'output': utils_plot.SymmetricNorm(vmax=None),
                                     'prediction': utils_plot.SymmetricNorm(vmax=None)},
             'individually_normed': {'input': utils_plot.PositiveNorm(vmax=0.5, cmap='gray'),
                                     'output': utils_plot.SymmetricNorm(vmax='individual'),
                                     'prediction': utils_plot.SymmetricNorm(vmax='individual')},
                            }
        
        figscale= self.logger_params.get('figscale', 2)
        cscheme=colorscheme_dict[ self.logger_params.get('predfig_cscheme', 'individually_normed') ]

        nrows = input.shape[0] # Batch size
        ncols = input.shape[1] + output.shape[1] + prediction.shape[1]

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale*ncols, figscale*nrows), squeeze=False)

        assert ncols==5, "Number of columns (%u) not equal to 5"%ncols

        with torch.no_grad():
        
            for b in range(nrows): # b for batch
                if torch.is_tensor(input): input = input.cpu().numpy()
                if torch.is_tensor(output): output = output.cpu().numpy()
                if torch.is_tensor(prediction): prediction = prediction.cpu().numpy()

                ax[b][0].imshow(input[b][0]/input[b][0].max(), origin='lower', **cscheme['input'](input, b))

                ax[b][1].imshow(output[b][0], origin='lower', **cscheme['output'](output, b, 0))
                ax[b][2].imshow(output[b][1], origin='lower', **cscheme['output'](output, b, 1))

                ax[b][3].imshow(prediction[b][0], origin='lower', **cscheme['output'](output, b, 0))
                ax[b][4].imshow(prediction[b][1], origin='lower', **cscheme['output'](output, b, 1)) # Use same color scheme as output
    
        for a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)
        ax[0][1].text(s='Target\n(Channel 0)', **utils_plot.texttop, transform=ax[0][1].transAxes)
        ax[0][2].text(s='Target\n(Channel 1)', **utils_plot.texttop, transform=ax[0][2].transAxes)
        ax[0][3].text(s='Prediction\n(Channel 0)', **utils_plot.texttop, transform=ax[0][3].transAxes)
        ax[0][4].text(s='Prediction\n(Channel 1)', **utils_plot.texttop, transform=ax[0][4].transAxes)


        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("predictions/model_%u"%self.index, fig, close=True, global_step=epoch)
        return

    def draw_kernel_figure(self, epoch, logger):

        figscale= self.logger_params.get('figscale', 2)

        nrows = 1
        ncols = self.kernels.shape[0]

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale*ncols, figscale*nrows), squeeze=False)

        with torch.no_grad():
        
            for b in range(nrows): # b for batch
                for c in range(ncols):
                    k_plot = torch.fft.ifftshift(self.kernels[c], axis=(-1,-2)).abs().cpu().numpy()
                    ax[b][c].imshow(k_plot/k_plot.max(), origin='lower', cmap='inferno', norm=utils_plot.lognorm(linthresh=k_plot.mean()*1e-1/k_plot.max(), vmin=0))

        for a in ax.flat: a.axis('off')

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("kernels/model_%u"%self.index, fig, close=True, global_step=epoch)
        return


    def draw_f1f2_figure(self, epoch, input, logger):

        figscale= self.logger_params.get('figscale', 2)

        nrows = input.shape[0] # Batch size
        ncols = input.shape[1] + self.f1_tracked.shape[1] + self.f2_tracked.shape[1]

        fig, ax = plt.subplots(nrows, ncols, figsize=(figscale*ncols, figscale*nrows), squeeze=False)

        with torch.no_grad():
        
            for b in range(nrows): # b for batch
                if torch.is_tensor(input): input = input.cpu().numpy()

                ax[b][0].imshow(input[b][0]/input[b][0].max(), origin='lower', **utils_plot.PositiveNorm(vmax=0.5, cmap='gray')(input, b))

                for f1_ch in range(self.f1_tracked.shape[1]):
                    ax[b][1+f1_ch].imshow(self.f1_tracked[b][f1_ch]/np.max(np.abs(self.f1_tracked[b][f1_ch])), origin='lower', vmax=0.7, vmin=-0.7)

                    ax[b][1+f1_ch].text(s="Max: %0.2f\nMin: %0.2f"%(self.f1_tracked[b][f1_ch].max(),self.f1_tracked[b][f1_ch].min()),
                                     **utils_plot.texttopright, transform=ax[b][1+f1_ch].transAxes)
                    ax[b][1+f1_ch].text(s="F1 (Ch. %u)"%(f1_ch), **utils_plot.texttop, transform=ax[b][1+f1_ch].transAxes)
                

                for f2_ch in range(self.f2_tracked.shape[1]):
                    ax[b][1+self.f1_tracked.shape[1]+f2_ch].imshow(self.f2_tracked[b][f2_ch]/np.max(np.abs(self.f2_tracked[b][f2_ch])), origin='lower', vmax=0.7, vmin=-0.7)

                    ax[b][1+self.f1_tracked.shape[1]+f2_ch].text(s="Max: %0.2f\nMin: %0.2f"%(self.f2_tracked[b][f2_ch].max(),self.f2_tracked[b][f2_ch].min()), 
                                                            **utils_plot.texttopright, transform=ax[b][1+self.f1_tracked.shape[1]+f2_ch].transAxes)
                    ax[b][1+self.f1_tracked.shape[1]+f2_ch].text(s="F2 (Ch. %u)"%(f2_ch), **utils_plot.texttop, transform=ax[b][1+self.f1_tracked.shape[1]+f2_ch].transAxes)
    
        for a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)


        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("F1F2/model_%u"%self.index, fig, close=True, global_step=epoch)
        return


    def training_step(self, batch, epoch=None):
        """
        batch is a dict with keys ['zyxin', 'mask', 'output', ('actin')]
        each of those items has shape [B, 1, H, W] except for output which has 2 channels
        """
        self.train()
        self.optimizer.zero_grad()

        prediction = self(self.select_inputs(self.input_type, batch))
    # Calculate loss
        loss_dict = self.loss_function(prediction, batch['output']) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'
        kernel_loss = (torch.conj(self.kernels.exp()) * self.kernels.exp()).real.abs().sum()

        loss = loss_dict['base_loss'] + self.loss_hparams.get('kernel_regularization')*kernel_loss 

    # Backprop
        loss.backward()
        self.optimizer.step()

        loss_dict = {**loss_dict, 'kernel_loss': kernel_loss}
        
        if not self.running_train_loss:
            self.running_train_loss = loss_dict
        else:
            self.running_train_loss = {key: item + loss_dict[key] for key, item in self.running_train_loss.items()} 
        self.n_training_batches += 1

        return 

    def validation_step(self, batch, epoch=None):
        self.eval()

        with torch.no_grad():

            if self.sample_chosen_for_callback==False: self.track_f1f2=True ## SWITCH: Save f1 and f2 so we can plot if necessary
            prediction = self(self.select_inputs(self.input_type, batch))
            if self.sample_chosen_for_callback==False: self.track_f1f2=False ## SWITCH: Save f1 and f2 so we can plot if necessary

        # Calculate loss
            loss_dict = self.loss_function(prediction, batch['output']) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'
            kernel_loss = (torch.conj(self.kernels.exp()) * self.kernels.exp()).real.abs().sum()

            loss = loss_dict['base_loss'] + self.loss_hparams.get('kernel_regularization')*kernel_loss 

            loss_dict = {**loss_dict, 'kernel_loss': kernel_loss}

            if not self.running_val_loss:
                self.running_val_loss = loss_dict
            else:
                self.running_val_loss = {key: item + loss_dict[key] for key, item in self.running_val_loss.items()} 
            
            if self.sample_chosen_for_callback==False:
                self.first_validation_sample = {**batch, 'prediction': prediction}
                
                if prediction.shape[0]>=2:
                    self.sample_chosen_for_callback=True
        
        self.n_validation_batches += 1

        return

    
    def reset_running_train_loss(self):
        self.running_train_loss = {}
        self.n_training_batches=0
        self.sample_chosen_for_callback=False
        return

    def reset_running_val_loss(self):
        self.running_val_loss = {}
        self.n_validation_batches=0
        self.sample_chosen_for_callback=False
        return

    def log_scalars(self, epoch=0, n_batches=0., model_label=None):

    # Log scalars
        train_loss = {key: item/self.n_training_batches for key, item in self.running_train_loss.items()}
        val_loss = {key: item/self.n_validation_batches for key, item in self.running_val_loss.items()}

        
        if not self.metrics_dict:
            self.metrics_dict = {'train_'+key: train_loss[key] for key in train_loss}# Used for logging to hparams
            self.metrics_dict = {**self.metrics_dict, **{'val_'+key: val_loss[key] for key in val_loss} }# Used for logging to hparams

        for key in train_loss:
            self.logger.add_scalar('Train/%s'%(key), train_loss[key], global_step=epoch) 
            if train_loss[key] < self.metrics_dict['train_'+key]: self.metrics_dict['train_'+key] = train_loss[key]
        for key in val_loss:
            self.logger.add_scalar('Val/%s'%(key), train_loss[key], global_step=epoch) 
            if val_loss[key] < self.metrics_dict['val_'+key]: self.metrics_dict['val_'+key] = val_loss[key]


        return
       # return train_loss, val_loss        

    def log_images(self, epoch=0):
    # Log images
        if epoch%self.logger_params['image_epoch_freq']==0:
            if 'prediction' in self.logger_params['image_callbacks']:
                self.draw_prediction_figure(epoch, 
                                            self.first_validation_sample[self.input_type],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
            if 'kernel' in self.logger_params['image_callbacks']:
                self.draw_kernel_figure(epoch, self.logger )

            if 'vectorfield' in self.logger_params['image_callbacks']:
                self.draw_vectorfields_figure(epoch, 
                                            self.first_validation_sample[self.input_type],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
            if 'f1f2' in self.logger_params['image_callbacks']:
                self.draw_f1f2_figure(epoch, self.first_validation_sample[self.input_type], self.logger )

        return
            
    def log_histograms(self, epoch=0):
    # Log images
        if epoch%self.logger_params['image_epoch_freq']==0:
            if 'prediction' in self.logger_params['image_callbacks']:
                self.draw_prediction_figure(epoch, 
                                            self.first_validation_sample[self.input_type],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
        return
 


class UNet(nn.Module):
    def __init__(self, 
            n_channels_in, 
            n_layers, 
            strides=1, 
            kernels=3, 
            dilations=1, 
            prepend_append=[[],[]], 
            dropout_rate=0.2,
            lr=0.001, 
            gamma=0.95,
            name=None,
            angmag=False,
            app_activation='none',
            app_split=False
            ):

        super(UNet, self).__init__()
        
        if name is None:
            self.name = 'unet_%uch_in_%ulayers.pt' % (n_channels_in, n_layers)
        else: 
            self.name=name
        
        self.angmag = angmag
        self.dropout_rate = dropout_rate


        strides_enc = [strides[i] if len(strides)>1 else strides[0] for i in range(n_layers)]
        strides_dec = [strides[-i] if len(strides)>1 else strides[0] for i in range(n_layers)]
        dilations_enc = [dilations[i] if len(dilations)>1 else dilations[0] for i in range(n_layers)]
        dilations_dec = [dilations[-i] if len(dilations)>1 else dilations[0] for i in range(n_layers)]
        kernel = kernels[0] 
        self.kernel = kernels[0] 
        print("UNET SELF.KERNEL :", self.kernel)
    
        self.incoming_channel = prepend_append[0][0]
        self.prepend_append_layers(prepend_append, app_split, app_activation)
        self.encoder(n_channels_in, n_layers, strides_enc, dilations_enc) # Makes encoder
        self.decoder(n_channels_in, n_layers, strides_dec, dilations_dec) # Makes decoder
        
        self.optimizer = torch.optim.Adam(self.named_grad_parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma) #gamma decay rate
        self.runcount = 0
        
        #self.logger = SummaryWriter()

    def prepend_append_layers(self, prepend_append, app_split, app_activation):
        self.prepended_layers = nn.ModuleList()     
        self.appended_layers = nn.ModuleList()
        split_enc = False   
        for i, (ch, ch_nxt) in enumerate(zip(prepend_append[0][:-1], prepend_append[0][1:])):
            if ch>1: split=split_enc
            else: split=False
            self.prepended_layers.append(CnnCell(ch, ch_nxt, stride=1, kernel=3, split_conv=False))
        for i, (ch, ch_nxt) in enumerate(zip(prepend_append[1][:-1], prepend_append[1][1:])):
            if self.angmag:
                self.appended_layers.append(CnnCell(ch, ch_nxt, stride=1, kernel=3, activation='split', batchnorm=False, split_conv=app_split))
            else:
                self.appended_layers.append(CnnCell(ch, ch_nxt, stride=1, kernel=3, activation=app_activation, batchnorm=False, split_conv=app_split))
        return 
    
    def encoder(self, n_ch, n_layers, strides_enc, dilations_enc):
        self.layers_encode = nn.ModuleList()
        self.interlayer_cnn = nn.ModuleList()
        
        for i in range(n_layers):
            cell_encode = nn.ModuleList()
            cell_encode.append(CnnCell(n_ch, n_ch*2, stride=strides_enc[i], kernel=3, dilation=dilations_enc[i], split_conv=False))         
            n_ch *= 2
            cell_encode.append(CnnCell(n_ch, n_ch, stride=1, kernel=3, dilation=1, split_conv=False))
            
            self.layers_encode.append(cell_encode)
            self.interlayer_cnn.append(CnnCell(n_ch, n_ch, stride=1, kernel=1, dilation=1, split_conv=False))

        self.latent_ch = n_ch
        
        return

    def decoder(self, n_channels_in, n_layers, strides_dec, dilations_dec):
        n_ch = self.latent_ch
        
        self.layers_decode = nn.ModuleList()
        split_dec = False
        for i in range(n_layers):
            cell_decode = nn.ModuleList()
            cell_decode.append(DeCnnCell(n_ch, n_ch//2, stride=strides_dec[i], kernel=3, dilation=dilations_dec[i], split_conv=split_dec))
            if n_ch//2 != n_channels_in: 
                cell_decode.append(DeCnnCell(n_ch, n_ch//2, stride=1, kernel=3, dilation=1, split_conv=split_dec))
            n_ch = n_ch//2  
            self.layers_decode.append(cell_decode)  
        return
    

    def forward(self, x):
        """
        x shape: [B, C, H, W], with C=1 
        """
        latents = []
        for cell in self.prepended_layers:
            x = cell(x)
        for L,layer in enumerate(self.layers_encode):
            for cell in layer: 
                x = cell(x)
            if L<len(self.layers_encode)-1:
                latents.append(self.interlayer_cnn[L](x))
        for n, ( layer, latent ) in enumerate( zip(self.layers_decode, latents[::-1])):
            x = layer[0](x)
            x = torch.cat((x, latent), axis=1)
            x = nn.Dropout(p=self.dropout_rate)(x) # used in xy predictions
            x = layer[1](x)

        x = self.layers_decode[-1][0](x)
        for cell in self.appended_layers:
            x = cell(x)

        return x        
    
    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params


class UNet_Reduced(UNet):    

    def encoder(self, n_ch, n_layers, strides_enc, dilations_enc):
        self.layers_encode = nn.ModuleList()
        self.interlayer_cnn = nn.ModuleList()
        
        for i in range(n_layers):
            self.interlayer_cnn.append(CnnCell(n_ch, n_ch, stride=1, kernel=self.kernel, dilation=1, split=False))
            self.layers_encode.append(CnnCell(n_ch, n_ch*2, stride=strides_enc[i], kernel=self.kernel, dilation=dilations_enc[i], split=False))         
            n_ch *= 2

        self.latent_ch = n_ch
        self.layers_encode.append(CnnCell(n_ch, n_ch, stride=1, kernel=self.kernel, dilation=dilations_enc[i], split=False)) # The bottom layer         
        
        return

    def decoder(self, n_channels_in, n_layers, strides_dec, dilations_dec):
        n_ch = self.latent_ch
        
        self.layers_decode = nn.ModuleList()
        split_dec = False
        for i in range(n_layers):
            self.layers_decode.append(DeCnnCell(n_ch, n_ch//2, stride=strides_dec[i], kernel=self.kernel, dilation=dilations_dec[i], split=split_dec))
            n_ch = n_ch//2  
        return
    

    def forward(self, x, return_input_after_BN=False):
        latents = []

        #x = self.whiten(x)
        if return_input_after_BN: 
            x_postBN = x
            x_postBN.retain_grad()

        for cell in self.prepended_layers:
            x = cell(x)
        for L,layer in enumerate(self.layers_encode):
            if L<len(self.layers_encode)-1: # if L=len()-1, it's the last layer, which is the "basement" and shouldn't get a latent 
                latents.append(self.interlayer_cnn[L](x))
            x = layer(x)

        for n, ( layer, latent ) in enumerate( zip(self.layers_decode, latents[::-1])):
            x = layer(x)
            x = x*latent
            x = nn.Dropout(p=self.dropout_rate)(x) # used in xy predictions

        for cell in self.appended_layers:
            x = cell(x)

        if return_input_after_BN: 
            return x, x_postBN
        else:
            return x        



