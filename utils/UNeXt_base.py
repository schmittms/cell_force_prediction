import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_base_layers import DownsampleLayer, ConvNextCell
from utils.utils_loss import loss_function_dict 
import utils.utils_plot as utils_plot

from torch.utils.tensorboard import SummaryWriter


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape((x.shape[0], -1))

       

class UNet(nn.Module):
    r"""
    Rough approximation of a UNet, which is a linear combination of terms of the form f1 * \nabla (G \star f2)

    Currently, f1 and f2 networks can only take one activation function (respectively). If this is an issue, will try 2-step chains in the future. 

    """

    verbose = False



    def __init__(self,
                input_type,
                prepend_hparams, # Prepended layers, keys=channel_list, stride, kernel, dilation, dropout_rate, activation_function, batchnorm, split
                encoder_hparams, #encoder hparams keys=n_ch, n_layers
                decoder_hparams, #decoder keys=n_layers
                append_hparams, # appended layers keys=(same as prepended)+"split"
                optimizer_hparams, # Learning rate, scheduling, etc.
                loss_hparams, # Regularization terms etc.
                logger_params,
                name,
                model_idx):
        
        super(UNet, self).__init__()

        self.name = name
        self.input_type = input_type

        self.loss_hparams = loss_hparams 
    
        #  def make_encoder(self, n_ch, n_layers, N_node_blocks, N_skip_blocks, downsample_kwargs, interlayer_kwargs):
        self.encoder = self.make_encoder(**encoder_hparams) #n_ch, n_layers, strides_enc, dilations_enc
        # def make_decoder(self, n_layers, N_node_blocks, upsample_kernel, **post_concat_kwargs):
        self.decoder = self.make_decoder(**decoder_hparams) #n_ch, n_layers, strides_enc, dilations_enc

        #def make_prepend(self, start_channel, resnet_channel, end_channel, N_blocks, **layer_kwargs):
        self.prepend = self.make_prepend(**prepend_hparams)
        self.append = self.make_append(**append_hparams)

        self.optimizer = torch.optim.AdamW([{'params': self.named_grad_parameters()}],
                                             lr=optimizer_hparams['LR'])
                                            #{'params': self.encoder.named_grad_parameters()},
                                            #{'params': self.decoder.named_grad_parameters()},
                                            #{'params': self.append.named_grad_parameters()}],

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate']) #gamma decay rate

        self.logger_params = logger_params
        self.logdir = logger_params['log_dir']+'_%u'%model_idx 
        self.logger = SummaryWriter( self.logdir ) 
        self.index = model_idx
        self.logger.add_text('Name', self.name, global_step=0) 
        self.metrics_dict = {}

        #self.hparam_dict = {**optimizer_hparams, 
        #                    'input_type': self.input_type, 
        #                    **{k: i for k,i in net1_hparams.items() if k in ['n_layers', 'dropout_rate', 'n_channels_firstlayer', 'kernel', 'activation_function']},
        #                    **loss_hparams,
        #                    }

#        upsample = nn.Upsample(scale_factor=kernel, mode=mode)
# def __init__(self, in_channel, out_channel, kernel=4, activation='relu', batchnorm=True, bias=True, verbose
# def __init__(self, in_out_channel, kernel=7, stride=1, dilation=1, activation='gelu', batchnorm=True, dropout_rate=0.0, inv_bottleneck_factor=4, verbose=False, bias=True):


        self.loss_function = loss_function_dict[loss_hparams['loss_type']](**loss_hparams)

        if self.verbose:
            print(self) 
        
        self.track_activations=False

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
        if arg_type=='enc_struct':
            keys = ['n_ch', 'n_layers', 'N_node_blocks', 'N_skip_blocks']
            paramtypes = [int, int, int, int]
            return {key: typemap(param) for key,typemap, param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='dec_struct':
        # def make_decoder(self, n_layers, N_node_blocks, upsample_kernel, **post_concat_kwargs):
            keys = ['n_layers', 'N_node_blocks', 'upsample_kernel']
            paramtypes = [int, int, int]
            return {key: typemap(param) for key,typemap, param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='pre_struct':
        #def make_prepend(self, start_channel, resnet_channel, end_channel, N_blocks, **layer_kwargs):
            keys = ['start_channel', 'resnet_channel', 'end_channel', 'N_blocks']
            paramtypes = [int, int, int, int]
            return {key: typemap(param) for key,typemap, param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='convnext_layer':
            keys = ['kernel','stride','inv_bottleneck_factor','dilation','dropout_rate','activation','batchnorm']
            paramtypes = [int, int, int, int, float, str, int] # Last two are actually bools, but beware that mapping bool(str) is true even if str='0'
            return {key: typemap(param) for key,typemap,param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='downsample_layer':
            keys = ['kernel','activation','batchnorm']
            paramtypes = [int, str, int] # Last two are actually bools, but beware that mapping bool(str) is true even if str='0'
            return {key: typemap(param) for key,typemap,param in zip(keys, paramtypes, arg_string.split(','))}
        if arg_type=='loss':
            keys = ['loss_type']
            paramtypes = [str]
            
            rawdict = {s.split(',')[0]: p( s.split(',')[1] ) for s,p in zip(arg_string.split(':'), paramtypes)}
            return rawdict        
            
        if arg_type=='logger':
            keys = ['log_dir', 'image_epoch_freq','image_callbacks', 'figscale','predfig_cscheme', ]
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
        
        inputs = torch.cat(inputs, axis=1)


        if self.verbose: print(" Models: Inputs shape: \t %s, dict entry shape\t%s"%(inputs.shape, dict_['mask'].shape))

        return inputs 
            
    
    def make_prepend(self, start_channel, resnet_channel, end_channel, N_blocks, **layer_kwargs):
        self.prepended_layers = nn.ModuleList()     

        self.prepended_layers.append( nn.Conv2d( start_channel, resnet_channel, kernel_size=3, stride=1, padding=1, bias=True))
        for N in range(N_blocks):
            self.prepended_layers.append(ConvNextCell(resnet_channel, **layer_kwargs))

        self.prepended_layers.append( nn.Conv2d( resnet_channel, end_channel, kernel_size=3, stride=1, padding=1, bias=True))

        return 


    def make_encoder(self, n_ch, n_layers, N_node_blocks, N_skip_blocks, downsample_kwargs, interlayer_kwargs):
        self.layers_encode = nn.ModuleList()
        self.interlayer_cnn = nn.ModuleList()
        
        for i in range(n_layers):

        # DOWNSAMPLE
            downsample = nn.ModuleList()
            for N in range(N_node_blocks):
                downsample.append(ConvNextCell(n_ch, **interlayer_kwargs))
            downsample.append(DownsampleLayer(n_ch, n_ch*2, **downsample_kwargs))
            self.layers_encode.append(downsample)         

        # INTERLAYER SKIP CONNECTION
            interlayer = nn.ModuleList()
            for N in range(N_skip_blocks):
                interlayer.append(ConvNextCell(n_ch,  **interlayer_kwargs))
            self.interlayer_cnn.append(interlayer)

            n_ch *= 2

        self.latent_ch = n_ch
        interlayer = nn.ModuleList()
        for N in range(N_skip_blocks):
            interlayer.append(ConvNextCell(n_ch,  **interlayer_kwargs))
        self.layers_encode.append(interlayer) # The bottom layer         
        return

    def make_decoder(self, n_layers, N_node_blocks, upsample_kernel, **post_concat_kwargs):
        n_ch = self.latent_ch
        
        self.layers_decode = nn.ModuleList()
        for i in range(n_layers):
            single_layer = nn.ModuleList()

            single_layer.append(nn.Upsample(scale_factor=upsample_kernel))
            # Concat after this

            post_concat = nn.ModuleList()
            for N in range(N_node_blocks):
                post_concat.append(ConvNextCell(n_ch+n_ch//2, **post_concat_kwargs))

            single_layer.append(post_concat)

            single_layer.append( nn.Conv2d(n_ch+n_ch//2, n_ch//2, kernel_size=3, stride=1, padding=1, bias=True))
            n_ch = n_ch//2  

            self.layers_decode.append(single_layer)
        return
 
    def make_append(self, start_channel, resnet_channel, end_channel, N_blocks, **layer_kwargs):
        self.appended_layers = nn.ModuleList()     

        self.appended_layers.append( nn.Conv2d( start_channel, resnet_channel, kernel_size=3, stride=1, padding=1, bias=True))
        for N in range(N_blocks):
            self.appended_layers.append(ConvNextCell(resnet_channel, **layer_kwargs))

        self.appended_layers.append( nn.Conv2d( resnet_channel, end_channel, kernel_size=3, stride=1, padding=1, bias=True))


        return 

    def named_grad_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
            if self.verbose: print(f"\t {name}")
        return params




    def forward(self, x, return_input_after_BN=False):
        if self.track_activations: self.f1_tracked = f1.detach().cpu().numpy().copy()
        latents = []


        for cell in self.prepended_layers:
            x = cell(x)
        for L,layer in enumerate(self.layers_encode):
            if L<len(self.layers_encode)-1: # if L=len()-1, it's the last layer, which is the "basement" and shouldn't get a latent 
                Lx = x*1.0
                for block in self.interlayer_cnn[L]:
                    Lx = block(Lx)
                latents.append(Lx)
            for block in layer:
                x = block(x)

        for n, ( layer, latent ) in enumerate( zip(self.layers_decode, latents[::-1])):
            x = layer[0](x)
            x = layer[1][0](torch.cat([x, latent], axis=1)) # Layer 0: upsample, layer 1: resnet, layer 2: final conv
            for block in layer[1][1:]: # Resnet part
                x = block(x)
            x = layer[2](x)

        for cell in self.appended_layers:
            x = cell(x)

        return x        

   
         
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

        loss = loss_dict['base_loss'] 

    # Backprop
        loss.backward()
        self.optimizer.step()

        loss_dict = {**loss_dict}
        
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

            loss = loss_dict['base_loss'] 
            loss_dict = {**loss_dict}

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
            self.logger.add_scalar('Val/%s'%(key), val_loss[key], global_step=epoch) 
            if val_loss[key] < self.metrics_dict['val_'+key]: self.metrics_dict['val_'+key] = val_loss[key]


        return
       # return train_loss, val_loss        

    def log_images(self, epoch=0):
    # Log images
        if epoch%self.logger_params['image_epoch_freq']==0:
            if 'prediction' in self.logger_params['image_callbacks']:
                self.draw_prediction_figure(epoch, 
                                            #self.first_validation_sample[self.input_type],
                                            self.first_validation_sample[self.input_type.split(',')[0]],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )

            if 'vectorfield' in self.logger_params['image_callbacks']:
                self.draw_vectorfields_figure(epoch, 
                                            self.first_validation_sample[self.input_type.split(',')[0]],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
            if 'hists' in self.logger_params['image_callbacks']:
                self.draw_force_hists_figure(epoch, 
                                            #self.first_validation_sample[self.input_type],
                                            self.first_validation_sample[self.input_type.split(',')[0]],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
                

        return
            
 


