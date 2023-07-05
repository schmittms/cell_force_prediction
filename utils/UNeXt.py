import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr

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
    """
    UNet base class.

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

        # Network is composed of four essential parts: prepended layers, encoder layers, skip connection layers, decoder layers, and appended layers. Skip connection layers are created when the encoder layers are created
    
        self.encoder = self.make_encoder(**encoder_hparams) #args:  n_ch, n_layers, N_node_blocks, N_skip_blocks, downsample_kwargs, interlayer_kwargs
        self.decoder = self.make_decoder(**decoder_hparams) #args: n_layers, N_node_blocks, upsample_kernel, **post_concat_kwargs 
        self.prepend = self.make_prepend(**prepend_hparams) #args: start_channel, resnet_channel, end_channel, N_blocks, **layer_kwargs
        self.append = self.make_append(**append_hparams)

        self.optimizer = torch.optim.AdamW([{'params': self.named_grad_parameters()}],
                                             lr=optimizer_hparams['LR'])

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, optimizer_hparams['schedule_rate']) #gamma decay rate

        self.logger_params = logger_params
        self.logdir = logger_params['log_dir']+'_%u'%model_idx 
        self.logger = SummaryWriter( self.logdir ) 
        self.index = model_idx
        self.logger.add_text('Name', self.name, global_step=0) 
        self.metrics_dict = {}



        self.loss_function = loss_function_dict[loss_hparams['loss_type']](**loss_hparams)

        if self.verbose:  print(self) 
        
        self.track_activations=False
        self.angmag = True if 'am' in loss_hparams['loss_type'] else False
        
            
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

     
    def reg_scheduler(self, schedule, x, e):
        """
        Regularization scheduler.
        Modifies base value of regularization parameter x according to schedule, and current epoch e.
        Schedule is a dict with keys ['type', 'e_crit', 'width']. Should be self-explanatory with the examples below.
        """ 
        
        if schedule['type']=='sigmoid':
            e_crit = schedule['e_crit'] # epoch at which switch should take place
            width = schedule['width'] # Width of 2 gives variation over ~10 epochs

            return x/(1+np.exp(-(e-e_crit)/width))

        if schedule['type']=='linear':
            e_crit = schedule['e_crit'] # epoch at which switch should take place
            width = schedule['width'] # Width of 2 gives variation over ~10 epochs

            return x*np.maximum( (e-e_crit)/width, 0)

        if schedule['type']=='none':
            return x 
        
        
    def strainenergy_loss(self, Fpred, Fexp, Uexp, mask):
        """
        Calculate the strain energy of predicted and experimental forces. 
        Strain energy obtained for both by dotting into displacement field U from experiment.
        Optimal additional regularization/loss term. In practice did not seem to make a difference at all.
        """

        if self.angmag:
            xt = Fexp[:,0]*torch.cos(Fexp[:,1])
            yt = Fexp[:,0]*torch.sin(Fexp[:,1])

            xp = Fpred[:,0]*torch.cos(Fpred[:,1])
            yp = Fpred[:,0]*torch.sin(Fpred[:,1])
    
            Fexp = torch.cat([xt.unsqueeze(1), yt.unsqueeze(1)], axis=1)
            Fpred = torch.cat([xp.unsqueeze(1), yp.unsqueeze(1)], axis=1)
    
        W_pred = torch.sum( Fpred*Uexp, axis=1, keepdim=True)
        W_exp = torch.sum( Fexp*Uexp, axis=1, keepdim=True)

        W_pred = W_pred[mask!=0].mean()
        W_exp = W_exp[mask!=0].mean()

        #assert W_exp>=0
        return (W_pred-W_exp).pow(2)



    def forward(self, x, return_input_after_BN=False):
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


    def training_step(self, batch, epoch=None):
        """
        batch is a dict with keys ['zyxin', 'mask', 'output', ('actin')]
        each of those items has shape [B, 1, H, W] except for output which has 2 channels
        """
        self.train()
        self.optimizer.zero_grad()

        prediction = self(self.select_inputs(self.input_type, batch))

    # Calculate loss
        expweight = self.loss_hparams.get('exp_weight')*self.reg_scheduler(  self.loss_hparams.get('exp_schedule'), 1., epoch)
        loss_dict = self.loss_function(prediction, batch['output'], expweight=expweight) 
        # loss_dict contains potential loss values. The one which should be gradded is called 'base_loss'

        strainenergy_loss = self.strainenergy_loss(prediction, batch['output'], batch['displacements'], batch['mask'].bool())

        loss = loss_dict['base_loss'] \
                + self.reg_scheduler(self.loss_hparams.get('reg_schedule'), 
                                     self.loss_hparams.get('strainenergy_regularization'),
                                     epoch) *strainenergy_loss

        if torch.isnan(loss):
            print("LOSS IS NAN")
            print(loss_dict, {'strainenergy_loss': strainenergy_loss})


    # Backprop
        loss.backward()
        self.optimizer.step()

    # Store loss dict values in "running_train_loss" which is used to show scalars in tensorboard
        for x in loss_dict:
            loss_dict[x].detach()
        loss_dict = {**loss_dict, 'strainenergy_loss': strainenergy_loss.sqrt().detach(),
                        'exp_schedule': expweight,
                        'reg_schedule': self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), 1., epoch)}

        if not self.running_train_loss: 
            self.running_train_loss = loss_dict # if empty, then initialize 
        else:
            self.running_train_loss = {key: item + loss_dict[key] for key, item in self.running_train_loss.items()} 
        self.n_training_batches += 1

        return 


    def validation_step(self, batch, epoch=None):
        self.eval()

        with torch.no_grad():
            prediction = self(self.select_inputs(self.input_type, batch))

        # Calculate loss
            loss_dict = self.loss_function(prediction, batch['output']) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

            loss = loss_dict['base_loss'] 
            loss_dict = {**loss_dict}

            expweight = self.loss_hparams.get('exp_weight')*self.reg_scheduler(  self.loss_hparams.get('exp_schedule'), 1., epoch)
            loss_dict = self.loss_function(prediction, batch['output'], expweight=expweight) 
            # Contains loss values. The one which should be gradded is called 'base_loss'

            strainenergy_loss = self.strainenergy_loss(prediction, batch['output'], batch['displacements'], batch['mask'].bool())
            loss = loss_dict['base_loss'] \
                + self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), self.loss_hparams.get('strainenergy_regularization'), epoch)*strainenergy_loss

        # Save loss dict values in "running_val_loss" which is used to show scalars in tensorboard
            loss_dict = {**loss_dict, 'strainenergy_loss': strainenergy_loss.sqrt().detach(),
                            'exp_schedule': expweight,
                            'reg_schedule': self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), 1., epoch)}
            
            if not self.running_val_loss:
                self.running_val_loss = loss_dict
            else:
                self.running_val_loss = {key: item + loss_dict[key] for key, item in self.running_val_loss.items()} 
            
        # Save one prediction from validation set to show full prediction as image in tensorboard. Used in "log_images".
            if self.sample_chosen_for_callback==False:
                self.first_validation_sample = {**batch, 'prediction': prediction.detach()}
                
                if prediction.shape[0]>=2: # make sure size is at least 2
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
            
 
   



######
######
###### Beyond here, just logging functions used to log things in tensorboard.
######
######

    def log_scalars(self, epoch=0, n_batches=0., model_label=None):

    # Log scalars
        train_loss = {key: item/self.n_training_batches for key, item in self.running_train_loss.items()}
        val_loss = {key: item/self.n_validation_batches for key, item in self.running_val_loss.items()}

        for key in train_loss:
            self.logger.add_scalar('Train/%s'%(key), train_loss[key], global_step=epoch) 
            if train_loss[key] < self.metrics_dict['train_'+key]: self.metrics_dict['train_'+key] = train_loss[key]
        for key in val_loss:
            self.logger.add_scalar('Val/%s'%(key), val_loss[key], global_step=epoch) 
            if val_loss[key] < self.metrics_dict['val_'+key]: self.metrics_dict['val_'+key] = val_loss[key]

        return

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

                mag_T = output[b][0] if self.angmag else  np.linalg.norm(output[b], axis=0)
                mag_P = prediction[b][0] if self.angmag else  np.linalg.norm(prediction[b], axis=0)

                ax[b][0].imshow(input[b][0]/input[b][0].max(), origin='lower', **cscheme['input'](input, b))

                ax[b][1].imshow( mag_T, origin='lower', vmax=4, cmap='inferno')
                ax[b][1].quiver(*utils_plot.make_vector_field(*output[b], downsample=20, threshold=0.4, angmag=self.angmag), color='w', width=0.003, scale=20)

                ax[b][2].imshow( mag_P, origin='lower', vmax=4, cmap='inferno')
                ax[b][2].quiver(*utils_plot.make_vector_field(*prediction[b], downsample=20, threshold=0.4, angmag=self.angmag), color='w', width=0.003, scale=20)

    
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


    def draw_force_hists_figure(self, epoch, input, output, prediction, logger):
        
        figscale= self.logger_params.get('figscale', 3)

        nrows = 1 # Force, force conditional, angle,
        ncols = 3

        fig, ax = plt.subplots(1, 3, figsize=(figscale*ncols, figscale*nrows))

        Fmax = 6 
        Fbins = np.linspace(0,Fmax, 101)
        cmap='GnBu'

        with torch.no_grad():
        
            hist_joint = np.zeros((len(Fbins)-1, len(Fbins)-1))
            hist_cond = np.zeros((len(Fbins)-1, len(Fbins)-1))

            for b in range(input.shape[0]): # b for batch
                if torch.is_tensor(input): input = input.cpu()
                if torch.is_tensor(output): output = output.cpu().numpy()
                if torch.is_tensor(prediction): prediction = prediction.cpu().numpy()

                mag_T = output[b][0] if self.angmag else  np.linalg.norm(output[b], axis=0)
                mag_P = prediction[b][0] if self.angmag else  np.linalg.norm(prediction[b], axis=0)

                hist_joint += np.histogram2d(mag_T.ravel(), mag_P.ravel(), bins=(Fbins, Fbins))[0]

            hist_joint = hist_joint.T/np.sum(hist_joint)
        
            p_Fexp = np.sum(hist_joint, axis=1)
            hist_cond = hist_joint/p_Fexp[:,None]

            extent=[Fbins.min(), Fbins.max(), Fbins.min(), Fbins.max()]
            
            cmap_joint = ax[0].imshow(hist_joint, origin='lower', extent=extent, cmap=cmap,
                             norm=pltclr.SymLogNorm(linthresh=1., vmax=np.max(hist_joint)*1e-3, vmin=0))
            cmap_cond = ax[1].imshow(hist_cond, origin='lower', extent=extent, cmap=cmap, vmax=np.max(hist_cond)*1e-2)

            for a in ax:
                xlim=a.get_xlim()
                a.plot(xlim, xlim, 'gray', ls=':')
            
    
            ax[2].semilogy(0.5*(Fbins[1:]+Fbins[:-1]), np.sum(hist_joint, axis=0), label='$F_{exp}$')
            ax[2].semilogy(0.5*(Fbins[1:]+Fbins[:-1]), np.sum(hist_joint, axis=1), label='$F_{pred}$')


        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("hists/model_%u"%self.index, fig, close=True, global_step=epoch)
        return






