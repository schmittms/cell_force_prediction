import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.UNeXt import UNet
import utils.utils_plot as utils_plot
       
from torch.utils.tensorboard import SummaryWriter

class r_UNet(UNet):
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

    def training_step(self, batch, epoch=None):
        """
        batch is a dict with keys ['zyxin', 'mask', 'output', ('actin')]
        each of those items has shape [B, 1, H, W] except for output which has 2 channels
        """
        self.train()
        self.optimizer.zero_grad()

        prediction = self(batch['output'])

    # Calculate loss
        loss_dict = self.loss_function(prediction, batch['zyxin'], expweight=None) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

        loss = loss_dict['base_loss']

        if torch.isnan(loss):
            print("LOSS IS NAN")


    # Backprop
        loss.backward()
        self.optimizer.step()

        for x in loss_dict:
            loss_dict[x].detach()

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
            prediction = self(batch['output'])
            if self.sample_chosen_for_callback==False: self.track_f1f2=False ## SWITCH: Save f1 and f2 so we can plot if necessary


        # Calculate loss
            loss_dict = self.loss_function(prediction, batch['zyxin'], expweight=None) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'
            loss = loss_dict['base_loss'] 

            
            if not self.running_val_loss:
                self.running_val_loss = loss_dict
            else:
                self.running_val_loss = {key: item + loss_dict[key] for key, item in self.running_val_loss.items()} 
            
            if self.sample_chosen_for_callback==False:
                self.first_validation_sample = {**batch, 'prediction': prediction.detach()}
                
                if prediction.shape[0]>=2:
                    self.sample_chosen_for_callback=True
        
        self.n_validation_batches += 1

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

                mag_F = np.linalg.norm(output[b], axis=0)
                mag_Z = input[b][0]
                mag_Z_p = prediction[b][0]

                ax[b][0].imshow(mag_F, origin='lower', vmax=4, cmap='inferno')

                ax[b][1].imshow( mag_Z, origin='lower', vmax=3, cmap='gray')

                ax[b][2].imshow( mag_Z_p, origin='lower', vmax=3, cmap='gray')

    
        for a in ax.flat: a.axis('off')

        ax[0][0].text(s='Input', **utils_plot.texttop, transform=ax[0][0].transAxes)
        ax[0][1].text(s='Target', **utils_plot.texttop, transform=ax[0][1].transAxes)
        ax[0][2].text(s='Prediction', **utils_plot.texttop, transform=ax[0][2].transAxes)

        fig.subplots_adjust(wspace=0.01, hspace=0.01)

        logger.add_figure("vectorfields/model_%u"%self.index, fig, close=True, global_step=epoch)
        return

    def log_images(self, epoch=0):
    # Log images
        if epoch%self.logger_params['image_epoch_freq']==0:
            if 'vectorfield' in self.logger_params['image_callbacks']:
                self.draw_vectorfields_figure(epoch, 
                                            self.first_validation_sample['zyxin'],
                                            self.first_validation_sample['output'],
                                            self.first_validation_sample['prediction'],
                                            self.logger,
                                            )
                

        return
            
 



