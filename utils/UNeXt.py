import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr
from time import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_loss import loss_function_dict 
import utils.utils_plot as utils_plot
from utils.UNeXt_base import UNet as UNet_base

from torch.utils.tensorboard import SummaryWriter


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape((x.shape[0], -1))

#class VanillaCNN(nn.Module): # This will have

       
class UNet(UNet_base):
                #input_type,
                #prepend_hparams, # Prepended layers, keys=channel_list, stride, kernel, dilation, dropout_rate, activation_function, batchnorm, split
                #encoder_hparams, #encoder hparams keys=n_ch, n_layers
                #decoder_hparams, #decoder keys=n_layers
                #append_hparams, # appended layers keys=(same as prepended)+"split"
                #optimizer_hparams, # Learning rate, scheduling, etc.
                #loss_hparams, # Regularization terms etc.
                #logger_params,
                #name,
                #model_idx):

    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.angmag = True if 'am' in kwargs['loss_hparams']['loss_type'] else False

    @classmethod
    def str_to_dict(cls, arg_string, arg_type):
        """
        Arguments are passed as strings, we want to convert these into dicts of hyperparameters
        """
        if arg_type=='loss':
            keys = ['loss_type', 'exp_weight', 'exp_schedule', 'strainenergy_regularization', 'reg_schedule', 'loss_kwargs']
            paramtypes = [str, float, str, float, str, str]
            
            rawdict = {s.split(',')[0]: p( ','.join(s.split(',')[1:]) ) for s,p in zip(arg_string.split(':'), paramtypes)}
            
            #schedule_keys = ['type', 'width', 'e_crit']
            ptypes = [str, float, float]
            #loss kwargs_keys = ['max_force']
            ptypes_losskwargs=[float]
            rawdict['exp_schedule'] = {s.split('>')[0]: p( s.split('>')[1] ) for s,p in zip(rawdict['exp_schedule'].split(','), ptypes)}
            rawdict['reg_schedule'] = {s.split('>')[0]: p( s.split('>')[1] ) for s,p in zip(rawdict['reg_schedule'].split(','), ptypes)}
            rawdict['loss_kwargs'] = {s.split('>')[0]: p( s.split('>')[1] ) for s,p in zip(rawdict['loss_kwargs'].split(','), ptypes_losskwargs)}
            return rawdict        
        else:
            return super().str_to_dict(arg_string, arg_type)

    def reg_scheduler(self, schedule, x, e):
        """
        Modifies value of x according to schedule, and current epoch e
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
        loss_dict = self.loss_function(prediction, batch['output'], expweight=expweight) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

        strainenergy_loss = self.strainenergy_loss(prediction, batch['output'], batch['displacements'], batch['mask'].bool())
        loss = loss_dict['base_loss'] \
                + self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), self.loss_hparams.get('strainenergy_regularization'), epoch)*strainenergy_loss

        if torch.isnan(loss):
            print("LOSS IS NAN")
            print(loss_dict, {'strainenergy_loss': strainenergy_loss})


    # Backprop
        loss.backward()
        self.optimizer.step()

        for x in loss_dict:
            loss_dict[x].detach()
        loss_dict = {**loss_dict, 'strainenergy_loss': strainenergy_loss.sqrt().detach(),
                        'exp_schedule': expweight,
                        'reg_schedule': self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), 1., epoch)}

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

            expweight = self.loss_hparams.get('exp_weight')*self.reg_scheduler(  self.loss_hparams.get('exp_schedule'), 1., epoch)
            loss_dict = self.loss_function(prediction, batch['output'], expweight=expweight) # This is a dictionary of potential loss values. The one which should be gradded is called 'base_loss'

            strainenergy_loss = self.strainenergy_loss(prediction, batch['output'], batch['displacements'], batch['mask'].bool())
            loss = loss_dict['base_loss'] \
                + self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), self.loss_hparams.get('strainenergy_regularization'), epoch)*strainenergy_loss

            loss_dict = {**loss_dict, 'strainenergy_loss': strainenergy_loss.sqrt().detach(),
                            'exp_schedule': expweight,
                            'reg_schedule': self.reg_scheduler(  self.loss_hparams.get('reg_schedule'), 1., epoch)}
            
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

                #ax[b][0].imshow(input[b][0]/input[b][0].max(), origin='lower', **cscheme['input'](input, b))


                #ax[b][1].imshow( mag_T, origin='lower', vmax=4, cmap='inferno')
        
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


