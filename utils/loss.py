import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class XY_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        print("LOSS KWARGS", kwargs)
        self.max_force= kwargs['loss_kwargs'].get('max_force')
        super().__init__()
        
    def forward(self, prediction, target, expweight=0.):
        mag = torch.linalg.norm(target, dim=1, keepdim=True) 
        
        MSE =F.mse_loss(prediction, target, reduction='none')

        loss_weight = torch.exp(torch.minimum(torch.abs(mag),self.max_force*torch.ones_like(mag))*expweight)
    
    
        return {'mse_loss': MSE.mean(), 'base_loss': torch.mean(MSE*loss_weight) }


class r_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, expweight=0.):
        MSE =F.mse_loss(prediction, target)
        return {'mse_loss': MSE, 'base_loss': MSE }

     


def angle_error(ang1, ang2):
    if torch.is_tensor(ang1):
        return (torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)**2 
    else:
        return np.abs(np.remainder(ang1 - ang2 + np.pi, 2*np.pi) - np.pi)    



def angle_loss(ang, pred_ang):
    return (torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)**2 


class AM_loss_dict(nn.MSELoss): # Angle magnitude loss
    def __init__(self, **kwargs):
        print("LOSS KWARGS", kwargs)
        super().__init__()
        self.max_force= kwargs['loss_kwargs'].get('max_force')

    def forward(self, prediction, target, expweight=0., batch_avg=True):
        """
        MAG ERR, ANG ERR  HAVE SHAPES [B, H, W]
        """
        
        ang = target[..., -1, :, :]
        mag = target[..., -2, :, :]
        pred_ang = prediction[..., -1, :, :]
        pred_mag = prediction[..., -2, :, :]

        nonzero = (mag>0)

        mag_err =F.mse_loss(pred_mag, mag, reduction='none')
        ang_err =(torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)**2 #L1
        
        
        loss_weight = torch.exp(torch.minimum(torch.abs(mag),self.max_force*torch.ones_like(mag))*expweight)

        if batch_avg:
            mag_err_weighted = torch.mean(mag_err*loss_weight)
            ang_err_weighted = torch.mean((ang_err*loss_weight)[nonzero])
        else:
            ang_err[~nonzero] = torch.nan
            mag_err_weighted = torch.mean(mag_err*loss_weight, axis=(-1,-2))
            ang_err_weighted = torch.nanmean((ang_err*loss_weight), axis=(-1,-2))

        x = mag*torch.cos(ang)
        y = mag*torch.sin(ang)

        x_pr = pred_mag*torch.cos(pred_ang)
        y_pr = pred_mag*torch.sin(pred_ang)

        mse_loss = (x-x_pr)**2 + (y-y_pr)**2

        return {'base_loss': mag_err_weighted*10 + ang_err_weighted, 'mse_loss': mse_loss.mean().detach(), 'mag_loss': mag_err.mean().detach(), 'ang_loss': ang_err[nonzero].mean().detach(), 'mag_loss_weighted': mag_err_weighted.detach(), 'ang_loss_weighted': ang_err_weighted.detach()}
    
    
    def all_metrics(self, prediction, target, mask, expweight=0., batch_avg=True):
        """
        MAG ERR, ANG ERR  HAVE SHAPES [B, H, W]
        """
        
        ang = target[..., -1, :, :]
        mag = target[..., -2, :, :]
        pred_ang = prediction[..., -1, :, :]
        pred_mag = prediction[..., -2, :, :]

        nonzero = (mag>0)

        mag_err =F.mse_loss(pred_mag, mag, reduction='none')
        ang_err =(torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)**2 #L1
        
        mag_err_l1 = torch.abs(mag - pred_mag)
        #print(mag_err.shape, mask.shape)
        
        
        if mask.shape[0]==1:
            # Just squeeze one dim
            mag_err[mask[0]==0] = torch.nan
            ang_err[mask[0]==0] = torch.nan
        else:
            mag_err[mask.squeeze()==0] = torch.nan
            ang_err[mask.squeeze()==0] = torch.nan
        ang_err[~nonzero] = torch.nan

        loss_weight = (mag-0.5)# *expweight

        mag_err_weighted = torch.nanmean(mag_err*loss_weight, axis=(-1,-2))
        ang_err_weighted = torch.nanmean((ang_err*loss_weight), axis=(-1,-2))

        x = mag*torch.cos(ang)
        y = mag*torch.sin(ang)

        x_pr = pred_mag*torch.cos(pred_ang)
        y_pr = pred_mag*torch.sin(pred_ang)
        
        if mask.shape[0]==1:
            # Just squeeze one dim
            x[mask[0]==0] = torch.nan
            y[mask[0]==0] = torch.nan
            x_pr[mask[0]==0] = torch.nan
            y_pr[mask[0]==0] = torch.nan
            
            ang[mask[0]==0] = torch.nan
            mag[mask[0]==0] =torch.nan
            pred_ang[mask[0]==0] = torch.nan
            pred_mag[mask[0]==0] = torch.nan
        else:
            x[mask.squeeze()==0] = torch.nan
            y[mask.squeeze()==0] = torch.nan
            x_pr[mask.squeeze()==0] = torch.nan
            y_pr[mask.squeeze()==0] = torch.nan
            
            ang[mask.squeeze()==0] = torch.nan
            mag[mask.squeeze()==0] =torch.nan
            pred_ang[mask.squeeze()==0] = torch.nan
            pred_mag[mask.squeeze()==0] = torch.nan

        mse_loss = (x-x_pr)**2 + (y-y_pr)**2
        mse_mag_loss = torch.sqrt(mse_loss)
        
        mse_weighted = torch.nanmean(mse_loss*loss_weight, axis=(-1,-2))
        mse_mag_weighted = torch.nanmean(mse_mag_loss*loss_weight, axis=(-1,-2))
        
        #print(mse_weighted.shape, mse_loss.shape, loss_weight.shape, x.shape, mag.shape)
        
        
        #rmse_loss = torch.sqrt(mse_loss)
        #print('mag', mag.shape)
        #print('mag_err', mag_err.shape)
        #print('pred_mag', pred_mag.shape)

        return {'base_loss': mag_err_weighted*10 + ang_err_weighted, 
                    'mse_loss': torch.nanmean(mse_loss, axis=(-1,-2)).detach(), # <(vec F  - vec F')^2>
                    'mse_weighted': mse_weighted.detach(),                      # <(vec F  - vec F')^2>
                    'mse_mag_weighted': mse_mag_weighted.detach(), 
                    'mse_mag_loss': torch.nanmean(mse_mag_loss, axis=(-1,-2)).detach(), 
                    'mag_loss': torch.nanmean(mag_err, axis=(-1,-2)).detach(), # <(F  - F')^2>
                    'mag2_loss': torch.nanmean(torch.sqrt(mag_err), axis=(-1,-2)).detach(), # < |F  - F'| >
                    'mag_sum_loss': torch.nansum(mag_err, axis=(-1,-2)).detach(), # # < |F  - F'| >
                    'mag2_sum_loss': torch.nansum(torch.sqrt(mag_err), axis=(-1,-2)).detach(), 
                    'rel_mag_loss': torch.nanmean(mag_err/mag**2, axis=(-1,-2)).detach(), 
                    'rel2_mag_loss': torch.nanmean(mag_err/(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(), 
                    'rel3_mag_loss': torch.nanmean(torch.sqrt(mag_err)/torch.sqrt(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(),
                    'rel4_mag_loss': torch.nanmean(torch.sqrt(mag_err)/(0.5*(mag+pred_mag)), axis=(-1,-2)).detach(), 
                    'rel2_mse_loss': torch.nanmean(mse_loss/(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(), 
                    'rel3_mse_loss': torch.nanmean(torch.sqrt(mse_loss)/torch.sqrt(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(),
                    'rel4_mse_loss': torch.nanmean(torch.sqrt(mse_loss)/(0.5*(mag+pred_mag)), axis=(-1,-2)).detach(), 
                    'sum_F': torch.nansum(mag, axis=(-1,-2)).detach(), 
                    'sum_Fp': torch.nansum(pred_mag, axis=(-1,-2)).detach(), 
                    #'sum_F_0.5': torch.nansum(mag[mag>0.5]).detach(), 
                    #'sum_Fp_0.5': torch.nansum(pred_mag[mag>0.5], axis=(-1,-2)).detach(), 
                    #'sum_Fp_0.5p': torch.nansum(pred_mag[pred_mag>0.5], axis=(-1,-2)).detach(), 
                    #'sum_F_1.0': torch.nansum(mag[mag>1.], axis=(-1,-2)).detach(), 
                    #'sum_Fp_1.0': torch.nansum(pred_mag[mag>1.], axis=(-1,-2)).detach(), 
                    #'sum_Fp_1.0p': torch.nansum(pred_mag[pred_mag>1.], axis=(-1,-2)).detach(), 
                    'mean_F': torch.nanmean(mag, axis=(-1,-2)).detach(), 
                    'mean_Fp': torch.nanmean(pred_mag, axis=(-1,-2)).detach(), 
                    'mean_F_Fp': torch.nanmean(torch.sqrt(0.5*(mag**2 + pred_mag**2)), axis=(-1,-2)).detach(), 
                    'ang_loss': torch.nanmean(ang_err, axis=(-1,-2)).detach(), 
                    'mag_loss_weighted': mag_err_weighted.detach(), 
                    'ang_loss_weighted': ang_err_weighted.detach()}

            
        

loss_function_dict = {
                        'xy': XY_loss_dict,
                        'am': AM_loss_dict,
                        'r_mse': r_MSE_loss_dict, 
                        }
