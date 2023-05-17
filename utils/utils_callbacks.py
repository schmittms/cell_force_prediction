import os
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from datetime import datetime
import scipy.ndimage
from skimage.morphology import disk
from skimage.measure import label, regionprops, regionprops_table

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import skimage

def peak_sum(regionmask, intensity_image):
        return np.sum(intensity_image)
    
##############################################
#                                            #
#------- SECOND GENERATION CALLBACKS --------#
#                                            #
##############################################

class Gradients(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, force_thresh=0.):
        self.cb_names = ['grad_', 'sumFp', 
                         'MSE', 'MSEthreshold', 
                         'fbalance', 'fbalance_p', 'fbalance_thresh', 'fbalance_p_thresh']
        self.angmag = angmag
        self.thresh = force_thresh
        self.name = 'gradients'
        
    def __call__(self, inputs, outputs, prediction, cellmask, save_to=None):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 4 and len(outputs.shape)==4
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(outputs.shape)))
            
        if self.angmag:
            pred = prediction[0, 0, :, :]
        else:
            pred = torch.linalg.norm(prediction[0, :, :, :], dim=0)


        region_all = torch.ones_like(pred)
        region_highforce = pred>1.
        
        
        #inputs.requires_grad = True
        #pred.requires_grad = True
        
        G_all = []
        for region in [region_all, region_highforce]:
            region=torch.tensor(region, device=inputs.device, dtype=torch.float)
            G = torch.autograd.grad(pred, inputs, grad_outputs=region, retain_graph=True)

            G = np.squeeze(G[0].detach().cpu().numpy())
            G = np.abs(G)
            G_all.append(G)

        G_all = np.asarray(G_all)
        np.save(save_to, G_all)
    
            
            
        return
 
class GradientStats(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, force_thresh=0.):
        self.cb_names = ['grad_sum', 'input_sum', 
                         'prod_sum', 'quot_sum']
        self.angmag = angmag
        self.name = 'gradients'
        self.names_adjusted = False
        
    def __call__(self, inputs, cellmask, grad_loc=None, save_to=None):
        # modelidx could be done more elegantly with __getitem__ probably
        if self.names_adjusted==False:
            self.cb_names = [n+'_'+str(b) for n in self.cb_names for b in range(2*inputs.shape[1]) if ('input' not in n) or b<inputs.shape[1]]
            print(self.cb_names)
            self.names_adjusted=True

        with torch.no_grad():
            grad = np.abs(np.load(grad_loc))
            try:
                assert inputs.shape[-2:] == grad.shape[-2:] and len(inputs.shape)==4
            except:
                raise Exception("Shapes not correct, inputs: %s, grad %s"%(str(inputs.shape), str(grad.shape)))
           
            #inputs = np.tile(inputs, (2,1,1)) # double it so it's same size as grad
            prod = grad*np.abs(inputs)
            quot = grad/np.abs(inputs)
            
            quot[np.isnan(quot)] = 0
            quot[quot>np.max(grad)/(np.mean(np.abs(inputs))*0.0001)] = 0

            gradsum = np.sum(grad, axis=(-2, -1)).ravel()
            inputsum = np.sum(np.abs(inputs), axis=(-2, -1)).ravel()
            prodsum = np.sum(prod, axis=(-2, -1)).ravel()
            quotsum = np.nansum(quot, axis=(-2, -1)).ravel()

       # print(len(gradsum), len(inputsum), len(prodsum), len(quotsum))
        return_list = [*gradsum, *inputsum, *prodsum, *quotsum] 
         
        assert len(return_list) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_list), len(self.cb_names)) 
        return return_list, self.cb_names

   
        
        
        

class Forces(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, downsample=None, force_thresh=0.4):
        self.cb_names = ['sumF', 'sumFp', 
                         'MSE', 'MSEthreshold', 
                         'fbalance', 'fbalance_p', 'fbalance_thresh', 'fbalance_p_thresh']
        self.angmag = angmag
        self.name = 'forces'

        self.thresh = force_thresh

        self.downsample=downsample

    def __call__(self, prediction, target, cellmask):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            
        with torch.no_grad():
            if self.angmag:
                fxT, fyT = target[0]*np.cos(target[1]), target[0]*np.sin(target[1])
                fxP, fyP = prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])
            else:
                fxT, fyT = target[0], target[1]
                fxP, fyP = prediction[0], prediction[1]
                
            fxT[cellmask==0] = 0
            fyT[cellmask==0] = 0
            fxP[cellmask==0] = 0
            fyP[cellmask==0] = 0

            if self.downsample is not None:
                fxT = skimage.measure.block_reduce(fxT, block_size=(self.downsample,self.downsample), func=np.mean)
                fyT = skimage.measure.block_reduce(fyT, block_size=(self.downsample,self.downsample), func=np.mean)
                fxP = skimage.measure.block_reduce(fxP, block_size=(self.downsample,self.downsample), func=np.mean)
                fyP = skimage.measure.block_reduce(fyP, block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = skimage.measure.block_reduce(cellmask*1., block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = cellmask.astype(int)

            F = np.sqrt(fxT**2 + fyT**2)
            Fp = np.sqrt(fxP**2 + fyP**2)
            
            mse = (fxT-fxP)**2 + (fyT-fyP)**2
            msethresh = np.sum(mse[F>self.thresh])
            
            fbalance = np.sqrt( np.sum(fxT)**2+np.sum(fyT)**2)
            fbalance_p = np.sqrt( np.sum(fxP)**2+np.sum(fyP)**2) 

            fbalance_thresh = np.sqrt( np.sum(fxT[F>self.thresh])**2+np.sum(fyT[F>self.thresh])**2)
            fbalance_p_thresh = np.sqrt( np.sum(fxP[Fp>self.thresh])**2+np.sum(fyP[Fp>self.thresh])**2) 

            sumF = np.sum(F[F>self.thresh])
            sumFp = np.sum(Fp[Fp>self.thresh])

        return_list = [sumF, sumFp, mse.sum(), msethresh, fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh]
        
        assert len(return_list) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_list), len(self.cb_names)) 
        return return_list, self.cb_names




class ForcesMagOnly(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, downsample=None, force_thresh=0.4):
        self.cb_names = ['sumF', 'sumFp', 
                         'MSE']
        self.angmag = angmag
        self.name = 'forcesmag'

        self.thresh=force_thresh
        self.downsample=downsample

    def __call__(self, prediction, target, cellmask):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            
        with torch.no_grad():
            F = np.linalg.norm(target, axis=0)
            Fp = prediction[0]
            
            mse = F - Fp
            
            sumF = np.sum(F[F>self.thresh])
            sumFp = np.sum(Fp[Fp>self.thresh])

        return_list = [sumF, sumFp, mse.sum()]
        
        assert len(return_list) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_list), len(self.cb_names)) 
        return return_list, self.cb_names


class BoundaryStats(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, boundary_vals = [20,50,100], downsample=None, force_thresh=0.4):
        self.cb_names = ['boundary_sumF', 'boundary_sumFp', 
                         'boundary_sumFthresh', 'boundary_sumFpthresh',
                         'boundary_area',
                         'boundary_MSE', 'boundary_MSEthresh', 'boundary_dA']
        self.cb_names = [n+'_'+str(b) for n in self.cb_names for b in boundary_vals]
        self.angmag = angmag
        self.boundary_vals = boundary_vals
        self.name = 'boundarystats'
        self.thresh=force_thresh

        self.downsample = downsample

    def __call__(self, prediction, target, cellmask):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            
        with torch.no_grad():
            if self.angmag:
                fxT, fyT = target[0]*np.cos(target[1]), target[0]*np.sin(target[1])
                fxP, fyP = prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])
            else:
                fxT, fyT = target[0], target[1]
                fxP, fyP = prediction[0], prediction[1]
                
            fxT[cellmask==0] = 0
            fyT[cellmask==0] = 0
            fxP[cellmask==0] = 0
            fyP[cellmask==0] = 0
            
            if self.downsample is not None:
                fxT = skimage.measure.block_reduce(fxT, block_size=(self.downsample,self.downsample), func=np.mean)
                fyT = skimage.measure.block_reduce(fyT, block_size=(self.downsample,self.downsample), func=np.mean)
                fxP = skimage.measure.block_reduce(fxP, block_size=(self.downsample,self.downsample), func=np.mean)
                fyP = skimage.measure.block_reduce(fyP, block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = skimage.measure.block_reduce(cellmask*1., block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = cellmask.astype(int)

            mse = (fxT-fxP)**2 + (fyT-fyP)**2

            F = np.sqrt(fxT**2 + fyT**2)
            Fp = np.sqrt(fxP**2 + fyP**2)
            
            Fmask = F>self.thresh
            Fpmask = Fp>self.thresh
            
            ang = np.arctan2(fyT, fxT)
            angp = np.arctan2(fyP, fxP)
            dang = np.abs(np.remainder(ang - angp + np.pi, 2*np.pi) - np.pi) #L1
            
            dang[F<self.thresh] = 0
            
            grad = scipy.ndimage.morphology.distance_transform_edt((cellmask!=0)*1.)
            
            boundary_sumF = np.zeros(3)
            boundary_sumFp = np.zeros(3)
            boundary_sumFthresh = np.zeros(3)
            boundary_sumFpthresh = np.zeros(3)
            boundary_area = np.zeros(3)
            boundary_mse = np.zeros(3)
            boundary_msethresh = np.zeros(3)
            boundary_dA = np.zeros(3)
            
            for i, bound in enumerate(self.boundary_vals):
                boundary = (grad<bound)*(grad>0)
                boundary_area[i] = np.sum(boundary!=0)
                boundary_sumF[i] = np.sum( F[boundary])
                boundary_sumFp[i] = np.sum( Fp[boundary])
                boundary_sumFthresh[i] = np.sum( F[boundary*Fmask])
                boundary_sumFpthresh[i] = np.sum( Fp[boundary*Fpmask])
                boundary_mse[i] = np.mean( mse[boundary])
                boundary_msethresh[i] = np.mean( mse[boundary*Fmask])
                boundary_dA[i] = np.sum( dang[boundary])

        return_list = [*boundary_sumF, *boundary_sumFp, 
                       *boundary_sumFthresh, *boundary_sumFpthresh,
                       *boundary_area,
                       *boundary_mse, *boundary_msethresh, *boundary_dA]
        
        assert len(return_list) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_list), len(self.cb_names)) 

        return return_list, self.cb_names


class PeakStats(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, angmag, peak_thresholds = [0.5, 1, 2], downsample=None, force_thresh=0.4):
        self.cb_names = ['peak_F_mean', 'peak_F_sum', 'peak_Fp_mean', 'peak_Fp_sum',
                        'peak_MSE_mean', 'peak_MSE_max', 'peak_DF_mean', 'peak_dA_mean',
                        'peak_DF_sum', 'peak_dA_sum', 'peak_dA_avg',
                        'peak_F_max', 'peak_Fp_max', 'peak_F_hit', 'peak_Fp_miss',
                        'peak_area_mean', 'peak_total_area'
                        ]

        self.cb_names = [n+'_'+str(p) for n in self.cb_names for p in peak_thresholds]
        self.angmag = angmag
        self.peak_thresholds = peak_thresholds
        self.name = 'peakstats'

        self.thresh=force_thresh

        self.downsample=downsample

                
    def __call__(self, prediction, target, cellmask):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
            assert np.all(target[0]<100) and np.all(prediction[0]<100)
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            raise Exception('or: \n Forces not normalized, max target %0.1f, max pred %0.1f'%(np.max(np.abs(target)), np.max(np.abs(prediction))))

            
        with torch.no_grad():
 
            if self.angmag:
                fxT, fyT = target[0]*np.cos(target[1]), target[0]*np.sin(target[1])
                fxP, fyP = prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])
            else:
                fxT, fyT = target[0], target[1]
                fxP, fyP = prediction[0], prediction[1]
               
            fxT[cellmask==0] = 0
            fyT[cellmask==0] = 0
            fxP[cellmask==0] = 0
            fyP[cellmask==0] = 0

            
            if self.downsample is not None:
                fxT = skimage.measure.block_reduce(fxT, block_size=(self.downsample,self.downsample), func=np.mean)
                fyT = skimage.measure.block_reduce(fyT, block_size=(self.downsample,self.downsample), func=np.mean)
                fxP = skimage.measure.block_reduce(fxP, block_size=(self.downsample,self.downsample), func=np.mean)
                fyP = skimage.measure.block_reduce(fyP, block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = skimage.measure.block_reduce(cellmask*1., block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = cellmask.astype(int)
            
            mse = (fxT-fxP)**2 + (fyT-fyP)**2

            F = np.sqrt(fxT**2 + fyT**2)
            Fp = np.sqrt(fxP**2 + fyP**2)

            ang = np.arctan2(fyT, fxT)
            angp = np.arctan2(fyP, fxP)
            dang = np.abs(np.remainder(ang - angp + np.pi, 2*np.pi) - np.pi) #L1
            
            dang[F<self.thresh] = 0
            
            peak_F_mean = np.zeros(3)
            peak_Fp_mean = np.zeros(3)
            peak_F_sum = np.zeros(3)
            peak_Fp_sum = np.zeros(3)

            peak_MSE_mean = np.zeros(3)
            peak_MSE_max = np.zeros(3)
            peak_DF_mean = np.zeros(3)
            peak_DA_mean = np.zeros(3)
            peak_DF_sum = np.zeros(3)
            peak_DA_sum = np.zeros(3)
            peak_DA_avg = np.zeros(3)

            peak_F_max = np.zeros(3)
            peak_Fp_max = np.zeros(3)
            peak_F_hit = np.zeros(3)
            peak_Fp_miss = np.zeros(3)

            peak_area_mean = np.zeros(3)
            peak_total_area = np.zeros(3)
            
            for i,thresh in enumerate(self.peak_thresholds):
                L = label(F>thresh)
                Lp = label(Fp>thresh)
                peakarea =  np.asarray([x.area for x in regionprops(L)])
                peakFparea =  np.asarray([x.area for x in regionprops(Lp)])

                #peakecc =  np.asarray([x.eccentricity for x in regionprops(L)])
                peakF = np.asarray([x.mean_intensity for x in regionprops(L, F)])
                peakFp = np.asarray([x.mean_intensity for x in regionprops(L, Fp)])
                peakFsum = np.asarray([x.peak_sum for x in regionprops(L, F, extra_properties=(peak_sum,))])
                peakFpsum = np.asarray([x.peak_sum for x in regionprops(L, Fp, extra_properties=(peak_sum,))])
                
                peak_Fx = np.asarray([x.mean_intensity for x in regionprops(L, fxT)])
                peak_Fy = np.asarray([x.mean_intensity for x in regionprops(L, fyT)])

                peak_FxP = np.asarray([x.mean_intensity for x in regionprops(L, fxP)])
                peak_FyP = np.asarray([x.mean_intensity for x in regionprops(L, fyP)])

                peak_avg_ang = np.arctan(peak_Fy, peak_Fx)
                peak_avg_angP = np.arctan(peak_FyP, peak_FxP)

                peak_Davg_ang = np.abs(np.remainder(peak_avg_ang - peak_avg_angP + np.pi, 2*np.pi) - np.pi) #L1

                peakDF_frame_sum = np.asarray([x.peak_sum for x in regionprops(L, Fp-F, extra_properties=(peak_sum,))])
                peakDA_frame_sum = np.asarray([x.peak_sum for x in regionprops(L, dang, extra_properties=(peak_sum,))])
                peakDF_frame_mean = np.asarray([x.mean_intensity for x in regionprops(L, Fp-F)])
                peakDA_frame_mean= np.asarray([x.mean_intensity for x in regionprops(L, dang)])
                peakMSE = np.asarray([x.mean_intensity for x in regionprops(L, mse)])

                peakFhit = np.asarray([x.peak_sum for x in regionprops(L, Lp!=0, extra_properties=(peak_sum,))])
                peakFpmiss = np.asarray([x.peak_sum for x in regionprops(Lp, L==0, extra_properties=(peak_sum,))])

                if len(np.unique(L))==1:
                    peak_F_mean[i] = np.nan
                    peak_Fp_mean[i] = np.nan
                    peak_F_sum[i] = np.nan
                    peak_Fp_sum[i] = np.nan

                    peak_MSE_mean[i] = np.nan
                    peak_MSE_max[i] = np.nan # Max of means
                    peak_DF_mean[i] = np.nan
                    peak_DA_mean[i] = np.nan
                    peak_DF_sum[i] = np.nan
                    peak_DA_sum[i] = np.nan
                    peak_DA_avg[i] = np.nan

                    peak_F_max[i] = np.nan# Max of means
                    peak_Fp_max[i] = np.nan # Max of means
                    peak_F_hit[i] = np.nan
                    peak_Fp_miss[i] = np.nan
                    
                    peak_area_mean[i] = np.nan
                    peak_total_area[i] = np.nan
                else:
                    peak_F_mean[i] = peakF.mean()
                    peak_Fp_mean[i] = peakFp.mean()
                    peak_F_sum[i] = peakFsum.sum()
                    peak_Fp_sum[i] = peakFpsum.sum()

                    peak_MSE_mean[i] = peakMSE.mean()
                    peak_MSE_max[i] = peakMSE.max() # Max of means
                    peak_DF_mean[i] = peakDF_frame_mean.mean()
                    peak_DA_mean[i] = peakDA_frame_mean.mean()
                    peak_DF_sum[i] = peakDF_frame_sum.sum() 
                    peak_DA_sum[i] = peakDA_frame_sum.sum()
                    peak_DA_avg[i] = peak_Davg_ang.mean()

                    peak_F_max[i] = peakF.max() # Max of means
                    peak_Fp_max[i] = peakFp.max() # Max of means
                    peak_F_hit[i] = np.sum(peakFhit)/np.sum(L!=0)
                    peak_Fp_miss[i] = np.sum(peakFpmiss)/np.sum(Lp!=0)

                    peak_area_mean[i] = peakarea.mean()
                    peak_total_area[i] = peakarea.sum()
                    
        return_list = [*peak_F_mean, *peak_F_sum, *peak_Fp_mean, *peak_Fp_sum,
                       *peak_MSE_mean, *peak_MSE_max, *peak_DF_mean, *peak_DA_mean,
                       *peak_DF_sum, *peak_DA_sum, *peak_DA_avg,
                       *peak_F_max, *peak_Fp_max, *peak_F_hit, *peak_Fp_miss,
                       *peak_area_mean, *peak_total_area]

        assert len(return_list) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_list), len(self.cb_names)) 
        return return_list, self.cb_names


class HistStats(object): 
    def __init__(self, angmag, boundary_vals=[20,50,100], peak_thresholds=[0.5,1.,2.], downsample=None, force_thresh=0.4):
        self.boundary_vals=boundary_vals
        self.peak_thresholds=peak_thresholds
        self.F_bins = np.linspace(0,8,201)
        self.ang_bins = np.linspace(-np.pi,np.pi,201)
        self.dang_bins = np.linspace(0,np.pi,201)
        self.mse_bins = np.linspace(0,8,201)
        self.dm_bins = np.linspace(-7,7,201)
        self.cb_names = ['histF', 'histA', 'histAthresh', 'histMSE', 'histFA',
                        'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
                        'histFboundary', 'histAboundary', 'histMSEboundary', 'histFAboundary',
                        'histFpeak', 'histApeak', 'histMSEpeak', 'histFApeak']

        self.cb_names = [[n+'_'+str(p) for p in peak_thresholds] if 'peak' in n else [n] for n in self.cb_names]
        self.cb_names = [n for x in self.cb_names for n in x]
        self.cb_names = [[n+'_'+str(b)  for b in boundary_vals] if 'boundary' in n else [n] for n in self.cb_names]
        self.cb_names = [n for x in self.cb_names for n in x]
        
        self.angmag = angmag
        self.thresh=force_thresh
        self.name = 'histstats'
        self.n_boundaries=len(self.boundary_vals)
        self.n_peaks=len(self.peak_thresholds)
        self.bins_all = [[self.F_bins, self.F_bins], [self.ang_bins, self.ang_bins], [self.ang_bins, self.ang_bins], [self.F_bins, self.mse_bins], [self.F_bins, self.dang_bins],
                    [self.F_bins, self.F_bins], [self.ang_bins, self.ang_bins], [self.F_bins, self.mse_bins], [self.F_bins, self.dang_bins],
                   *[[self.F_bins, self.F_bins]]*self.n_boundaries, *[[self.ang_bins, self.ang_bins]]*self.n_boundaries, *[[self.F_bins, self.mse_bins]]*self.n_boundaries, *[[self.F_bins, self.dang_bins]]*self.n_boundaries,
                    *[[self.F_bins, self.F_bins]]*self.n_peaks, *[[self.ang_bins, self.ang_bins]]*self.n_peaks, *[[self.F_bins, self.mse_bins]]*self.n_peaks, *[[self.F_bins, self.dang_bins]]*self.n_peaks]

        self.downsample=downsample

    

    def __call__(self, prediction, target, cellmask):
    
        #print("HISTSTATS: \nself.thresh, self.angmag, \t", self.thresh, self.angmag)    
        F_thresh=self.thresh
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
            #assert np.all(target[0]<100) and np.all(prediction[0]<100)
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            raise Exception('or: \n Forces not normalized, max target %0.1f, max pred %0.1f'%(np.max(np.abs(target)), np.max(np.abs(prediction))))
            
        with torch.no_grad():

            #print(self.angmag)
            #print(np.max(target[0]), np.min(target[0]), np.max(target[1]), np.min(target[1]))
            #print(np.max(prediction[0]), np.min(prediction[0]), np.max(prediction[1]), np.min(prediction[1]))

            if self.angmag:
                fxT, fyT = target[0]*np.cos(target[1]), target[0]*np.sin(target[1])
                fxP, fyP = prediction[0]*np.cos(prediction[1]), prediction[0]*np.sin(prediction[1])
            else:
                fxT, fyT = target[0], target[1]
                fxP, fyP = prediction[0], prediction[1]
                
            fxT[cellmask==0] = 0
            fyT[cellmask==0] = 0
            fxP[cellmask==0] = 0
            fyP[cellmask==0] = 0


            if self.downsample is not None:
                fxT = skimage.measure.block_reduce(fxT, block_size=(self.downsample,self.downsample), func=np.mean)
                fyT = skimage.measure.block_reduce(fyT, block_size=(self.downsample,self.downsample), func=np.mean)
                fxP = skimage.measure.block_reduce(fxP, block_size=(self.downsample,self.downsample), func=np.mean)
                fyP = skimage.measure.block_reduce(fyP, block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = skimage.measure.block_reduce(cellmask*1., block_size=(self.downsample,self.downsample), func=np.mean)
                cellmask = cellmask.astype(int)
            
            
            mse = (fxT-fxP)**2 + (fyT-fyP)**2

            F = np.sqrt(fxT**2 + fyT**2)
            Fp = np.sqrt(fxP**2 + fyP**2)

            ang = np.arctan2(fyT, fxT)
            angp = np.arctan2(fyP, fxP)
            dang = np.abs(np.remainder(ang - angp + np.pi, 2*np.pi) - np.pi) #L1
            
            dang[F<F_thresh] = 0

           # ang[F<0.4] = 0
           # angp[F<0.4]= 0
            
            grad = scipy.ndimage.morphology.distance_transform_edt((cellmask!=0)*1.)
            
            HFpeak = []
            Hangpeak = []
            Hmsepeak = []
            HFangpeak = []
            for i,thresh in enumerate(self.peak_thresholds):
                L = F>thresh
                HFpeak.append( np.histogram2d(F[L].ravel(), Fp[L].ravel(), bins=(self.F_bins, self.F_bins))[0] )
                Hangpeak.append( np.histogram2d(ang[L].ravel(), angp[L].ravel(), bins=(self.ang_bins, self.ang_bins))[0] )
                Hmsepeak.append( np.histogram2d(F[L].ravel(), mse[L].ravel(), bins=(self.F_bins, self.mse_bins))[0] )
                HFangpeak.append( np.histogram2d(F[L].ravel(), dang[L].ravel(), bins=(self.F_bins, self.dang_bins))[0] )
           

            HF, _, _ = np.histogram2d(F.ravel(), Fp.ravel(), bins=(self.F_bins, self.F_bins))
            Hang, _, _ = np.histogram2d(ang[F>F_thresh].ravel(), angp[F>F_thresh].ravel(), bins=(self.ang_bins, self.ang_bins))
            Hangthresh, _, _ = np.histogram2d(ang[(F>F_thresh)*(Fp>F_thresh)].ravel(), angp[(Fp>F_thresh)*(F>F_thresh)].ravel(), bins=(self.ang_bins, self.ang_bins))
            Hmse, _, _ = np.histogram2d(F.ravel(), mse.ravel(), bins=(self.F_bins, self.mse_bins))
            HFang, _, _ = np.histogram2d(F.ravel(), dang.ravel(), bins=(self.F_bins, self.dang_bins))
           
            #'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
            HFmask, _, _ = np.histogram2d(F[cellmask].ravel(), Fp[cellmask].ravel(), bins=(self.F_bins, self.F_bins))
            Hangmask, _, _ = np.histogram2d(ang[(F>F_thresh)*cellmask].ravel(), angp[(F>F_thresh)*cellmask].ravel(), bins=(self.ang_bins, self.ang_bins))
            Hmsemask, _, _ = np.histogram2d(F[cellmask].ravel(), mse[cellmask].ravel(), bins=(self.F_bins, self.mse_bins))
            HFangmask, _, _ = np.histogram2d(F[cellmask].ravel(), dang[cellmask].ravel(), bins=(self.F_bins, self.dang_bins))
           

            HFboundary = []
            Hangboundary = []
            Hmseboundary = []
            HFangboundary = []
            for i, bound in enumerate(self.boundary_vals):
                boundary = (grad<bound)*(grad>0)
            
                HFboundary.append( np.histogram2d(F[boundary].ravel(), Fp[boundary].ravel(), bins=(self.F_bins, self.F_bins))[0] )
                Hangboundary.append( np.histogram2d(ang[boundary*(F>F_thresh)].ravel(), angp[boundary*(F>F_thresh)].ravel(), bins=(self.ang_bins, self.ang_bins))[0] )
                Hmseboundary.append( np.histogram2d(F[boundary].ravel(), mse[boundary].ravel(), bins=(self.F_bins, self.mse_bins))[0] )
                HFangboundary.append( np.histogram2d(F[boundary].ravel(), dang[boundary].ravel(), bins=(self.F_bins, self.dang_bins))[0] )
                    
        hists_all = [HF, Hang, Hangthresh, Hmse, HFang, 
                    HFmask, Hangmask, Hmsemask, HFangmask,
                    *HFboundary, *Hangboundary, *Hmseboundary, *HFangboundary,
                    *HFpeak, *Hangpeak, *Hmsepeak, *HFangpeak]
        
        assert len(hists_all)==len(self.cb_names), "%u not equal to %u"%(len(hists_all), len(self.cb_names))
        assert len(hists_all)==len(self.bins_all), "%u not equal to %u"%(len(hists_all), len(self.bins_all))
        
        return [hists_all, self.bins_all], self.cb_names




class HistStatsMagOnly(object): 
    def __init__(self, angmag, downsample=None, force_thresh=0.4):
        self.F_bins = np.linspace(0,5,201)
        self.cb_names = ['histF']

        
        self.angmag = angmag
        self.name = 'histstatsmag'
        self.bins_all = [[self.F_bins, self.F_bins]]
    
        self.thresh=force_thresh



    def __call__(self, prediction, target, cellmask):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
            assert prediction.shape[0] == 1 and target.shape[0] == 2
            assert np.all(target[0]<100) and np.all(prediction[0]<100)
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            raise Exception('or: \n Forces not normalized, max target %0.1f, max pred %0.1f'%(np.max(np.abs(target)), np.max(np.abs(prediction))))
            
        with torch.no_grad():

            F = np.linalg.norm(target, axis=0)
            Fp = prediction[0]
           
            HF, _, _ = np.histogram2d(F.ravel(), Fp.ravel(), bins=(self.F_bins, self.F_bins))
                   
        hists_all = [HF]
        
        assert len(hists_all)==len(self.cb_names), "%u not equal to %u"%(len(hists_all), len(self.cb_names))
        assert len(hists_all)==len(self.bins_all), "%u not equal to %u"%(len(hists_all), len(self.bins_all))
        
        return [hists_all, self.bins_all], self.cb_names




##############################################
#                                            #
#------- FIRST GENERATION CALLBACKS ---------#
#                                            #
##############################################

class AngleMagErrorCallback(object):
    def __init__(self, num_models, num_epochs, angmag):
        self.history = {'train': np.zeros((num_models, num_epochs, 2)),
                'test':  np.zeros((num_models, num_epochs, 2))}
        self.mag_err = {'train': np.zeros(num_models),
                'test': np.zeros(num_models)}
        self.ang_err = {'train': np.zeros(num_models),
                'test': np.zeros(num_models)}
        self.batch_counter = {'train': 0, 'test': 0}
        self.epoch_counter = 0
        self.cb_names = ['mag_mse', 'ang_mse']
        self.angmag = angmag
    def sample_cb(self, model_idx, prediction, target, return_value=False):
        # modelidx could be done more elegantly with __getitem__ probably
        with torch.no_grad():
            if self.angmag:
                ang = target[..., -1, :, :]
                mag = target[..., -2, :, :]
                pred_ang = prediction[..., -1, :, :]
                pred_mag = torch.abs(prediction[..., -2, :, :])

            else:
                x = target[..., -1, :, :]
                y = target[..., -2, :, :]
                x_pred = prediction[..., -1, :, :]
                y_pred = prediction[..., -2, :, :]

                mag = x*x + y*y
                pred_mag = x_pred*x_pred + y_pred*y_pred

                ang = torch.atan2(y, x)
                pred_ang = torch.atan2(y_pred, x_pred)

            mag_err = torch.mean(torch.abs(mag-pred_mag))
            ang_err = torch.abs(torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)
            ang_err = torch.mean(ang_err)


        if return_value:
            return [mag_err, ang_err], self.cb_names
        else:   
            self.mag_err[self.mode][model_idx] += mag_err
            self.ang_err[self.mode][model_idx]  += ang_err
            self.batch_counter[self.mode]  += 1
            return 

    
    def epoch_cb(self):
        for mode in self.history.keys():
            for m in range(len(self.history[mode])):
                self.history[mode][m, self.epoch_counter, 0] = self.mag_err[mode][m]/self.batch_counter[mode]
                self.history[mode][m, self.epoch_counter, 1] = self.ang_err[mode][m]/self.batch_counter[mode]
            self.mag_err[mode] *= 0.
            self.ang_err[mode] *= 0 
        
            self.batch_counter[mode] = 0
        
        self.epoch_counter += 1
        return

    def set_mode(self, mode):
        self.mode = mode
        return

    def save(self, out_dir):
        for mode in self.history.keys():
            fname = 'angmag_error_' + mode  
            np.save(out_dir + fname, np.asarray(self.history[mode]))
        return

class ForceBalanceViolationCallback(object):
    def __init__(self, num_models, num_epochs, angmag):
        self.history = {'train': np.zeros((num_models, num_epochs)),
                'test': np.zeros((num_models, num_epochs))}

        self.imbalance = {'train': np.zeros(num_models),
                'test': np.zeros(num_models)}
        self.batch_counter = {'train': 0, 'test': 0}
        self.epoch_counter = 0
        self.cb_names = ['force_balance']
        self.angmag = angmag

    def sample_cb(self, model_idx, prediction, target, return_value=False):
        # modelidx could be done more elegantly with __getitem__ probably
        with torch.no_grad():
            if self.angmag:
                ang = target[..., -1, :, :]
                mag = target[..., -2, :, :]
                pred_ang = prediction[..., -1, :, :]
                pred_mag = torch.abs(prediction[..., -2, :, :])

                x = mag*torch.cos(ang) # shape: (B, H, W)
                y = mag*torch.sin(ang)

                x_pr = pred_mag*torch.cos(pred_ang)
                y_pr = pred_mag*torch.sin(pred_ang)

            else:
                x = target[..., -1, :, :]
                y = target[..., -2, :, :]
                x_pr = prediction[..., -1, :, :]
                y_pr = prediction[..., -2, :, :]

            imbalance = (torch.abs(torch.mean(x_pr-x)) + torch.abs(torch.mean(y_pr-y)))/torch.mean(torch.sqrt(x*x + y*y))
        
        if return_value:
            return [imbalance], self.cb_names
        else:   
            self.imbalance[self.mode][model_idx]  += imbalance  
            self.batch_counter[self.mode]  += 1
            return 

    
    def epoch_cb(self):
        for mode in self.history.keys():
            for m in range(len(self.history[mode])):
                self.history[mode][m, self.epoch_counter] = self.imbalance[mode][m]/self.batch_counter[mode]
            #print('hist rel err\t', self.history[mode][:, :3])     
            self.imbalance[mode] *= 0.  
            self.batch_counter[mode] = 0
        
        self.epoch_counter += 1
        return

    def set_mode(self, mode):
        self.mode = mode
        return

    def save(self, out_dir):
        for mode in self.history.keys():
            fname = 'force_balance_' + mode     
            np.save(out_dir + fname, np.asarray(self.history[mode]))
        return



class RelativeErrorCallback(object):
    def __init__(self, num_models, num_epochs, angmag):
        self.history = {'train': np.zeros((num_models, num_epochs)),
                'test': np.zeros((num_models, num_epochs))}

        self.rel_acc = {'train': np.zeros(num_models),
                'test': np.zeros(num_models)}
        self.batch_counter = {'train': 0, 'test': 0}
        self.epoch_counter = 0
        self.cb_names = ['rel_err']
        self.angmag = angmag

    def sample_cb(self, model_idx, prediction, target, return_value=False):
        # modelidx could be done more elegantly with __getitem__ probably
        with torch.no_grad():
            if self.angmag:
                ang = target[..., -1, :, :]
                mag = target[..., -2, :, :]
                pred_ang = prediction[..., -1, :, :]
                pred_mag = torch.abs(prediction[..., -2, :, :])

                x = mag*torch.cos(ang) # shape: (B, H, W)
                y = mag*torch.sin(ang)

                x_pr = pred_mag*torch.cos(pred_ang)
                y_pr = pred_mag*torch.sin(pred_ang)

            else:
                x = target[..., -1, :, :]
                y = target[..., -2, :, :]
                x_pr = prediction[..., -1, :, :]
                y_pr = prediction[..., -2, :, :]



        
            rel_acc = torch.sqrt((x-x_pr)**2 + (y-y_pr)**2)/(torch.sqrt(x*x + y*y)+torch.sqrt(x_pr**2 + y_pr**2))
            rel_acc = rel_acc[~(torch.isnan(rel_acc)+torch.isinf(torch.abs(rel_acc)))].mean()

            

        if return_value:
            return [rel_acc], self.cb_names
        else:   
            self.rel_acc[self.mode][model_idx]  += rel_acc  
            self.batch_counter[self.mode]  += 1
            return 

    
    def epoch_cb(self):
        for mode in self.history.keys():
            for m in range(len(self.history[mode])):
                self.history[mode][m, self.epoch_counter] = self.rel_acc[mode][m]/self.batch_counter[mode]
        #   print('hist rel err\t', self.history[mode][:, :3])      
            self.rel_acc[mode] *= 0.    
            self.batch_counter[mode] = 0
        
        self.epoch_counter += 1
        return

    def set_mode(self, mode):
        self.mode = mode
        return

    def save(self, out_dir):
        for mode in self.history.keys():
            fname = 'relative_error_' + mode    
            np.save(out_dir + fname, np.asarray(self.history[mode]))
        return


class CallbackPlot(object):
    def __init__(self,
            out_dir,
            model_names, 
            device,
            norm_im_out='all',
            vae=False):
        self.out_dir = out_dir
        self.model_names = model_names
        self.device = device
        self.norm_im_out = norm_im_out
        self.vae = vae
        
    def make_preds(self, sample, models):
        inputs, gt = sample

        predictions = []
        losses = []
        for model in models:# Just use model 0
            model.eval()
            pred = model(inputs.to(self.device))
            predictions.append(pred.cpu().detach())
#           L=criterion(pred.detach().cpu(), gt.detach().cpu())
#           print(L.shape)
#           losses.append(L)    
        return np.concatenate((inputs.cpu().detach(), gt.cpu().detach()), axis=1), predictions
    def make_preds_vae(self, sample, models):
        predictions = []
        losses = []
        for model in models:# Just use model 0
            model.eval()
            pred, _, _ = model(sample.to(self.device))
            predictions.append(pred.cpu().detach())
#           L=criterion(pred.detach().cpu(), gt.detach().cpu())
#           print(L.shape)
#           losses.append(L)    
        print('Callbacks shapes: pred', pred.shape, '\n inputs ', sample.shape)
        return [sample.cpu().detach()], [pred.cpu().detach()]

    def make_preds_multiinput(self, sample, models, coarse_grain=8):
        inputs, gt = sample
        CG = coarse_grain
        predictions = []
        losses = []
        for i in range(len(inputs)):
            for m in range(len(models)//len(inputs)):
                model_idx = m + i*len(models)//len(inputs)# Just use model 0
                print('predicting model %d\t %s'%(model_idx, self.model_names[model_idx]))
                models[model_idx].eval()
                pred = models[model_idx](inputs[i].to(self.device))
                predictions.append(pred.cpu())
            inputs[i] = inputs[i].cpu()

        for n,i in enumerate(inputs):
            i = skimage.measure.block_reduce(i.detach().numpy(), (1,1, CG, CG), np.mean)
            inputs[n] = i
        for n, p in enumerate(predictions):
            p = skimage.measure.block_reduce(p.detach().numpy(), (1,1, CG, CG), np.mean)
            predictions[n] = p
        gt = skimage.measure.block_reduce(gt.cpu().detach().numpy(), (1,1, CG, CG), np.mean)    
        zipped = [[np.concatenate([input, gt], axis=1),]*(len(models)//len(inputs)) for input in inputs]
        zipped = [t for z in zipped for t in z] 
        return zipped, predictions
    
    def make_fig(self, models, loader, epoch, multiinput=True):
        for model in models:
            model.eval()
        with torch.no_grad():
            for sample in loader:   
                if multiinput:
                    inout, preds = self.make_preds_multiinput(sample, models, coarse_grain=8)
                elif self.vae:   
                    inout, preds = self.make_preds_vae(sample, models)
                else:   
                    inout, preds = self.make_preds(sample, models)
                d = datetime.now()
                dstr = d.strftime("%y%m%d_%H%M")
                for n, (io, p) in enumerate(zip(inout, preds)):
                    full = np.concatenate((io, p), axis=1)  
                    ax_dim = full.shape[:2]
                    fig, ax = plt.subplots(*ax_dim, figsize=(ax_dim[1]*1.5, ax_dim[0]*1.5), constrained_layout=1)
                    in_len = ax_dim[1]-4
                    outlen = 2
                    

                    for b, a_row in enumerate(ax):
                        out1_vmax = np.amax(np.abs(full[b][in_len]))
                        out2_vmax = np.amax(np.abs(full[b][in_len+1]))
                        inmaxes = [0.4,] * in_len
                        inmins = [0,] * in_len
                        vmaxs = [*inmaxes, out1_vmax, out2_vmax, out1_vmax, out2_vmax]

                        vmins = [*inmins, -1*out1_vmax, -1*out2_vmax, -1*out1_vmax, -1*out2_vmax]
                        cmaps = ['gray',]*in_len + ['seismic',]*4
                        if np.all(full[b][in_len]>0): 
                            out1_vmax=0
                            cmaps[-2] = None
                            cmaps[-4] = None
                        if np.all(full[b][in_len+1]>0): 
                            out2_vmax=0
                            cmaps[-1] = None
                            cmaps[-3] = None

                        for ch, a in enumerate(a_row):
                            im = full[b][ch]
                            if ch<in_len: im /= np.max(im)
                            a.imshow(im, origin='lower', vmin=vmins[ch], vmax=vmaxs[ch], cmap=cmaps[ch])
                            if b==0:
                                if ch<in_len:
                                    a.text(0.5, 1.05, 'Input', 
                                        transform=a.transAxes, 
                                        ha='center', va='bottom')
                                elif ch>=in_len and ch<outlen+in_len:
                                    a.text(0.5, 1.05, 'Target\nOutput',
                                        transform=a.transAxes, 
                                        ha='center', va='bottom')
                                fig.suptitle(self.model_names[n])
                            a.axis('off')
                
            
                    modelN = self.model_names[n].split('.')[0]
                    print(modelN)
                    modelN = ''.join(modelN.split(','))
                    print(modelN)
                    save_to = os.path.join(self.out_dir, "pred_fig_%s_epoch%u.png" %(modelN, epoch))
                    fig.savefig(save_to, dpi=49)
                    if epoch%2:
                        np.save(os.path.join(self.out_dir, 'full_%s_%s.npy'%('train', modelN)), full)
                    else:
                        np.save(os.path.join(self.out_dir, 'full_%s_%s.npy'%('test', modelN)), full)
                    del full
                    fig.clear()
                    plt.close('all')
                break

                    

scalar_callback_dict = {
    'angmag_mse': AngleMagErrorCallback,
    'relative_err': RelativeErrorCallback,
    'forces': Forces, #ForceBalanceViolationCallback,
    'forcesmag': ForcesMagOnly, #ForceBalanceViolationCallback,
    'peakstats': PeakStats,
    'boundarystats': BoundaryStats,
    'histstats': HistStats,
    'histstatsmag': HistStatsMagOnly,
    'gradientstats': GradientStats,
    }


"""
Below: IMAGE CALLBACKS

Intended for prediction only 
These are callbacks where we want either to evaluate a callback on each pixel and save the result, or where we want to make some image of a callback (essentially same thing). All return image of same shape as target and predictions


"""
class AngMagErrImageCallback(object):
    def __init__(self, num_models, num_epochs, angmag):
        
        self.name = 'angmagerr_img'
        self.angmag = angmag
    def generate(self, target, prediction, save_to=None):
        # modelidx could be done more elegantly with __getitem__ probably
        with torch.no_grad():
            if self.angmag:
                ang = target[..., -1, :, :]
                mag = target[..., -2, :, :]
                pred_ang = prediction[..., -1, :, :]
                pred_mag = torch.abs(prediction[..., -2, :, :])

            else:
                x = target[..., -1, :, :]
                y = target[..., -2, :, :]
                x_pred = prediction[..., -1, :, :]
                y_pred = prediction[..., -2, :, :]

                mag = x*x + y*y
                pred_mag = x_pred*x_pred + y_pred*y_pred

                ang = torch.atan2(y, x)
                pred_ang = torch.atan2(y_pred, x_pred)

            mag_err = torch.abs(mag-pred_mag)
            ang_err = torch.abs(torch.remainder(ang - pred_ang + np.pi, 2*np.pi) - np.pi)
            
            print(mag_err.shape)
            self.item = torch.cat((pred_mag, mag, pred_ang, ang), axis=0)#ang_err, mag), axis=0)
            print(self.item.shape)

        if save_to is not None:
            assert(save_to.split('.')[-1]=='npy')
            np.save(save_to, self.item.cpu().numpy())   
            return
        else:   
            return 


class MSEImageCallback(object):
    def __init__(self, num_models, num_epochs, angmag):
        
        self.name = 'mse_img'
        self.angmag = angmag
    def generate(self, target, prediction, save_to=None):
        # modelidx could be done more elegantly with __getitem__ probably
        with torch.no_grad():
            if self.angmag:
                ang = target[..., -1, :, :]
                mag = target[..., -2, :, :]
                pred_ang = prediction[..., -1, :, :]
                pred_mag = torch.abs(prediction[..., -2, :, :])

                x = mag*torch.cos(ang)
                y = mag*torch.sin(ang)
            
                x_pr = pred_mag*torch.cos(pred_ang)
                y_pr = pred_mag*torch.sin(pred_ang)
                
            else:
                x = target[..., -1, :, :]
                y = target[..., -2, :, :]
                x_pr = prediction[..., -1, :, :]
                y_pr = prediction[..., -2, :, :]


            self.item = (x-x_pr)**2 + (y-y_pr)**2

        if save_to is not None:
            assert(save_to.split('.')[-1]=='npy')
            np.save(save_to, self.item.cpu().numpy())   
            return
        else:   
            return 


image_callback_dict = {
    'angmag_img': AngMagErrImageCallback,
    'mse_img':  MSEImageCallback,
    'gradients': Gradients

    }


"""
def callback_figs(out_dir, models, model_names, device, test_loader, epoch, norm_im_out='all', weighted_target='none'):
    norm_im_out = 'none'
    for sample in test_loader:
        d = datetime.now()
        dstr = d.strftime("%d%m%y_%H%M")
        input, gt = sample
        predictions = []
        losses = []
        for model in models:# Just use model 0
            model.eval()
            pred = model(input.to(device))
            predictions.append(pred.cpu())
#           L=criterion(pred.detach().cpu(), gt.detach().cpu())
#           print(L.shape)
#           losses.append(L)    
        full = torch.cat((input.cpu(), gt.cpu(), *predictions), axis=1)
        full = full.detach().numpy()
        
        ax_dim = full.shape[:2]
        subplot_shape = full.shape[:2]
        print('plot shape: ', subplot_shape)
        input_len = input.shape[1]
        output_len = gt.shape[1]
        print(input_len, output_len)
        fig, ax = plt.subplots(*ax_dim, figsize=(ax_dim[1]*1.5, ax_dim[0]*1.5), constrained_layout=1)
        gt_max = np.asarray([torch.max(torch.abs(g)) for g in gt])
        for b, a_row in enumerate(ax):
            for ch, a in enumerate(a_row):
                if ch<input_len:
                    vmax = None
                    cmap='viridis'
                else:
                    if norm_im_out=='all':
                        vmax = np.max(gt_max)
                    elif norm_im_out=='none':
                        vmax=None
                    else:   
                        vmax = gt_max[b]
                    
                if vmax is not None:
                    vmin = -1*vmax
                else:
                    vmin = None     
                a.imshow(full[b][ch], vmin=vmin, vmax=vmax, cmap=cmap)
                if b==0:
                    if ch<input_len:
                        a.text(0.5, 1.05, 'Input', 
                            transform=a.transAxes, 
                            ha='center', va='bottom')
                    elif ch>=input_len and ch<output_len+input_len:
                        a.text(0.5, 1.05, 'Target\nOutput',
                            transform=a.transAxes, 
                            ha='center', va='bottom')
                    elif (ch-(output_len+input_len))%output_len == 0:
                        txt = model_names[(ch-(input_len+output_len))//output_len]
                        a.text(0.5, 1.05, txt[:int(len(txt)/2)]+'\n'+txt[int(len(txt)/2):],
                            transform=a.transAxes, 
                            ha='center', va='bottom', fontsize=4.5)
                a.axis('off')
        try:
            plt.savefig(out_dir+"/pred_fig_%s_epoch%u.png" %(dstr, epoch), bbox_inches='tight')
            np.save(out_dir+'/full.npy', full)
        except:
            print('Failed to save to ' + out_dir + '/pred_fig_...')
            plt.savefig("./pred_fig_%s_%u_epoch.png" %(dstr, epoch), bbox_inches='tight')
        del full
        fig.clear()
        plt.close('all')
        if weighted_target!='none':
            full = torch.cat((input.cpu(), gt.cpu(), *losses), axis=1)
            full = full.detach().numpy()
            fig, ax = plt.subplots(*ax_dim, figsize=(ax_dim[1]*1.5, ax_dim[0]*1.5), constrained_layout=1)
            for b, a_row in enumerate(ax):
                for ch, a in enumerate(a_row):
                    if ch<input_len:
                        vmax = 1
                        cmap='viridis'
                    elif ch>=input_len and ch<output_len+input_len:
                        vmax=None
                        cmap='viridis'
                    else:
                        vmax=None
                        cmap='bone' 
                        
                    a.imshow(full[b][ch], vmin=0, vmax=vmax, cmap=cmap)
                    if b==0:
                        if ch<input_len:
                            a.text(0.5, 1.05, 'Input', 
                                transform=a.transAxes, 
                                ha='center', va='bottom')
                        elif ch>=input_len and ch<output_len+input_len:
                            a.text(0.5, 1.05, 'Target\nOutput',
                                transform=a.transAxes, 
                                ha='center', va='bottom')
                        else:
                            txt = torch.mean(losses[ch-2][b])
                        a.text(0.05, .95, 'L = '+str(txt),
                            transform=a.transAxes, 
                            ha='left', va='top', fontsize=8, color='w', bbox=dict(facecolor='k', alpha=0.4))
                    a.axis('off')
            try:
                plt.savefig(out_dir+"/loss_fig_%s_epoch%u.png" %(dstr, epoch), bbox_inches='tight')
            except:
                print('Failed to save to ' + out_dir + '/pred_fig_...')
                plt.savefig("./pred_fig_%s_%u_epoch.png" %(dstr, epoch), bbox_inches='tight')
            del full
            fig.clear()
            plt.close('all')
        break

"""






