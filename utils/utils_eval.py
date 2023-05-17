import os
import sys
import pickle
import pickle
#from natsort import natsorted
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
from datetime import datetime
import scipy.ndimage
from skimage.morphology import disk
from skimage.measure import label, regionprops, regionprops_table

from utils.utils_data_processing_base import SubsetSampler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import skimage
 
def predict(out_dir, device, models, dataset, scalar_callbacks, hist_callbacks, skip, include):
    d = datetime.now()
    dstr = d.strftime("_%y%m%d_%H%M")

    if not os.path.exists(out_dir):
        print('Making directory\t', out_dir)
        os.makedirs(out_dir)
    
    if not os.path.exists(os.path.join(out_dir, 'histograms')):
        print('Making directory\t', os.path.join(out_dir, 'histograms'))
        os.makedirs(os.path.join(out_dir, 'histograms'))
        
    scalar_cb_names = [n for cb in scalar_callbacks for n in cb.cb_names]
    hist_cb_names = [n for cb in hist_callbacks for n in cb.cb_names]

    scalar_cb_df = pd.DataFrame(columns= ['idx', 'time', 'cell', 'model_idx', 'test_cell', *scalar_cb_names] )

    cells = sorted(dataset.info['folder'].unique())
    
    try:
        test_cells = dataset.test_cells['test_cells']
    except:
        print("COULD NOT LOAD TEST CELLS")
        test_cells = []
        

    print("LEN DATASET: ", len(dataset))
    total_cnt = 0
    for cell in cells:
        if cell in skip: 
            print("Skipping cell %s"%cell)
            continue
        if 'all' not in include and cell not in include: 
            print("Skipping cell %s"%cell)
            continue


        filenames = dataset.info.loc[dataset.info['folder']==cell]['filename']
        indices = filenames.index.values
        
        sampler = SubsetSampler(indices)
        loader = torch.utils.data.DataLoader(dataset, 
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            pin_memory=True)

        hist_values_cell = {}
        t0 = time.time()
        for t, (sample, idx) in enumerate(zip(loader, indices)):
            for key in sample:
                sample[key] = sample[key].to(device)
                #inputs.requires_grad = True
            

            for model in models: 
                prediction = model(sample[model.input_type]).detach().cpu().numpy().squeeze() # Each prediction should have shape (2, H, W)
    
                cellmask = sample['mask'].detach().cpu().numpy().squeeze() > 0
                cellmask = scipy.ndimage.binary_dilation(cellmask, structure=disk(radius=50))

                output = sample['output'].detach().cpu().numpy().squeeze() # Should have shape [2, H, W]

                scalar_cb_df.loc[total_cnt] = np.nan # Initialize row
                scalar_cb_df.iloc[total_cnt]['idx'] = idx 
                scalar_cb_df.iloc[total_cnt]['cell'] = cell 
                scalar_cb_df.iloc[total_cnt]['time'] = t 
                scalar_cb_df.iloc[total_cnt]['model_idx'] = model.index 
                scalar_cb_df.iloc[total_cnt]['test_cell'] = 1 if cell in test_cells else 0

                for cb in scalar_callbacks:
                    cb_dict = cb(prediction, output, cellmask, angmag=model.angmag) # Should be dict with cb names keys
                    scalar_cb_df.iloc[total_cnt][list(cb_dict.keys())] = cb_dict 
                    
                for cb in hist_callbacks:
                    cb_dict = cb(prediction, output, cellmask, angmag=model.angmag) # Should be dict with cb names keys
                    cb_dict = {key+'_modelidx_%u'%model.index: cb_dict[key] for key in cb_dict }
                    for key in cb_dict:
                        if t==0:  hist_values_cell[key] = cb_dict[key] # Only for first time value
                        else: hist_values_cell[key][0] += cb_dict[key][0]

                total_cnt += 1

            if not t%20: print('Time for %u frames: %0.2f'%(t+1, time.time()-t0))

        with open(os.path.join(out_dir, 'histograms', 'hist_dict_%s.p'%cell), 'wb') as handle:
            pickle.dump( hist_values_cell, handle )

        if os.path.exists(os.path.join(out_dir, 'eval_prediction_values.csv')):
            print("PATH EXISTS, SAVING ELSEWHERE")
            scalar_cb_df.to_csv(os.path.join(out_dir, 'eval_prediction_values_1.csv'))
        else:
            scalar_cb_df.to_csv(os.path.join(out_dir, 'eval_prediction_values.csv'))

   # scalar_cb_df.to_csv(os.path.join(out_dir, 'eval_prediction_values.csv'))

    return




def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size




def peak_sum(regionmask, intensity_image):
        return np.sum(intensity_image)
       
        

class Forces(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, downsample=None, force_thresh=0.4):
        self.cb_names = ['sumF', 'sumFp', 
                         'MSE', 'MSEthreshold', 
                        ]# 'fbalance', 'fbalance_p', 'fbalance_thresh', 'fbalance_p_thresh']
        self.name = 'forces'

        self.thresh = force_thresh

        self.downsample=downsample

    def __call__(self, prediction, target, cellmask, angmag):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            
        with torch.no_grad():
            if angmag:
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
            
            #fbalance = np.sqrt( np.sum(fxT)**2+np.sum(fyT)**2)
            #fbalance_p = np.sqrt( np.sum(fxP)**2+np.sum(fyP)**2) 

            #fbalance_thresh = np.sqrt( np.sum(fxT[F>self.thresh])**2+np.sum(fyT[F>self.thresh])**2)
            #fbalance_p_thresh = np.sqrt( np.sum(fxP[Fp>self.thresh])**2+np.sum(fyP[Fp>self.thresh])**2) 

            sumF = np.sum(F[F>self.thresh])
            sumFp = np.sum(Fp[Fp>self.thresh])

        return_dict = {'sumF': sumF, 'sumFp': sumFp, 'MSE': mse.sum(), 'MSEthreshold': msethresh }# fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh]
        
        assert len(return_dict) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_dict), len(self.cb_names)) 
        return return_dict


class BoundaryStats(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, boundary_vals = [20,50,100], downsample=None, force_thresh=0.4):
        self.cb_names = ['boundary_sumF', 'boundary_sumFp', 
                         'boundary_sumFthresh', 'boundary_sumFpthresh',
                         'boundary_area',
                         'boundary_MSE', 'boundary_MSEthresh', 'boundary_dA']
        self.cb_names = [n+'_'+str(b) for n in self.cb_names for b in boundary_vals]
        self.boundary_vals = boundary_vals
        self.name = 'boundarystats'
        self.thresh=force_thresh

        self.downsample = downsample

    def __call__(self, prediction, target, cellmask, angmag):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            
        with torch.no_grad():
            if angmag:
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
            
            return_dict = {}
            for i, bound in enumerate(self.boundary_vals):
                boundary = (grad<bound)*(grad>0)
                return_dict['boundary_area_'+str(bound)] = np.sum(boundary!=0)
                return_dict['boundary_sumF_'+str(bound)] = np.sum( F[boundary])
                return_dict['boundary_sumFp_'+str(bound)] = np.sum( Fp[boundary])
                return_dict['boundary_sumFthresh_'+str(bound)] = np.sum( F[boundary*Fmask])
                return_dict['boundary_sumFpthresh_'+str(bound)] = np.sum( Fp[boundary*Fpmask])
                return_dict['boundary_MSE_'+str(bound)] = np.mean( mse[boundary])
                return_dict['boundary_MSEthresh_'+str(bound)] = np.mean( mse[boundary*Fmask])
                return_dict['boundary_dA_'+str(bound)] = np.sum( dang[boundary])

                
        assert len(return_dict) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_dict), len(self.cb_names)) 

        return return_dict


class PeakStats(object): # Returns fbalance, fbalance_p, fbalance_thresh, fbalance_p_thresh
    def __init__(self, peak_thresholds = [0.5, 1, 2], downsample=None, force_thresh=0.4):
        self.cb_names = ['peak_F_mean', 'peak_F_sum', 'peak_Fp_mean', 'peak_Fp_sum',
                        'peak_MSE_mean', 'peak_MSE_max', 'peak_DF_mean', 'peak_dA_mean',
                        'peak_DF_sum', 'peak_dA_sum', 'peak_dA_avg',
                        'peak_F_max', 'peak_Fp_max', 'peak_F_hit', 'peak_Fp_miss',
                        'peak_area_mean', 'peak_total_area'
                        ]

        self.cb_names = [n+'_'+str(p) for n in self.cb_names for p in peak_thresholds]
        self.peak_thresholds = peak_thresholds
        self.name = 'peakstats'

        self.thresh=force_thresh

        self.downsample=downsample

                
    def __call__(self, prediction, target, cellmask, angmag):
        # modelidx could be done more elegantly with __getitem__ probably
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
            assert np.all(target[0]<100) and np.all(prediction[0]<100)
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            raise Exception('or: \n Forces not normalized, max target %0.1f, max pred %0.1f'%(np.max(np.abs(target)), np.max(np.abs(prediction))))

            
        with torch.no_grad():
 
            if angmag:
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
        
            return_dict = {}
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

                return_dict['peak_F_mean_'+str(thresh)] = peakF.mean() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_Fp_mean_'+str(thresh)] = peakFp.mean() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_F_sum_'+str(thresh)] = peakFsum.sum() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_Fp_sum_'+str(thresh)] = peakFpsum.sum() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_F_max_'+str(thresh)] = peakF.max() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_Fp_max_'+str(thresh)] = peakFp.max() if len(np.unique(L))>1 else np.nan 

                return_dict['peak_MSE_mean_'+str(thresh)] = peakMSE.mean() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_MSE_max_'+str(thresh)] = peakMSE.max() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_MSE_max_'+str(thresh)] = peakMSE.max() if len(np.unique(L))>1 else np.nan # max of means 

                return_dict['peak_DF_mean_'+str(thresh)] = peakDF_frame_mean.mean() if len(np.unique(L))>1 else np.nan # max of means 
                return_dict['peak_dA_mean_'+str(thresh)] = peakDA_frame_mean.mean() if len(np.unique(L))>1 else np.nan # max of means 
                return_dict['peak_DF_sum_'+str(thresh)] = peakDF_frame_sum.sum() if len(np.unique(L))>1 else np.nan # max of means 
                return_dict['peak_dA_sum_'+str(thresh)] = peakDA_frame_sum.sum() if len(np.unique(L))>1 else np.nan # max of means 
                return_dict['peak_dA_avg_'+str(thresh)] = peak_Davg_ang.mean() if len(np.unique(L))>1 else np.nan # max of means 


                return_dict['peak_F_hit_'+str(thresh)] = np.sum(peakFhit)/np.sum(L!=0) if len(np.unique(L))>1 else np.nan 
                return_dict['peak_Fp_miss_'+str(thresh)] = np.sum(peakFpmiss)/np.sum(Lp!=0) if len(np.unique(L))>1 else np.nan 

                return_dict['peak_area_mean_'+str(thresh)] = peakarea.mean() if len(np.unique(L))>1 else np.nan 
                return_dict['peak_total_area_'+str(thresh)] = peakarea.sum() if len(np.unique(L))>1 else np.nan 


        assert len(return_dict) == len(self.cb_names), 'Len of return list (%u) and names (%u) not equal'%(len(return_dict), len(self.cb_names)) 
        return return_dict


class HistStats(object): 
    def __init__(self, boundary_vals=[20,50,100], peak_thresholds=[0.5,1.,2.], downsample=None, force_thresh=0.4):
        self.Nelem=101
        self.boundary_vals=boundary_vals
        self.peak_thresholds=peak_thresholds
        self.F_bins = np.linspace(0,8,self.Nelem)
        self.ang_bins = np.linspace(-np.pi,np.pi,self.Nelem)
        self.dang_bins = np.linspace(0,np.pi,self.Nelem)
        self.mse_bins = np.linspace(0,8,self.Nelem)
        self.dm_bins = np.linspace(-7,7,self.Nelem)
        self.cb_names = ['histF', 'histA', 'histAthresh', 'histMSE', 'histFA',
                        'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
                        'histFboundary',# 'histAboundary', 'histMSEboundary', 'histFAboundary',
                        'histFpeak',]# 'histApeak', 'histMSEpeak', 'histFApeak']
        #self.cb_names = ['histF', 'histA', 'histAthresh', 'histMSE', 'histFA',
        #                'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
        #                'histFboundary', 'histAboundary', 'histMSEboundary', 'histFAboundary',
        #                'histFpeak', 'histApeak', 'histMSEpeak', 'histFApeak']

        self.cb_names = [[n+'_'+str(p) for p in peak_thresholds] if 'peak' in n else [n] for n in self.cb_names]
        self.cb_names = [n for x in self.cb_names for n in x]
        self.cb_names = [[n+'_'+str(b)  for b in boundary_vals] if 'boundary' in n else [n] for n in self.cb_names]
        self.cb_names = [n for x in self.cb_names for n in x]
        
        self.thresh=force_thresh
        self.name = 'histstats'
        self.n_boundaries=len(self.boundary_vals)
        self.n_peaks=len(self.peak_thresholds)
        self.bins_all = {   'histF': [self.F_bins, self.F_bins],
                            'histA': [self.ang_bins, self.ang_bins], 
                            'histAthresh': [self.ang_bins, self.ang_bins], 
                            'histMSE': [self.F_bins, self.mse_bins],
                            'histFA': [self.F_bins, self.dang_bins],
                            'histFmask': [self.F_bins, self.F_bins], 
                            'histAmask': [self.ang_bins, self.ang_bins],
                            'histFAmask': [self.F_bins, self.dang_bins],
                            'histMSEmask': [self.F_bins, self.mse_bins],
                            'histFAboundary': [self.F_bins, self.dang_bins],
                            **{'histFboundary_'+str(b): [self.F_bins, self.F_bins] for b in boundary_vals},
                           # **{'histAboundary_'+str(b): [self.ang_bins, self.ang_bins] for b in boundary_vals},
                           # **{'histMSEboundary_'+str(b): [self.F_bins, self.mse_bins] for b in boundary_vals},
                           # **{'histFAboundary_'+str(b): [self.F_bins, self.dang_bins] for b in boundary_vals},
                            **{'histFpeak_'+str(p): [self.F_bins, self.F_bins] for p in peak_thresholds},
                           # **{'histApeak_'+str(p): [self.ang_bins, self.ang_bins] for p in peak_thresholds},
                           # **{'histMSEpeak_'+str(p): [self.F_bins, self.mse_bins] for p in peak_thresholds},
                           # **{'histFApeak_'+str(p): [self.F_bins, self.dang_bins] for p in peak_thresholds},
                        }

        self.downsample=downsample

    

    def __call__(self, prediction, target, cellmask, angmag):
    
        F_thresh=self.thresh
        try:
            assert len(prediction.shape) == 3 and len(target.shape)==3
        except:
            raise Exception("Shapes not correct, prediction: %s, target %s"%(str(prediction.shape), str(target.shape)))
            raise Exception('or: \n Forces not normalized, max target %0.1f, max pred %0.1f'%(np.max(np.abs(target)), np.max(np.abs(prediction))))
            
        with torch.no_grad():

            if angmag:
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

            grad = scipy.ndimage.morphology.distance_transform_edt((cellmask!=0)*1.)
            
            HFpeak = []
            Hangpeak = []
            Hmsepeak = []
            HFangpeak = []

            return_dict = {}

            for i,thresh in enumerate(self.peak_thresholds):
                L = F>thresh
                return_dict['histFpeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), Fp[L].ravel(), bins=(self.F_bins, self.F_bins))[0]
            #    return_dict['histApeak_'+str(thresh)] = np.histogram2d(ang[L].ravel(), angp[L].ravel(), bins=(self.ang_bins, self.ang_bins))[0] 
            #    return_dict['histMSEpeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), mse[L].ravel(), bins=(self.F_bins, self.mse_bins))[0] 
            #    return_dict['histFApeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), dang[L].ravel(), bins=(self.F_bins, self.dang_bins))[0] 
           

            return_dict['histF'] = np.histogram2d(F.ravel(), Fp.ravel(), bins=(self.F_bins, self.F_bins))[0]
            return_dict['histA'] = np.histogram2d(ang[F>F_thresh].ravel(), angp[F>F_thresh].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
            return_dict['histAthresh'] = np.histogram2d(ang[(F>F_thresh)*(Fp>F_thresh)].ravel(), angp[(Fp>F_thresh)*(F>F_thresh)].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
            return_dict['histMSE'] = np.histogram2d(F.ravel(), mse.ravel(), bins=(self.F_bins, self.mse_bins))[0]
            return_dict['histFA'] = np.histogram2d(F.ravel(), dang.ravel(), bins=(self.F_bins, self.dang_bins))[0]
           
            #'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
            return_dict['histFmask'] = np.histogram2d(F[cellmask].ravel(), Fp[cellmask].ravel(), bins=(self.F_bins, self.F_bins))[0]
            return_dict['histAmask'] = np.histogram2d(ang[(F>F_thresh)*cellmask].ravel(), angp[(F>F_thresh)*cellmask].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
            return_dict['histMSEmask'] = np.histogram2d(F[cellmask].ravel(), mse[cellmask].ravel(), bins=(self.F_bins, self.mse_bins))[0]
            return_dict['histFAmask'] = np.histogram2d(F[cellmask].ravel(), dang[cellmask].ravel(), bins=(self.F_bins, self.dang_bins))[0]
           

            for i, bound in enumerate(self.boundary_vals):
                boundary = (grad<bound)*(grad>0)
            
                return_dict['histFboundary_'+str(bound)] = np.histogram2d(F[boundary].ravel(), Fp[boundary].ravel(), bins=(self.F_bins, self.F_bins))[0]
            #    return_dict['histAboundary_'+str(bound)] = np.histogram2d(ang[boundary*(F>F_thresh)].ravel(), angp[boundary*(F>F_thresh)].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
            #    return_dict['histMSEboundary_'+str(bound)] = np.histogram2d(F[boundary].ravel(), mse[boundary].ravel(), bins=(self.F_bins, self.mse_bins))[0] 
            #    return_dict['histFAboundary_'+str(bound)] = np.histogram2d(F[boundary].ravel(), dang[boundary].ravel(), bins=(self.F_bins, self.dang_bins))[0] 
                    
        
        assert len(return_dict)==len(self.cb_names), "%u not equal to %u"%(len(return_dict), len(self.cb_names))

        return_dict = {k: [return_dict[k], *self.bins_all[k]] for k in return_dict}        

        return return_dict
    
    
    def batched_hist(self, prediction, target, cellmask, angmag):
    
        F_thresh=self.thresh
        
        #with torch.no_grad():
            
        #print(prediction.shape, target.shape, cellmask.shape)

        if angmag:
            fxT, fyT = target[:, 0]*np.cos(target[:, 1]), target[:, 0]*np.sin(target[:, 1])
            fxP, fyP = prediction[:, 0]*np.cos(prediction[:, 1]), prediction[:, 0]*np.sin(prediction[:, 1])
        else:
            fxT, fyT = target[:, 0], target[:, 1]
            fxP, fyP = prediction[:, 0], prediction[:, 1]

        fxT[cellmask==0] = 0
        fyT[cellmask==0] = 0
        fxP[cellmask==0] = 0
        fyP[cellmask==0] = 0


        if self.downsample is not None:
            fxT = skimage.measure.block_reduce(fxT, block_size=(1, self.downsample,self.downsample), func=np.mean)
            fyT = skimage.measure.block_reduce(fyT, block_size=(1, self.downsample,self.downsample), func=np.mean)
            fxP = skimage.measure.block_reduce(fxP, block_size=(1, self.downsample,self.downsample), func=np.mean)
            fyP = skimage.measure.block_reduce(fyP, block_size=(1, self.downsample,self.downsample), func=np.mean)
            cellmask = skimage.measure.block_reduce(cellmask*1., block_size=(1, self.downsample,self.downsample), func=np.mean)
            cellmask = cellmask.astype(int)

        #print(prediction.shape, target.shape, cellmask.shape)

        mse = (fxT-fxP)**2 + (fyT-fyP)**2

        F = np.sqrt(fxT**2 + fyT**2)
        Fp = np.sqrt(fxP**2 + fyP**2)

        ang = np.arctan2(fyT, fxT)
        angp = np.arctan2(fyP, fxP)
        dang = np.abs(np.remainder(ang - angp + np.pi, 2*np.pi) - np.pi) #L1

        dang[F<F_thresh] = 0

        #grad = scipy.ndimage.morphology.distance_transform_edt((cellmask!=0)*1.)

        HFpeak = []
        Hangpeak = []
        Hmsepeak = []
        HFangpeak = []

        return_dict = {}

        for i,thresh in enumerate(self.peak_thresholds):
            L = F>thresh
            return_dict['histFpeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), Fp[L].ravel(), bins=(self.F_bins, self.F_bins))[0]
            #return_dict['histApeak_'+str(thresh)] = np.histogram2d(ang[L].ravel(), angp[L].ravel(), bins=(self.ang_bins, self.ang_bins))[0] 
        #    return_dict['histMSEpeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), mse[L].ravel(), bins=(self.F_bins, self.mse_bins))[0] 
        #    return_dict['histFApeak_'+str(thresh)] = np.histogram2d(F[L].ravel(), dang[L].ravel(), bins=(self.F_bins, self.dang_bins))[0] 


        return_dict['histF'] = np.histogram2d(F.ravel(), Fp.ravel(), bins=(self.F_bins, self.F_bins))[0]
        return_dict['histA'] = np.histogram2d(ang[F>F_thresh].ravel(), angp[F>F_thresh].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
        return_dict['histAthresh'] = np.histogram2d(ang[(F>F_thresh)*(Fp>F_thresh)].ravel(), angp[(Fp>F_thresh)*(F>F_thresh)].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
        return_dict['histMSE'] = np.histogram2d(F.ravel(), mse.ravel(), bins=(self.F_bins, self.mse_bins))[0]
        return_dict['histFA'] = np.histogram2d(F.ravel(), dang.ravel(), bins=(self.F_bins, self.dang_bins))[0]

        #'histFmask', 'histAmask', 'histMSEmask', 'histFAmask',
        return_dict['histFmask'] = np.histogram2d(F[cellmask].ravel(), Fp[cellmask].ravel(), bins=(self.F_bins, self.F_bins))[0]
        return_dict['histAmask'] = np.histogram2d(ang[(F>F_thresh)*cellmask].ravel(), angp[(F>F_thresh)*cellmask].ravel(), bins=(self.ang_bins, self.ang_bins))[0]
        return_dict['histMSEmask'] = np.histogram2d(F[cellmask].ravel(), mse[cellmask].ravel(), bins=(self.F_bins, self.mse_bins))[0]
        return_dict['histFAmask'] = np.histogram2d(F[cellmask].ravel(), dang[cellmask].ravel(), bins=(self.F_bins, self.dang_bins))[0]


        #assert len(return_dict)==len(self.cb_names), "%u not equal to %u"%(len(return_dict), len(self.cb_names))

        return_dict = {k: [return_dict[k], *self.bins_all[k]] for k in return_dict if 'boundary' not in k}        

        return return_dict



scalar_callback_dict = {'forces': Forces, 
                        'boundary': BoundaryStats, 
                        'peaks': PeakStats }




hist_callback_dict = {'hists': HistStats}






