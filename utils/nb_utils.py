import numpy as np
import matplotlib.pyplot as plt
import itertools

# For data class:
import pandas as pd 
#from predict import parse_file_args 
import os
#from natsort import natsorted
from argparse import Namespace
#from data_processing import prediction_transforms

import matplotlib.colors as colors
import matplotlib.cm
import copy
import skimage.measure as measure

def linescan(image, points, width=3):
    segments = []
    for p, pnext in zip(points[:-1], points[1:]):
        x0, y0 = p # These are in _pixel_ coordinates!!
        x1, y1 = pnext
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

        segments.append([x, y])
        
    shifts = np.arange(-(width//2), (width//2))
    
    segment_lens = [len(s[0]) for s in segments]
    segment_lens = [0, *segment_lens]
    
    dline = np.zeros((width**2, np.sum(segment_lens)))
    xlines = np.zeros((width**2, np.sum(segment_lens)))
    ylines = np.zeros((width**2, np.sum(segment_lens)))

    # Actual line 
    xL = np.zeros(np.sum(segment_lens))
    yL = np.zeros(np.sum(segment_lens))

    segment_lens = np.cumsum(segment_lens)

    for i, line in enumerate(segments):
        for d, (dx, dy) in enumerate(itertools.product(shifts, shifts)):
            dline[d][segment_lens[i]:segment_lens[i+1]] = image[ line[0].astype(int) + dy, line[1].astype(int) + dx]
            ylines[d][segment_lens[i]:segment_lens[i+1]] = line[0].astype(int) + dy
            xlines[d][segment_lens[i]:segment_lens[i+1]] = line[1].astype(int) + dx
            
            if dx==0 and dy==0:
                yL[segment_lens[i]:segment_lens[i+1]] = line[0].astype(int)
                xL[segment_lens[i]:segment_lens[i+1]] = line[1].astype(int)

    dline = np.mean(dline, axis=0)
    
    return [xL, yL], dline, xlines, ylines

def nxtkey(dictionary, layer=0):
    if layer==0:
        return list(dictionary.keys())[0]
    
    if layer==1:
        layer1key = list(dictionary.keys())[0]
        
        return list(dictionary[layer1key].keys())[0]
        
        

def ax_adjust(ax, axlw=3, tickparams={"direction": "in"}):
    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(axlw)
            
    ax.tick_params(axis="both", **tickparams, width=axlw)
    return

def get_yavg_scatter(xs, ys, nbins=20, crop=None):
    
    if crop:
        bins = np.linspace(0, np.min([crop, np.min([np.max(ys), np.max(xs)])]), nbins+1)
    else:
        bins = np.linspace(0, np.max([np.max(ys), np.max(xs)]), nbins+1)

    H, _, _ = np.histogram2d(xs, ys, bins=(bins, bins))
    H += 1e-3

    H = H.T
    # Assume ybins is just 
    bins = 0.5*(bins[1:]+bins[:-1])
    ymat = np.tile(bins[..., None], (1,len(bins)))  # Shape now (n, n). Each column is an array of ybins    
    yavg = np.average(ymat, weights=H, axis=0)
    std = np.average((ymat-yavg)**2, weights=H, axis=0)
    #std = np.std(hist2D, axis=0)#/np.sqrt(np.sum(hist2D, axis=0))
    
    return yavg,std,bins

def get_yavg(hist2D, ybins, thresh=None, crop=0, threshby='sum', angle=False, stderr=False):
    # Assume ybins is just 
    
    if crop:
        ybins = ybins[crop:]
        hist2D = hist2D[crop:]
        
    if angle:
        shifts = np.arange(-len(hist2D)//2, len(hist2D)//2)[::-1]
        rows, column_indices = np.ogrid[:hist2D.shape[0], :hist2D.shape[1]]

        # Use always a negative shift, so that column_indices are valid.
        # (could also use module operation)
        shifts[shifts < 0] += hist2D.shape[1]
        column_indices = column_indices - shifts[:, np.newaxis]

        hist2D = hist2D.T[rows, column_indices]
        hist2D = hist2D.T
        diff = np.max(ybins)-np.min(ybins)

    if len(ybins)==len(hist2D)+1:
        ybins = 0.5*(ybins[1:]+ybins[:-1])
    ymat = np.tile(ybins[..., None], (1,hist2D.shape[1]))  # Shape now (n, n). Each column is an array of ybins    
    yavg = np.average(ymat, weights=hist2D, axis=0)
    
    if stderr: std = np.sqrt(np.average((ymat-yavg)**2, weights=hist2D, axis=0))/np.sum(hist2D, axis=0)
    else: std = np.sqrt(np.average((ymat-yavg)**2, weights=hist2D, axis=0))
    
    
    if angle:
        shifts = np.arange(-len(hist2D)//2, len(hist2D)//2)[::-1]
        yavg -= shifts*diff/len(shifts)
    
    if threshby=='sum':
        yavg[np.sum(hist2D, axis=0)<thresh] = np.nan
        std[np.sum(hist2D, axis=0)<thresh] = np.nan
    if threshby=='max':
        yavg[np.max(hist2D, axis=0)>thresh] = np.nan
        std[np.max(hist2D, axis=0)>thresh] = np.nan
    if threshby=='maxmin':
        yavg[np.max(hist2D, axis=0)<thresh[0]] = np.nan
        std[np.max(hist2D, axis=0)<thresh[0]] = np.nan
        

    return yavg,std
    
    
def make_vector_field(component1, component2, downsample=1, threshold=None, angmag=False):
    """
    Assumes comp1, comp2 are the same size, useful for plotting on images as I often do
    """
    if angmag: # component1 = mag, component2 = ang
        vx, vy = component1*np.cos(component2), component1*np.sin(component2)
    else:
        vx, vy = component1, component2

    X, Y = np.meshgrid( np.arange(vx.shape[1]), np.arange(vx.shape[0]))

    X = measure.block_reduce(X, (downsample, downsample), np.mean)
    Y = measure.block_reduce(Y, (downsample, downsample), np.mean)
    vx = measure.block_reduce(vx, (downsample, downsample), np.mean)
    vy = measure.block_reduce(vy, (downsample, downsample), np.mean)
    
    if threshold is not None:
        mask = np.sqrt(vx**2 + vy**2)>threshold
        X = X[mask]
        Y = Y[mask]
        vx = vx[mask]
        vy = vy[mask]
    
    return X, Y, vx, vy

def smooth(x, window=5): return np.convolve(x, np.ones(window)/window, mode='valid')

def log_norm(vmin=None, vmax=None):
    norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    return norm

def log_cmap(cmap):
    my_cmap = copy.copy(matplotlib.cm.get_cmap(cmap)) # copy the default cmap
    my_cmap.set_bad((0,0,0))
    return my_cmap
## Get distance function from a point

def get_radii(image_size=768, center=None, im_center=None):
    image = np.zeros((image_size, image_size))
    
    if im_center is None:
        center_im = np.asarray(image.shape)//2
    else:
        center_im = [0,0]

    if center is None:
        center = center_im
    else:
        center = center + center_im

    Y, X = np.ogrid[:image_size, :image_size]
    
    dist = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    return dist 


def rad_avg(image):
    R = get_radii(image_size=image.shape[0]) # assume square image shape
    
    f = lambda r : image[(R >= r-.5) & (R < r+.5)].mean()
    r  = np.arange(0,R.max())
    f = np.vectorize(f)
    
    return f(r)

def get_label_statistics(image, labels, max_labels=10):
    nlabels = max_labels # assume 0 is no label

    label_sets = [image[labels==(L+1)] if np.any(labels==L+1) else 0 for L in range(nlabels)]
    means = np.asarray([np.mean(L) for L in label_sets])
    stds = np.asarray([np.std(L) for L in label_sets])

    return means, stds

## Utils for making fake cells

def make_ellipse(radius, dx, dy, image_size=768, center=None, angle=0):
    image = np.zeros((image_size, image_size))
    center_im = np.asarray(image.shape)//2

    if center is None:
        center = center_im
    else:
        center = center + center_im

    Y, X = np.ogrid[:image_size, :image_size]

    X_tilde = (X - center[0])*np.cos(angle) + (Y-center[1])*np.sin(angle)
    Y_tilde = (X - center[0])*np.sin(angle) - (Y-center[1])*np.cos(angle)
    
    Ellipse_dist_from_center = np.sqrt(X_tilde**2/dx**2 + Y_tilde**2/dy**2)

    mask = Ellipse_dist_from_center < radius
    
    return mask #,Ellipse_dist_from_center, dist_from_center

def make_circle(radius, imsize=512, center=None):
    image = np.zeros((imsize, imsize))
    
    image_center = np.asarray(image.shape)//2
    if center is not None:
        center = image_center + center
    else:
        center = image_center
        
    Y, X = np.ogrid[:imsize, :imsize]

    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center < radius
    
    return mask, dist_from_center

def make_square(radius, imsize=512, center=None):
    image = np.zeros((imsize, imsize))
    if center is None:
        center = np.asarray(image.shape)//2

    Y, X = np.ogrid[:imsize, :imsize]

    dist_from_center = np.maximum(np.abs(X - center[0]), np.abs(Y-center[1]))

    mask = dist_from_center < radius
    
    return mask, dist_from_center


def make_cell(radius=500, FA_density=1, FA_len=50, FA_width=10, image_size=768):
    N = np.round(radius*2*np.pi/FA_width)
    angles = 2*np.pi*np.linspace(0,1,int(N*FA_density))[:-1]
    
    image = np.zeros((image_size, image_size))
    for i,a in enumerate(angles):
        image+= make_ellipse(FA_len, 1, FA_width/FA_len/2, center=[radius*np.cos(a), radius*np.sin(a)], angle=a, image_size = image_size)
    
    return image
    
    
class EnsemblePredictionData(object):
    terms_rank1 = ['6_{;a}', '6_{;bba}', '6_{;}6_{;a}', '6_{;}6_{;bba}', '6_{;a}6_{;bb}', '6_{;b}6_{;ba}', 
      '6_{;}6_{;}6_{;a}', '6_{;}6_{;}6_{;bba}', '6_{;}6_{;a}6_{;bb}', '6_{;}6_{;b}6_{;ba}', '6_{;b}6_{;b}6_{;a}']

    terms_rank0 = ['6_{;}', '6_{;}',
     '6_{;aa}',
     '6_{;}6_{;}',
     '6_{;}6_{;aa}',
     '6_{;a}6_{;a}',
     '6_{;}6_{;}6_{;}',
     '6_{;}6_{;}6_{;aa}',
     '6_{;}6_{;a}6_{;a}']

    terms_rank0_tex = ['$\\zeta$', '$\\zeta$',
     '$\\nabla^2\\zeta$',
     '$\\zeta^2$',
     '$\\zeta\\nabla^2\\zeta$',
     '$(\\nabla\\zeta)^2$',
     '$\\zeta^3$',
     '$\\zeta^2\\nabla^2\\zeta$',
     '$\\zeta(\\nabla\\zeta)^2$']
    
    def terms(self, rank=0):
        if rank==0:
            return self.terms_rank0_tex
        
    
    def __init__(self, prot, test_split_selection=None, filename_label='', rootopt=None, cells=None):
        
        if prot=='myo':
            root='/project/vitelli/cell_stress/ForcePrediction_03_06_21_Myo/AM_myo_protein_combos'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_210910_1643_AM_myo_protein_combos/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '7': 'Myosin', '6,7': 'Zyx+Myo'}
        elif prot=='act':
            root = '/project/vitelli/cell_stress/ForcePrediction_Actin_All/AM_act_protein_combos'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_210909_0902_AM_act_protein_combos/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '7': 'Actin', '6,7': 'Zyx+Act'}
        elif prot=='pax':
            #root = '/project/vitelli/cell_stress/ForcePrediction_27_04_21_Pax/AM_pax_protein_combos'
            root = '/project/vitelli/cell_stress/ForcePrediction_Paxillin_All/AM_pax_protein_combos_new'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_210922_0837_AM_pax_protein_combos_new/test_indices_all.txt'
            #testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_210909_0847_AM_pax_protein_combos/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '7': 'Paxillin', '6,7': 'Zyx+Pax'}
        elif prot=='all':
            root = '/project/vitelli/cell_stress/ForcePrediction_All_16kpa/AM_all_bigrun'
            #root = '/project/vitelli/cell_stress/ForcePrediction_All_16kpa/AM_all_bigrun'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_210904_1148_AM_all_bigrun/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '6,7': 'Zyx+Mask'}
        elif prot=='most':
            root = '/project/vitelli/cell_stress/ForcePrediction_Most_16kpa/AM_most_most'
            #root = '/project/vitelli/cell_stress/ForcePrediction_All_16kpa/AM_all_bigrun'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_211105_1508_AM_most_most/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '6,7': 'Zyx+Mask'}
        elif prot=='fixed':
            root = '/project/vitelli/cell_stress/ForcePrediction_All_16kpa_new/AM_all_fixed2'
            #root = '/project/vitelli/cell_stress/ForcePrediction_All_16kpa/AM_all_bigrun'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_211108_2050_AM_all_fixed2/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '6,7': 'Zyx+Mask'}
        elif prot=='u2os':
            root = '/project/vitelli/cell_stress/ForcePrediction_21_11_16_U2OS/AM_all_fixed_NormedF'
            testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_211108_2050_AM_all_fixed2/test_indices_all.txt'
            self.model_map={'4': 'Mask', '6': 'Zyxin', '6,7': 'Zyx+Mask'}
            
        if rootopt is not None:
            root = os.path.join(os.path.dirname(root), rootopt)
            
        print(root)
            
        
        self.cells_include = cells
            
        self.filename_label=filename_label


        self.root = root
        self.test_split_arr =  pd.read_csv(testsplits, header=None)
        
        if test_split_selection is not None:
            self.test_split_arr = self.test_split_arr.loc[test_split_selection]
        self.test_split_selection = test_split_selection
        
        
        args = parse_file_args(os.path.join(os.path.dirname(testsplits), 'testsplit_0'))
        args = Namespace(**{'crop_size': 960}, **vars(args), **{'lr': 0})
        self.transforms = prediction_transforms(args)
        
        self.get_dirs_all()
        self.load_csv_files()
        self.load_hist_files()
        self.pool_data_by_model()
        self.pool_data_by_split()
        
        self.test_cells={run: list(self.dirs_test[run].keys()) for run in self.dirs_test}
        self.train_cells={run: list(self.dirs_train[run].keys()) for run in self.dirs_test}
        
    def get_dirs_all(self):
        self.dirs_test = {}
        self.dirs_train = {}
        for testsplit in natsorted(next(os.walk(self.root))[1]):
            try:
                split_idx = int(testsplit.split('_')[-1])
                testcells = list(self.test_split_arr.loc[split_idx])
            except:
                split_idx = testsplit.split('_')[-1]
                testcells = list(self.test_split_arr.loc[0])
                print("USING SAME TEST CELLS FOR EACH RUN")
            if self.test_split_selection is not None and split_idx not in self.test_split_selection:
                continue
            
            self.dirs_test[split_idx] = {celldir: os.path.join(self.root, testsplit, celldir) 
                                         for celldir in natsorted(next(os.walk(os.path.join(self.root, testsplit)))[1]) 
                                         if celldir in testcells}

            self.dirs_train[split_idx] = {celldir: os.path.join(self.root, testsplit, celldir) 
                                       for celldir in natsorted(next(os.walk(os.path.join(self.root, testsplit)))[1]) 
                                       if celldir not in testcells}
        
            if self.cells_include is not None:
                self.dirs_test[split_idx] = {cell: item for cell,item in self.dirs_test[split_idx].items() if cell in self.cells_include}
                self.dirs_train[split_idx] = {cell: item for cell,item in self.dirs_train[split_idx].items() if cell in self.cells_include}

            if len(self.dirs_test[list(self.dirs_test.keys())[0]]) == 0:
                self.dirs_test=self.dirs_train
            if len(self.dirs_train[list(self.dirs_train.keys())[0]]) == 0:
                self.dirs_train=self.dirs_test
        return

    def load_csv_files(self):
        dirs_test = self.dirs_test
        dirs_train = self.dirs_train
        
        print(self.filename_label)

        self.csv_files_test = {testsplit: 
             {celldir: 
              [pd.read_csv(os.path.join(dirs_test[testsplit][celldir], file)) for file in natsorted(os.listdir(dirs_test[testsplit][celldir])) 
                  if 'callback_values%s.csv'%self.filename_label in file]
              for celldir in dirs_test[testsplit]} 
             for testsplit in dirs_test}

        self.csv_model_order_test = {testsplit: 
                     {celldir: 
                     [file for file in natsorted(os.listdir(dirs_test[testsplit][celldir])) 
                          if 'callback_values%s.csv'%self.filename_label in file]
                      for celldir in dirs_test[testsplit]} 
                     for testsplit in dirs_test}

        self.csv_filemapping_test = {testsplit: 
                     {celldir: 
                      pd.read_csv(os.path.join(dirs_test[testsplit][celldir], 'file_mapping.csv'))
                      for celldir in dirs_test[testsplit]} 
                     for testsplit in dirs_test}

        self.csv_files_train = {testsplit: 
                     {celldir: 
                      [pd.read_csv(os.path.join(dirs_train[testsplit][celldir], file)) for file in natsorted(os.listdir(dirs_train[testsplit][celldir])) 
                          if 'callback_values%s.csv'%self.filename_label in file]
                      for celldir in dirs_train[testsplit]} 
                     for testsplit in dirs_train}

        self.csv_filemapping_train = {testsplit: 
                     {celldir: 
                      pd.read_csv(os.path.join(dirs_train[testsplit][celldir], 'file_mapping.csv'))
                      for celldir in dirs_train[testsplit]} 
                     for testsplit in dirs_train}


        self.csv_model_order_train = {testsplit: 
                     {celldir: 
                      [file for file in natsorted(os.listdir(dirs_train[testsplit][celldir])) 
                          if 'callback_values%s.csv'%self.filename_label in file]
                      for celldir in dirs_train[testsplit]} 
                     for testsplit in dirs_train}
        
        self.get_model_names()
        
        return
    
    def get_model_names(self):
        self.model_names = np.unique([i for k,i in self.csv_model_order_test[list(self.csv_model_order_test.keys())[0]].items()])
        model_names2 = np.unique([i for k,i in self.csv_model_order_train[list(self.csv_model_order_train.keys())[0]].items()])
        
        if len(self.model_names)<len(model_names2):
            print("TRAIN TEST HAVE DIFFERENT NUMBER OF MODELS")
            self.model_names = model_names2
        assert np.all(self.model_names==model_names2), "\n%s\n%s"%(self.model_names, model_names2)
        
        
        
        self.n_models = len(self.model_names)
        self.model_channels = [[s[1:] for s in mn.split('_') if s[0]=='i'][0] for mn in self.model_names]

        self.model_names = {i: name for i, name in zip(self.model_channels, self.model_names)}
        self.models = list(self.model_names.keys())
        print("Unique model order (should only be one)\n", self.model_names)
        return
    
    def load_hist_files(self, model=0):
        dirs_test = self.dirs_test
        dirs_train = self.dirs_train
        root = self.root
        
        print('hist_dict%s'%self.filename_label)
        
        self.hist_files_test = {testsplit: 
             {celldir: 
              [np.load(os.path.join(dirs_test[testsplit][celldir], file), allow_pickle=True) for file in natsorted(os.listdir(dirs_test[testsplit][celldir])) 
                  if 'hist_dict%s'%self.filename_label in file][model]
              for celldir in dirs_test[testsplit]} 
             for testsplit in dirs_test}

        self.hist_files_train = {testsplit: 
                     {celldir: 
                      [np.load(os.path.join(dirs_train[testsplit][celldir], file), allow_pickle=True) for file in natsorted(os.listdir(dirs_train[testsplit][celldir])) 
                          if 'hist_dict%s'%self.filename_label in file][model]
                      for celldir in dirs_train[testsplit]} 
                     for testsplit in dirs_train}
        
        return
        

    #pooled = [[cell[n] for cell in csv_files if len(cell)==len(csv_files[0])] for n in range(len(csv_files[0]))]
    #pooled_by_model = [[cell[n] for cell in csv_files if len(cell)==len(csv_files[0])] for n in range(len(csv_files[0]))]
    #pooled = [pd.concat([cell for cell in model], ignore_index=True) for model in pooled]

    def get_sample(self, idx, cell, test_train, frame, suppress_output=True, strcrop=2):
        possible_cells_test = list(self.csv_filemapping_test[idx].keys())
        possible_cells_train = list(self.csv_filemapping_train[idx].keys())

        if test_train=='test':
            assert (cell in possible_cells_test), "Cell must be in test set %s"%str(possible_cells_test)
        elif test_train=='train':
            assert (cell in possible_cells_train), "Cell must NOT be in test set %s"%str(possible_cells_train)
        
        if not suppress_output:
            print(possible_cells_test)
            print(possible_cells_train)
            
        if test_train=='test':
            csvrow = self.csv_filemapping_test[idx][cell].iloc[frame]
            modelcols = [x for x in self.csv_filemapping_test[idx][cell].columns if 'prediction_dir' in x]
            target = np.load( os.path.join(csvrow.source_dir[strcrop:-strcrop].replace('\'', ''), csvrow.file[strcrop:-strcrop].replace('\'', '')))
            modelouts = [ np.load( os.path.join(csvrow[modelcols[n]], csvrow.file[strcrop:-strcrop].replace('\'', ''))) for n in range(len(modelcols))]
            modelnames = [os.path.basename(self.csv_filemapping_test[idx][cell][m].iloc[frame]) for m in modelcols]

        if test_train=='train':
            csvrow = self.csv_filemapping_train[idx][cell].iloc[frame]
            modelcols = [x for x in self.csv_filemapping_train[idx][cell].columns if 'prediction_dir' in x]
            target = np.load( os.path.join(csvrow.source_dir[strcrop:-strcrop].replace('\'', ''), csvrow.file[strcrop:-strcrop].replace('\'', '')))
            modelouts = [ np.load( os.path.join(csvrow[modelcols[n]], csvrow.file[strcrop:-strcrop].replace('\'', ''))) for n in range(len(modelcols))]
            modelnames = [os.path.basename(self.csv_filemapping_train[idx][cell][m].iloc[frame]) for m in modelcols]

        return self.transforms(target), modelouts, modelnames

    def get_error(self, idx, cell, test_train, frame, thresh=0.4, Fthresh=0.4):
        if test_train=='test':
            csvrow = self.csv_filemapping_test[idx][cell].iloc[frame]
            modelcols = [x for x in self.csv_filemapping_test[idx][cell].columns if 'prediction_dir' in x]
            target = np.load( os.path.join(csvrow.source_dir[2:-2], csvrow.file[2:-2]))
            modelouts = [ np.load( os.path.join(csvrow[modelcols[n]], csvrow.file[2:-2])) for n in range(len(modelcols))]
        if test_train=='train':
            csvrow = self.csv_filemapping_train[idx][cell].iloc[frame]
            modelcols = [x for x in self.csv_filemapping_train[idx][cell].columns if 'prediction_dir' in x]
            target = np.load( os.path.join(csvrow.source_dir[2:-2], csvrow.file[2:-2]))
            modelouts = [ np.load( os.path.join(csvrow[modelcols[n]], csvrow.file[2:-2])) for n in range(len(modelcols))]

        #print([os.path.basename(csvrow[m]) for m in modelcols])
        target = transforms(target)
        target = target[[2,3]].detach().cpu().numpy()
        mag = np.linalg.norm(target, axis=0)
        ang = np.arctan2(target[1], target[0])
        modelouts = [m.squeeze() for m in modelouts]

        F_errors = [m[0] - mag  for m in modelouts]

        ang_errors = [np.abs(np.remainder(ang - m[1] + np.pi, 2*np.pi) - np.pi) for m in modelouts]

        for a,pred in zip(ang_errors, modelouts):
            a[(~(mag>thresh))*(~(pred[0]>thresh))] = np.nan

        for df, pred in zip(F_errors, modelouts):
            df[(~(mag>Fthresh))*(~(pred[0]>Fthresh))] = 0

        return F_errors, ang_errors
    
    def plot_prediction_vfs(self, testsplitidx, cell, testtrain, frame, return_ims=False, strcrop=2, vmax=2, vmin=0, threshold=0.4, scale=10, disp=False, figscale=8):
        possible_cells = np.asarray(self.test_split_arr.iloc[testsplitidx])
        #print(possible_cells, cell)
        #if testtrain=='test':
        #    assert (cell in possible_cells), "Cell must be in test set %s"%str(possible_cells)
        #elif testtrain=='train':
        #    assert (cell not in possible_cells), "Cell must NOT be in test set %s"%str(possible_cells)
        
        x,y,m = self.get_sample(testsplitidx, cell, testtrain, frame, strcrop=strcrop)
        
        fig,ax=plt.subplots(1, len(y)+2, figsize=(figscale*(len(y)+2), figscale), constrained_layout=1)
        
        if disp:
            target=x[[0,1]]
        else:
            target=x[[2,3]]
            
        ax[0].imshow(x[6]/x[6].max(), origin='lower', vmax=0.3, cmap='gray')
        ax[0].axis('off')

        for n, a in enumerate(ax[1:]):
            if n==0:
                a.imshow( np.linalg.norm(target, axis=0), origin='lower', cmap='inferno', vmax=vmax, vmin=vmin)
                a.quiver(*make_vector_field(*target, downsample=20, threshold=threshold, angmag=False), color='w', width=0.003, scale=scale)
            else:
                a.imshow( y[n-1][0][0], origin='lower', cmap='inferno', vmax=vmax, vmin=vmin)
                a.quiver(*make_vector_field(*y[n-1][0], downsample=20, threshold=threshold, angmag=True), color='w', width=0.003, scale=scale)

            a.axis('off')
        plt.show()
        if return_ims:
            return x, y, m
        else:
            return
    
    
    def plot_MSE(self, testsplitidx, cell, testtrain, frame, normalize='sumF'):
        possible_cells = np.asarray(self.test_split_arr.iloc[testsplitidx])
        print(possible_cells, cell)
        if testtrain=='test':
            assert (cell in possible_cells), "Cell must be in test set %s"%str(possible_cells)
        elif testtrain=='train':
            assert (cell not in possible_cells), "Cell must NOT be in test set %s"%str(possible_cells)

        x,y,m = self.get_sample(testsplitidx, cell, testtrain, frame)
        
        fig,ax=plt.subplots(1, len(y), figsize=(8*(len(y)), 8), constrained_layout=1)

        print(m)
        for n, a in enumerate(ax):
            
            magp = y[n][0][0]
            angp = y[n][0][1]
            fxp = magp*np.cos(angp)
            fyp = magp*np.sin(angp)

            MSE= (x[2]-fxp)**2 + (x[3]-fyp)**2
            
            MSE[magp > np.linalg.norm(x[[2,3]], axis=0)] = 0

            a.imshow( MSE, origin='lower', cmap='inferno', vmax=2)
            
            a.text(0.02, 0.98, 'Sum %0.2f'%MSE.sum(), color='w', fontsize=20, transform=a.transAxes)

            a.axis('off')
            
        plt.show()
        
        return
    
    def pool_data_by_model(self):
        dirs_test = self.dirs_test
        dirs_train = self.dirs_train
        
        pooled_test_by_model =  {m:
            [self.csv_files_test[testsplit][celldir][i] for testsplit in dirs_test for celldir in dirs_test[testsplit] if len(self.csv_files_test[testsplit][celldir])!=0]
                                for i, (m, modelname) in enumerate(self.model_names.items())}
        pooled_train_by_model =  {m:
            [self.csv_files_train[testsplit][celldir][i] for testsplit in dirs_train for celldir in dirs_train[testsplit] if len(self.csv_files_train[testsplit][celldir])!=0]
                                for i, (m, modelname) in enumerate(self.model_names.items())}

        pooled_test = {m: pd.concat(PM, ignore_index=True) for m, PM in pooled_test_by_model.items()}
        pooled_train = {m: pd.concat(PM, ignore_index=True) for m, PM in pooled_train_by_model.items()}
        
        self.pooled_model = {'test': pooled_test, 'train': pooled_train}
        self.pooled_model_help = "pooled_model['test'/'train'] ['4'/'6,7' etc. (model)]"
        return
        
    def pool_data_by_split(self):
        dirs_test = self.dirs_test
        dirs_train = self.dirs_train
        
        pool_by_split_test = {m:
            [[self.csv_files_test[testsplit][celldir][i] for celldir in dirs_test[testsplit] if len(self.csv_files_test[testsplit][celldir])!=0] 
                                  for testsplit in dirs_test] 
                                for i, (m, modelname) in enumerate(self.model_names.items())}
        lensall= {m:
            [[ len(self.csv_files_test[testsplit][celldir]) for celldir in dirs_test[testsplit] ] 
                                  for testsplit in dirs_test] 
                                for i, (m, modelname) in enumerate(self.model_names.items())}
        emptycells= {m:
            [[ celldir for celldir in dirs_test[testsplit] if len(self.csv_files_test[testsplit][celldir])==0] 
                                  for testsplit in dirs_test] 
                                for i, (m, modelname) in enumerate(self.model_names.items())}
        #print('Lensall 1\n', lensall)
        #print('Celldirs \n', emptycells)
        pool_by_split_train = {m:
            [[self.csv_files_train[testsplit][celldir][i] for celldir in dirs_train[testsplit] if len(self.csv_files_train[testsplit][celldir])!=0] 
                                   for testsplit in dirs_train] 
                                for i, (m, modelname) in enumerate(self.model_names.items())}

        #print(pool_by_split_test.keys())
        # Len 
        lensall = {m: [len(x) for x in PM] for m,PM in pool_by_split_test.items()}
        print(lensall)

        
        pool_by_split_test = {m: [pd.concat(x, ignore_index=True) for x in PM] for m,PM in pool_by_split_test.items()}
        pool_by_split_train = {m: [pd.concat(x, ignore_index=True) for x in PM] for m,PM in pool_by_split_train.items()}
        
        self.pooled_split = {'test': pool_by_split_test, 'train': pool_by_split_train}
        self.pooled_split_help = "pooled_model['test'/'train'] ['4'/'6,7' etc. (model)] [int (test split)]"

        return
    
    def normed_stat(self, testtrain, key, normkey, testkey=None, thresh=None, pool='model'):
        """
        pool = [model, split]: model means pooled for the entire model, split means pooled for the entire
        if model, output will be indexable by protein ("model"))
        if split, will be indexable by model, and then by test split
        
        """

        if pool=='model':
            data = {}
            for model in self.pooled_model[testtrain]:
                if testkey is None:
                    if thresh is None:
                        idx = np.ones_like(self.pooled_model[testtrain][model][key]).astype(bool) # Force above 1000
                    else: # if testkey is none, but thresh is given, assume thresh applies to normkey
                        idx = self.pooled_model[testtrain][model][normkey]>thresh # Force above 1000
                else:
                    idx = self.pooled_model[testtrain][model][testkey]>thresh # Force above 1000


                data[model] = np.abs(self.pooled_model[testtrain][model][key][idx] / self.pooled_model[testtrain][model][normkey][idx])
                
            
            return data
        
        if pool=='split':
            data = {}
            stds = {}
            means = {}
            for model in self.pooled_split[testtrain]:
                data[model] = {}
                for split in range(len(self.pooled_split[testtrain][model])):
                    
                    if testkey is None:
                        if thresh is None:
                            idx = np.ones_like(self.pooled_split[testtrain][model][split][key]).astype(bool) # Force above 1000
                        else: # if testkey is none, but thresh is given, assume thresh applies to normkey
                            idx = self.pooled_split[testtrain][model][split][normkey]>thresh # Force above 1000
                    else:
                        idx = self.pooled_split[testtrain][model][split][testkey]>thresh # Force above 1000


                    data[model][split] = np.abs(self.pooled_split[testtrain][model][split][key][idx] / self.pooled_split[testtrain][model][split][normkey][idx])
                    
                stds[model] = np.std([np.mean(split) for _,split in data[model].items()])
                means[model] = np.mean([np.mean(split) for _,split in data[model].items()])
                
            
            return data, means, stds
        
        
    def normed_stat_difference(self, testtrain, proteins_to_compare, key, normkey, testkey=None, thresh=None, pool='model'):
        """
        pool = [model, split]: model means pooled for the entire model, split means pooled for the entire
        if model, output will be indexable by protein ("model"))
        if split, will be indexable by model, and then by test split
        
        """
        prot1, prot2 = proteins_to_compare

        if pool=='model':
            data = {}

            data1 = np.abs(self.pooled_model[testtrain][prot1][key] / self.pooled_model[testtrain][prot1][normkey])
            data2 = np.abs(self.pooled_model[testtrain][prot2][key] / self.pooled_model[testtrain][prot2][normkey])
            
            data = data1 - data2
        
            return data
        
        if pool=='split':
            data = {}
            
            for split in range(len(self.pooled_split[testtrain][prot1])): #assume same splits for prot1 and prot2

                data1 = np.abs(self.pooled_split[testtrain][prot1][split][key] / self.pooled_split[testtrain][prot1][split][normkey])
                data2 = np.abs(self.pooled_split[testtrain][prot2][split][key] / self.pooled_split[testtrain][prot2][split][normkey])
                data[split] = data1-data2

            stds = np.std([np.mean(split) for _,split in data.items()])
            means = np.mean([np.mean(split) for _,split in data.items()])
                
            
            return data, means, stds
        
        
        

class EnsemblePredictionDataScatterNoise(EnsemblePredictionData):
    terms_rank1 = ['6_{;a}', '6_{;bba}', '6_{;}6_{;a}', '6_{;}6_{;bba}', '6_{;a}6_{;bb}', '6_{;b}6_{;ba}', 
      '6_{;}6_{;}6_{;a}', '6_{;}6_{;}6_{;bba}', '6_{;}6_{;a}6_{;bb}', '6_{;}6_{;b}6_{;ba}', '6_{;b}6_{;b}6_{;a}']

    terms_rank0 = ['6_{;}', '6_{;}',
     '6_{;aa}',
     '6_{;}6_{;}',
     '6_{;}6_{;aa}',
     '6_{;a}6_{;a}',
     '6_{;}6_{;}6_{;}',
     '6_{;}6_{;}6_{;aa}',
     '6_{;}6_{;a}6_{;a}']

    terms_rank0_tex = ['$\\zeta$', '$\\zeta$',
     '$\\nabla^2\\zeta$',
     '$\\zeta^2$',
     '$\\zeta\\nabla^2\\zeta$',
     '$(\\nabla\\zeta)^2$',
     '$\\zeta^3$',
     '$\\zeta^2\\nabla^2\\zeta$',
     '$\\zeta(\\nabla\\zeta)^2$']
    
    points = {
     '02_cell_1': [[570, 850], [850, 400], [650, 350], [570, 250], [250, 750], [570,850]],
     '02_cell_4': [[250,300], [450,500], [400, 550], [620,650], [800, 700], [800, 550], [760, 350], [550, 250], [400,250], [250,300]],
     '11_cell_1': [[350, 600], [650, 650], [650, 450], [500, 350], [350, 600] ],
     '11_cell_4': [[550, 750], [620, 500], [550, 350], [400, 550], [400, 650], [550, 750] ], 
     '17_cell_0': [[350, 300], [450, 450], [400,600], [700, 600], [700, 450] ],
     '17_cell_1': [[760,200], [500,900], [250, 400], [650, 130] ],
     '17_cell_4': [[700, 200], [660,350], [760,800], [300,650], [500,500], [620,250] ],
     '25_cell_0': [[680,750], [750,600], [450,300], [380,420], [370,650]]
    }
    
    def terms(self, rank=0):
        if rank==0:
            return self.terms_rank0_tex
        
    
    def __init__(self, prot, rootopt=None, test_split_criteria=None):
        
        root='/project/vitelli/cell_stress/ForcePrediction_/XY__paramsweep_2'
#        root='/project/vitelli/cell_stress/ForcePrediction_/XY__paramsweep'
        #testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_211024_1632_XY__paramsweep/test_indices_all.txt'
        testsplits = '/home/schmittms/cell_stress/src_force_prediction/out/out_211026_1453_XY__paramsweep_2/test_indices_all.txt'
        self.model_map={'6': 'Zyxin'}

            
        if rootopt is not None:
            root = os.path.join(os.path.dirname(root), rootopt)
            
            
        self.filename_label=''


        self.root = root   
        subdirs = next(os.walk(os.path.dirname(testsplits)))[1]
        self.test_split_arr = pd.DataFrame(index=subdirs, columns=[0])
        self.test_split_arr[0] = 'none'
        
        
                
        
        self.test_split_selection = None
        self.test_split_criteria = test_split_criteria
        

        args = parse_file_args(os.path.join(os.path.dirname(testsplits), subdirs[0]))
        args = Namespace(**{'crop_size': 960}, **vars(args), **{'lr': 0})
        self.transforms = prediction_transforms(args)
        
        self.get_dirs_all()
        self.load_csv_files()
        self.load_hist_files()
        self.pool_data_by_model()
        self.pool_data_by_split()
        
        self.Nvals = np.unique([int(t.split('-')[-1]) for k in self.dirs_test.keys() for t in k.split('_') if t[0]=='N'])
        self.Ivals = np.unique([float(t.split('-')[-1]) for k in self.dirs_test.keys() for t in k.split('_') if t.split('-')[0]=='intensity'])
         
    def getkey(self, N, I, t):
        return 'type-scatter_N-%s_intensity-%s_testsplit_%s'%(N, I, t)
        
    def get_dirs_all(self):
        self.dirs_test = {}
        self.dirs_train = {}
        for testsplit in natsorted(next(os.walk(self.root))[1]):
            split_idx = testsplit
            if self.test_split_criteria is not None and not np.any([1 if y in x else 0 for x,y in itertools.product(split_idx.split('_'), self.test_split_criteria)]):
                continue

            self.dirs_train[split_idx] = {celldir: os.path.join(self.root, testsplit, celldir) 
                                       for celldir in natsorted(next(os.walk(os.path.join(self.root, testsplit)))[1]) 
                                       if celldir not in list(self.test_split_arr.loc[split_idx])}
            
        self.dirs_test = self.dirs_train

        return


    def get_model_names(self):
        self.model_names = np.unique([i for k,i in self.csv_model_order_train[list(self.csv_model_order_train.keys())[0]].items()])
                
        self.n_models = len(self.model_names)
        self.model_channels = [[s[1:] for s in mn.split('_') if s[0]=='i'][0] for mn in self.model_names]

        self.model_names = {i: name for i, name in zip(self.model_channels, self.model_names)}
        self.models = list(self.model_names.keys())
        print("Unique model order (should only be one)\n", self.model_names)
        return
    
    
    def plot_prediction_vfs(self, testsplitidx, cell, testtrain, frame, return_ims=False, strcrop=2, vmax=2, threshold=0.4, scale=10, figscale=8):

        x,y,m = self.get_sample(testsplitidx, cell, testtrain, frame, strcrop=strcrop)
        
        fig,ax=plt.subplots(1, len(y)+1, figsize=(figscale*(len(y)+1), figscale), constrained_layout=1)

        
        #print(x.shape)
        for n, a in enumerate(ax):
            if n==0:
                a.imshow( np.linalg.norm(x[[2,3]], axis=0), origin='lower', cmap='inferno', vmax=1, vmin=0)
                
                #print(np.max(x[2]), x[2].min())
            else:
                a.imshow( y[n-1][0][0], origin='lower', cmap='inferno', vmax=1, vmin=0)

                #a.imshow( y[n-1][0][0], origin='lower', cmap='inferno', vmax=vmax)

            a.axis('off')
        plt.show()
        if return_ims:
            return x, y, m
        else:
            return
        
    def load_hist_files(self, model=0):
        dirs_test = self.dirs_test
        dirs_train = self.dirs_train
        root = self.root
        
        print('hist_dict%s'%self.filename_label)        
       

        self.hist_files_train = {testsplit: 
                     {celldir: 
                      [np.load(os.path.join(dirs_train[testsplit][celldir], file), allow_pickle=True) for file in natsorted(os.listdir(dirs_train[testsplit][celldir])) 
                          if 'hist_dict%s'%self.filename_label in file][model]
                      for celldir in dirs_train[testsplit]} 
                     for testsplit in dirs_train}
        
        self.hist_files_test = self.hist_files_train
        
        return