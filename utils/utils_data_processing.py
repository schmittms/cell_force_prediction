import os
import re
import torch
import numpy as np
import pandas as pd
import random

import scipy.ndimage
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import filters
from skimage import morphology
from time import time

import warnings
warnings.filterwarnings("ignore")

from utils.utils_data_processing_base import CellCrop, SmoothForces, RandomBlur, RandomRescale, AddNoise, Magnitude, Threshold, AngleMag, ToTensor
from utils.utils_data_processing_base import CellDataset as CDataset


'''
***************************************************

Datasets

***************************************************
'''

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

channel_to_protein_dict = {
                            '4': 'mask',
                            '6': 'zyxin',
                            '7': 'actin',
                            '4,6': 'mask,zyxin',
                            '4,7': 'mask,actin',
                            '6,7': 'zyxin,actin',
                            '4,6,7': 'mask,zyxin,actin',
                            }
    

def args_to_transform_kwargs(norm_output=None, perturb_input='', perturb_output='', add_noise=''):
    n_o = {s.split(',')[0]: float( s.split(',')[1] ) for s in norm_output.split('/')} if norm_output is not None else None 
    #p_i = {s.split(',')[0]: float( s.split(',')[1] ) for s in perturb_input.split('/')}  if perturb_input is not None else None 
    p_i = perturb_input.split(',')
    p_o = perturb_output.split(',')
    try:
        noise = {s.split(',')[0]: float( s.split(',')[1] ) for s in add_noise.split('/')} 
    except: 
        print("Could not add noise to transform kwargs.")
        noise = {}
    return {'norm_output': n_o, 'perturb_input': p_i, 'perturb_output': p_o, 'add_noise': noise}





class CellDataset(CDataset):
    def __getitem__(self, idx, mask_crop=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = np.load(os.path.join(self.info.root[idx], self.info.folder[idx], self.info.filename[idx]))
        #image = image[:7]
        #image = self.mask_crop(image)
        if mask_crop: image = self.mask_crop(image)


        in_unique = np.unique([int(x) for ch in self.in_channels for x in ch])
        if 4 in in_unique: image[4] /= np.max(image[4])
        if 6 in in_unique: image[6] = self.rm_baseline_zyx(image[6], idx) # to center around 0
        if 7 in in_unique: image[7] = self.rm_baseline_zyx(image[7], idx) # to center around 0

        image[self.out_channels, :, :] = self.normalize_force(image[self.out_channels, :, :], idx) # to center around 0

        image = self.transform(image)

        mask = image[4].unsqueeze(0)
        zyxin= image[6].unsqueeze(0)
        output= image[self.out_channels, :, :]

  
        #return {'mask': mask, 'zyxin': zyxin, 'output': output, 'displacements': image[[0,1]]}

        if 7 in in_unique: return {'mask': image[4].unsqueeze(0), 'zyxin': image[6].unsqueeze(0), 'actin': image[7].unsqueeze(0), 'output': image[self.out_channels, :, :], 'displacements': image[[0,1]]}
        else:              return {'mask': image[4].unsqueeze(0), 'zyxin': image[6].unsqueeze(0), 'output': image[self.out_channels, :, :], 'displacements': image[[0,1]]}
    
    
    

             
"""-----------------------------------------------------------

Transform

-----------------------------------------------------------"""
   
class RandomRotate(object):
    def __init__(self, vector_components):
        self.vec_chs = vector_components # List of lists
        
    def __call__(self, image):
        angle = np.random.uniform()*360
        image = scipy.ndimage.rotate(image, angle, axes=(-1,-2), reshape=False)
        angle *= np.pi/180
        
        for vc in self.vec_chs:
            "vc = [vx, vy] for each vector (pair of components) in the image"
            image[vc[1]], image[vc[0]] = image[vc[1]]*np.cos(angle) - image[vc[0]]*np.sin(angle), \
                                                image[vc[1]]*np.sin(angle) + image[vc[0]]*np.cos(angle)
        return image


class RandomTranspose(object):
    def __init__(self, vector_components, prob=0.5):
        self.prob = prob
        self.vector_components = vector_components # was [2, 3]. Now is list of tuples [(0,1), (2,3) etc.] 

    def __call__(self, image):
        if np.random.random() < self.prob:
            image = np.swapaxes(image, -1, -2)
            
            for vec in self.vector_components:
                image[vec] = image[vec[::-1]]
        return image 




