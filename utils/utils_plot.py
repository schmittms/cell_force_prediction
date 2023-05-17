import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
import skimage.measure as measure


texttop = {'x': 0.5, 'y': 1.02, 'ha': 'center', 'va':'bottom'}
texttopright = {'x': 0.98, 'y': 0.98, 'ha': 'right', 'va':'top'}
texttopleft = {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va':'top'}
textleft = {'x': -0.05, 'y': 0.5, 'ha': 'right', 'va':'center', 'rotation': 90}


lognorm=mplcolors.SymLogNorm # When used, should take args linthresh=.., vmin=.., vmax=..

class PositiveNorm(object):
    def __init__(self, vmax, cmap='inferno'):
        self.vmax = vmax
        self.cmap = cmap


    def __call__(self, imagestack, idx):
        assert len(imagestack.shape)==4, "Image stack shape (%s)  not [B, C, H, W]"%str(imagestack.shape)

        return {'vmax': self.vmax, 'vmin': 0, 'cmap': self.cmap}

class SymmetricNorm(object):
    def __init__(self, vmax, cmap='bwr'):
        self.vmax = vmax
        self.cmap = cmap


    def __call__(self, imagestack, idx, channel):
        assert len(imagestack.shape)==4, "Image stack shape (%s)  not [B, C, H, W]"%str(imagestack.shape)

        if self.vmax=='individual':
            vmax = np.abs(imagestack[idx, channel]).max()
        elif self.vmax=='all':
            vmax = np.abs(imagestack[:, channel]).max()

        return {'vmax': vmax, 'vmin': -vmax, 'cmap': self.cmap}


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


