
# Summary

This repository contains code to train a U-Net to predict forces from protein distributions as described in [Machine learning interpretable models of cell mechanics from protein images](https://www.cell.com/cell/fulltext/S0092-8674(23)01331-4) ([arxiv](https://arxiv.org/abs/2303.00176)).

The other two repositories for this paper are:
 - Physical bottleneck analysis: https://github.com/schmittms/physical_bottleneck
 - Green's function neural network: https://github.com/jcolen/gfnn

This code was used to train the networks Figures 1-4. The trained model weights can be downloaded [here](https://uchicago.box.com/s/o6gpcdaxzv9t6ffus81o6we2226k3jc9), and in [load_trained_unet.ipynb](load_trained_unet.ipynb) we demonstrate how to load them. The raw data used for training can be downloaded from this [link](https://uchicago.box.com/s/s0poevx1iaa8f6iywv59uftbimjuoss1). Data pre-processing is described in the [DataProcessing.ipynb](DataProcessing.ipynb) notebook. This repository also contains a minimal working example notebook [train_unet.ipynb](train_unet.ipynb) which trains a U-Net on a small amount of data. This example dataset can be downloaded [here](https://uchicago.box.com/s/axbn54r31amvrnfck82hjmz01qsvd0ox). This notebook is for illustration purposes only; to train a network to the accuracy of those used in the paper, more data and longer training is needed.


## Data organization

All datasets are located in a master data directory. Each dataset consists of multiple cells. Each cell has its own folder where every frame of the time lapse is contained as a .npy file. This .npy file has shape `[C, L, L]` where `L` is the image size (images are square, typically `L=992` or `1120`), and there are `C=7 or 8` channels. The channels correspond to `[u_x, u_y, F_x, F_y, mask, forcemask, zyxin, actin]` where `u_x` is the displacement in the `x` direction.

```
data/
└───TractionData_16kPa/
│   │   cell_force_baselines_bydataset.csv
│   │   cell_force_baselines.csv
│   │   dataset.csv
│   │
│   └───dataset_A_cell_0
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy 
|   |
│   └───dataset_A_cell_1
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy
|   |
│   └───dataset_B_cell_0
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy 
|   |
│   └───dataset_B_cell_1
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy 
|  ...
...
```

