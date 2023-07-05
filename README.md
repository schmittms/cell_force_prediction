
# Summary

This repository contains code to train a U-Net to predict forces from protein distributions, as described in https://arxiv.org/abs/2303.00176. 


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

