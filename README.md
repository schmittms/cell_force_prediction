
# Summary

# Data processing and organization

Datasets are located in a master data directory. Each dataset consists of multiple cells. Each cell has its own folder where every frame of the time lapse is contained as a .npy file.
```
data/
└───TractionData_16kPa_WT/
│   │   cell_force_baselines_bydataset.csv
│   │   cell_force_baselines.csv
│   │   dataset.csv
│   │
│   └───cell_0
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy 
|   |
│   └───cell_1
│   |   │   frame_0.npy
│   |   │   frame_1.npy
│   |   │   ...
│   |   │   frame_T.npy 
|   ...
|
|
└───Dataset2
│   │   cell_force_baselines_bydataset.csv
│   │   cell_force_baselines.csv
│   │   dataset.csv
│   │
│   └───cell_0
│   |   │   frame_0.npy
│   |   │   ...
|  ...
...
```

