# TVNet: Temporal Voting Network for Action Localization

This repo holds the codes of paper: "TVNet: Temporal Voting Network for Action Localization".

## Paper Introduction

Temporal action localization is a vital task in video understranding. In this paper, we propose a Temporal Voting Network (TVNet) for action localization in untrimmed videos. This incorporates a novel Voting Evidence Module to locate temporal boundaries, more accurately, where temporal contextual evidence is accumulated to predict frame-level probabilities of start and end action boundaries.


## Dependencies

* Python == 2.7
* Tensorflow == 1.9.0
* CUDA==10.1.105
* GCC >= 5.4

Note that the PEM code from BMN is implemented in Pytorch==1.1.0 or 1.3.0

## Data Preparation

### Datasets

Our experiments is based on ActivityNet 1.3 and THUMOS14 datasets. 

### Feature for THUMOS14

You can download the feature on THUMOS14 at here [GooogleDrive](https://drive.google.com/file/d/18fm9xzfnLnkDEIsNThgRMtconGVyxHd3/view?usp=sharing).

Place it into a folder named thumos_features inside ./data.

You also need to download the feature for PEM (from BMN) at [GooogleDrive](https://drive.google.com/drive/folders/10PGPMJ9JaTZ18uakPgl58nu7yuKo8M_k?usp=sharing).
Please put it into a folder named Thumos_feature_hdf5 inside ./TVNet-THUMOS14/data/thumos_features.

If everything goes well, you can get the folder architecture of ./TVNet-THUMOS14/data like this:

    data                        
    └── thumos_features                    
    		├── Thumos_feature_dim_400              
    		├── Thumos_feature_hdf5               
    		├── features_train.npy 
    		└── features_test.npy

### Feature for ActivityNet 1.3
You can download the feature on ActivityNet 1.3 at here [GoogleCloud](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing).
Please put csv_mean_100 directory into ./TVNet-ANET/data/activitynet_feature_cuhk/.

If everything goes well, you can get the folder architecture of ./TVNet-ANET/data like this:

    data                        
    └── activitynet_feature_cuhk                    
    		    └── csv_mean_100

## Run all steps
### Run all steps on THUMOS14
```
cd TVNet-THUMOS14
```
Run the following script with all steps on THUMOS14:
```
bash do_all.sh
```

Note: If you use BlueCrystal 4, you can directly run the following script without any dependencies setup.
```
bash do_all_BC4.sh
```

### Run all steps on ActivityNet 1.3
```
cd TVNet-ANET
bash do_all.sh  or  bash do_all_BC4.sh
```


## Run steps separately
Take TVNet-THUMOS14 as an example:
```
cd TVNet-THUMOS14
```
#### 1. Temporal evaluation module

```
python TEM_train.py
```

```
python TEM_test.py
```

#### 2. Creat training data for voting evidence module


```
python VEM_create_windows.py --window_length L --window_stride S
```
L is the window length and S is the sliding stride. We generate training windows for length 10 with stride 5, and length 5 with stride 2.


#### 3. Voting evidence module

```
python VEM_train.py --voting_type TYPE --window_length L --window_stride S
```

```
python VEM_test.py --voting_type TYPE --window_length L --window_stride S
```
TYPE should be start or end. We train and test models with window length 10 (stride 5) and window length 5 (stride 2) for start and end separately.


#### 4. Proposal evaluation module from BMN

```
python PEM_train.py
```

#### 5. Proposal generation

```
python proposal_generation.py
```


#### 6. Post processing and detection

```
python post_postprocess.py
```

## Results
### THUMOS14


| tIoU|mAP@IoU (%) | 
|------------------|-----|
| 0.3 | 0.5724681814413137|
|0.4 | 0.5060844218403346|
|0.5 | 0.430414918823808|
|0.6 | 0.3297164845828022|
|0.7 | 0.202971546242546|


### ActivityNet 1.3

| tIoU  |mAP@IoU (%) | 
|--------|-----|
| Average | 0.3460396513933088|
|0.5 | 0.5135151163296395|
|0.75 | 0.34955648726767025|
|0.95 | 0.10121803584836778|


## Reference

This implementation borrows from:

[BSN](https://github.com/wzmsltw/BSN-boundary-sensitive-network): BSN-boundary-sensitive-network

[BMN](https://github.com/JJBOY/BMN-Boundary-Matching-Network): BMN-Boundary-Matching-Network

[G-TAD](https://github.com/frostinassiky/gtad): G-TAD: Sub-Graph Localization for Temporal Action Detection