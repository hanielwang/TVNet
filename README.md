# TVNet: Temporal Voting Network for Action Localization

This repo holds the codes of paper: "TVNet: Temporal Voting Network for Action Localization".

## Paper Introduction

Temporal action localization is a vital task in video understranding. In this paper, we propose a Temporal Voting Network (TVNet) for action localization in untrimmed videos. This incorporates a novel Voting Evidence Module to locate temporal boundaries, more accurately, where temporal contextual evidence is accumulated to predict frame-level probabilities of start and end action boundaries. Our action-independent evidence module is incorporated within a pipeline to calculate confidence scores and action classes. We achieve an average mAP of 34.6% on ActivityNet-1.3, particularly outperforming
previous methods with the highest IoU of 0.95. TVNet also achieves mAP of 56.0% at 0.5 IoU on THUMOS14 and outperforms prior work on all thresholds.


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
Place it into a folder named Thumos_feature_hdf5 inside ./data/thumos_features.


If everything goes well, you can get the folder architecture of ./data like this:

    data                        
    └── thumos_features                    
    	├── Thumos_feature_dim_400              
    	├── Thumos_feature_hdf5               
    	├── features_train.npy 
    	└── features_test.npy

### Feature for ActivityNet 1.3


## Run all steps

Run the following script with all steps on THUMOS14:
```
bash do_all.sh
```

## Run steps separately  

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

| mAP@IoU (%)                    |0.3  | 0.4 | 0.5| 0.6 | 0.7|
|--------------------------------|-----|-----|----|-----|----|
| First run                      | 37.2| 47.4|  |

### ActivityNet 1.3

## Reference

This implementation borrows from:

BSN
BMN
