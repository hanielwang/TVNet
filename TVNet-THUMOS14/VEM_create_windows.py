import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import time
import random
import scipy.io as sio
import pandas as pd
import options_voting as options
import math

args = options.parser.parse_args()

feature = np.load("./data/thumos_features/features_train.npy")
attention = np.load("./data/action_gt_train.npy")
thumos_test_anno = pd.read_csv("./data/thumos_annotations/val_Annotation.csv")
video_list = thumos_test_anno.video.unique()

starting_gt = [[]for n in range(200)]
ending_gt = [[]for n in range(200)]

starting_gt_final = []
ending_gt_final = []

for v,vid in  enumerate(video_list):

  s = thumos_test_anno[thumos_test_anno['video']==vid]['start'].values
  s = [int(round(ss*1.5625)) for ss in s]

  e = thumos_test_anno[thumos_test_anno['video']==vid]['end'].values
  e = [int(round(ee*1.5625)) for ee in e]

  starting_gt[v] = s
  ending_gt[v] = e

def rolling_window(a, length, stride):
    nrows = ((a.size-length)//stride)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,length), strides=(stride*n,n))

def generate_windows(video_list, generating_gt, is_starting = True):

    feature_windows= []
    attention_windows= []
    r_gt_windows = []

    for v,vid in  enumerate(video_list):
      feature_cls=[]
      attention_cls=[]
      r_gt_cls=[]

      all_seq = np.arange(len(feature[v]))
      slide_windows=rolling_window(all_seq,args.window_length,args.window_stride)

      point = generating_gt[v]
            
      for s in point:
        for row in slide_windows:
          for k in row:
            if k==s:
              features_window = np.array([feature[v][i]for i in row])
              feature_windows.append(features_window)

              attention_window = np.array([attention[v][i]for i in row])
              attention_windows.append(attention_window)
                            
              r_gt = []
              for i in row:
                if is_starting == True:
                  r = i-s
                else:
                  r = s-i
                r_gt.append(r)

              r_gt_windows.append(r_gt)

    return feature_windows,r_gt_windows,attention_windows

print("Training windows generation start...")

features_window_s,r_gt_s,attention_window_s = generate_windows(video_list, starting_gt, is_starting=True)

features_window_e,r_gt_e,attention_window_e = generate_windows(video_list, ending_gt, is_starting=False)


#############################################################################
np.save('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',features_window_s)
np.save('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',r_gt_s)
np.save('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',attention_window_s)
 
np.save('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',features_window_e)
np.save('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',r_gt_e)
np.save('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',attention_window_e)   

print("Training windows generation finished")
