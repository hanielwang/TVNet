import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import time
import random
import scipy.io as sio
#from video_dataset_attention import Dataset
import options_voting as options
import math
import load_dataset as load_data
#from video_dataset_attention import get_segment_gt as get_segment_gt
args = options.parser.parse_args()


#train_data_dict, video_list=video_load_data_anet_BSN_CNN.getFullData("val")
#print type(train_data_dict)  #key:gt_start,gt_action,gt_end,feature
#video_list_idx = range(0,len(video_list))
#batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature=video_load_data_anet_BSN_CNN.getBatchData(video_list_idx,train_data_dict)

if os.path.exists('./outputs/VEM_Train/gt_start.npy'):
  print ('gt data already exists')
  features = list(np.load('./outputs/VEM_Train/features.npy', encoding='bytes', allow_pickle=True))
  start_gt = list(np.load('./outputs/VEM_Train/gt_start.npy', allow_pickle=True))
  end_gt = list(np.load('./outputs/VEM_Train/gt_end.npy', allow_pickle=True))
  attention_gt = list(np.load('./outputs/VEM_Train/gt_action.npy', allow_pickle=True))

else:
  print ('creat gt data...')
  train_data_dict, video_list=load_data.getFullData_windows("train")
  start_gt = train_data_dict['gt_start']
  end_gt = train_data_dict['gt_end']
  attention_gt = train_data_dict['gt_action']
  features = train_data_dict['feature']
  #print attention_gt.shape

  np.save('./outputs/VEM_Train/gt_start.npy',start_gt)
  np.save('./outputs/VEM_Train/gt_end.npy',end_gt)
  np.save('./outputs/VEM_Train/gt_action.npy',attention_gt)
  np.save('./outputs/VEM_Train/features.npy',features)


def rolling_window(a, length, stride):
    nrows = ((a.size-length)//stride)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,length), strides=(stride*n,n))


def generate_windows(video_list, generating_gt, is_starting = True):

    feature_all_all=[]
    attention_all_all=[]
    r_gt_all_all=[]

    feature_all_all_bg=[]
    attention_all_all_bg=[]
    r_gt_all_all_bg=[]

    for x in range(0,9649):

      if len(features[x]) <= args.window_length:
        continue

      if x in video_list:
          all_seq = np.arange(len(features[x]))


          slide_windows=rolling_window(all_seq,args.window_length,args.window_stride)

          point = []
          if len(generating_gt[x]):
            point = generating_gt[x]
          point = [int(len(features[x])*point[i]) for i in range(len(point))]

          for s in point:
                for row in slide_windows:            
                    for k in row:                 
                        if k==s:

                            features_window = np.array([features[x][i]for i in row])
                            feature_all_all.append(features_window)

                            attention_window = np.array([attention_gt[x][i]for i in row])
                            attention_all_all.append(attention_window)
                         
                            r_gt = []

                            for i in row:
                              if is_starting == True:
                                r = i-s
                              else:
                                r = s-i
                              r_gt.append(r)
                              
                            r_gt_all_all.append(r_gt)



    

    return feature_all_all,r_gt_all_all,attention_all_all

train_list = range(0,9649)

features_window_all_train_s,r_gt_all_reshape_train_s,at_s= generate_windows(train_list,start_gt, is_starting=True)
print ('The totol number of windows for start training is %d' %(len(features_window_all_train_s)))
features_window_all_train_e,r_gt_all_reshape_train_e,at_e= generate_windows(train_list,end_gt, is_starting=False)
print ('The totol number of windows for end training is %d' %(len(features_window_all_train_e)))

np.save('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',features_window_all_train_s)
np.save('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',r_gt_all_reshape_train_s)
np.save('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_start.npy',at_s)


np.save('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',features_window_all_train_e)
np.save('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',r_gt_all_reshape_train_e)
np.save('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_end.npy',at_e)
