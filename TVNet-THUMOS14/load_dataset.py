import numpy as np
import glob
import pandas as pd
import time
import math
import options_voting as options
import copy
import random

import json, pickle
import torch.utils.data as data
import torch
import os
import h5py

args = options.parser.parse_args()

def iou_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len +box_max-box_min
    #print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard
    
def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

def getBatchList(numWindow,batch_size,shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    window_list=range(numWindow)
    batch_start_list=[i*batch_size for i in range(len(window_list)/batch_size)]
    batch_start_list.append(len(window_list)-batch_size)
    if shuffle==True:
        random.shuffle(window_list)
    batch_window_list=[]
    for bstart in batch_start_list:
        batch_window_list.append(window_list[bstart:(bstart+batch_size)])
    return batch_window_list


def getBatchData(window_list,data_dict):
    batch_bbox=[]
    batch_index=[0]
    batch_anchor_xmin=[]
    batch_anchor_xmax=[]
    batch_anchor_feature=[]
    for idx in window_list:
        batch_bbox.extend(data_dict["gt_bbox"][idx])
        batch_index.append(batch_index[-1]+len(data_dict["gt_bbox"][idx]))
        batch_anchor_xmin.append(data_dict["anchor_xmin"][idx])    
        batch_anchor_xmax.append(data_dict["anchor_xmax"][idx]) 
        batch_anchor_feature.append(data_dict["feature"][idx])
    batch_index=np.array(batch_index)
    batch_bbox=np.array(batch_bbox)
    batch_anchor_xmin=np.array(batch_anchor_xmin)
    batch_anchor_xmax=np.array(batch_anchor_xmax)
    batch_anchor_feature=np.array(batch_anchor_feature)
    #batch_anchor_feature=np.reshape(batch_anchor_feature,[len(video_list),100,-1])
    return batch_index,batch_bbox,batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature


def getFullData(dataSet):
    ii=0
    #dataSet="Test"
    annoDf=pd.read_csv("./data/thumos_annotations/"+dataSet+"_Annotation.csv")
    videoNameList=list(set(annoDf.video.values[:]))
    input_spatial_path="./data/thumos_features/Thumos_feature_dim_400/rgb/"
    input_temporal_path="./data/thumos_features/Thumos_feature_dim_400/flow/"
    
    list_data=[]
    list_anchor_xmins=[]
    list_anchor_xmaxs=[]
    list_gt_bbox=[]
    
    for videoName in videoNameList:
        #print ii
        ii+=1
        video_annoDf=annoDf[annoDf.video==videoName] 
        gt_xmins=video_annoDf.startFrame.values[:]
        gt_xmaxs=video_annoDf.endFrame.values[:]
        
        spatialDf=pd.read_csv(input_spatial_path+videoName+".csv")
        temporalDf=pd.read_csv(input_temporal_path+videoName+".csv")
        
        numSnippet=min(len(spatialDf),len(temporalDf))
        frameList=[3+5*i for i in range(numSnippet)]
        df_data=np.concatenate((spatialDf.values[:numSnippet,:],temporalDf.values[:numSnippet,:]),axis=1)
        df_snippet=frameList
        window_size=100
        stride=window_size/2
        n_window=(numSnippet+stride-window_size)/stride
        windows_start=[i*stride for i in range(n_window)]
        #print np.shape(df_data)
        if numSnippet<window_size:
            windows_start=[0]
            tmp_data=np.zeros((window_size-numSnippet,400))
            df_data=np.concatenate((df_data,tmp_data),axis=0)
            df_snippet.extend([df_snippet[-1]+5*(i+1) for i in range(window_size-numSnippet)])
        elif numSnippet-windows_start[-1]-window_size>20:
            windows_start.append(numSnippet-window_size)
            
        for start in windows_start:
            tmp_data=df_data[start:start+window_size,:]
            tmp_snippets=np.array(df_snippet[start:start+window_size])
            tmp_anchor_xmins=tmp_snippets-2.5
            tmp_anchor_xmaxs=tmp_snippets+2.5
            tmp_gt_bbox=[]
            tmp_ioa_list=[]
            for idx in range(len(gt_xmins)):
                tmp_ioa=ioa_with_anchors(gt_xmins[idx],gt_xmaxs[idx],tmp_anchor_xmins[0],tmp_anchor_xmaxs[-1])
                tmp_ioa_list.append(tmp_ioa)
                if tmp_ioa>0:
                    tmp_gt_bbox.append([gt_xmins[idx],gt_xmaxs[idx]])
            #print tmp_ioa_list
            if len(tmp_gt_bbox)>0 and max(tmp_ioa_list)>0.9:
                list_gt_bbox.append(tmp_gt_bbox)
                list_anchor_xmins.append(tmp_anchor_xmins)
                list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                list_data.append(tmp_data)
        # print('------------------------------------')
        # print gt_xmins
        # print gt_xmaxs
        # print tmp_gt_bbox
        # print df_data.shape
    dataDict={"gt_bbox":list_gt_bbox,"anchor_xmin":list_anchor_xmins,"anchor_xmax":list_anchor_xmaxs,"feature":list_data}
    return dataDict


def getVideoData(videoName):

    input_spatial_path="./data/thumos_features/Thumos_feature_dim_400/rgb/"
    input_temporal_path="./data/thumos_features/Thumos_feature_dim_400/flow/"
    
    list_data=[]
    list_snippets=[]
    spatialDf=pd.read_csv(input_spatial_path+videoName+".csv")
    temporalDf=pd.read_csv(input_temporal_path+videoName+".csv")
    
    numSnippet=min(len(spatialDf),len(temporalDf))
    frameList=[3+5*i for i in range(numSnippet)]
    df_data=np.concatenate((spatialDf.values[:numSnippet,:],temporalDf.values[:numSnippet,:]),axis=1)
    df_snippet=frameList
    window_size=100
    stride=window_size/2
    n_window=(numSnippet+stride-window_size)/stride
    windows_start=[i*stride for i in range(n_window)]
    
    if numSnippet<window_size:
        windows_start=[0]
        tmp_data=np.zeros((window_size-numSnippet,400))
        df_data=np.concatenate((df_data,tmp_data),axis=0)
        df_snippet.extend([df_snippet[-1]+5*(i+1) for i in range(window_size-numSnippet)])
    else:
        windows_start.append(numSnippet-window_size)
        
    for start in windows_start:
        tmp_data=df_data[start:start+window_size,:]
        tmp_snippets=np.array(df_snippet[start:start+window_size])
        list_data.append(tmp_data)
        list_snippets.append(tmp_snippets)

    list_snippets=np.array(list_snippets)
    list_data=np.array(list_data)
    return list_snippets,list_data,df_snippet



class Dataset():
    def __init__(self, args):

        self.features_test = np.load('./data/thumos_features/features_test.npy')
        self.test_list = np.load('./data/video_list_test.npy')
        self.currenttestidx = 0

    def load_data_slide_window(self, batch_size = 0, is_training=True):

        if is_training==True:

            features_window = np.load('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)
            attention_window = np.load('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)
            r_gt_window  = np.load('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)

            window_list = range(0,len(features_window))
            idx = random.sample(window_list, batch_size)

            return np.array([features_window[i] for i in idx]), np.array([attention_window[i] for i in idx]), np.array([r_gt_window[i] for i in idx]),idx       

        else: #For Testing

            feat = self.features_test[self.currenttestidx]
            video_name = self.test_list[self.currenttestidx]
            attention_df = pd.read_csv("./outputs/VEM_Train/TEM_action/"+video_name+".csv")
            attention = attention_df.action.values[:]

            #print (video_name)

            if self.currenttestidx == len(self.test_list)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return video_name,np.array(feat),attention,done    

