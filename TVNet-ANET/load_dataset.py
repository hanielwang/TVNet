import random
import numpy as np
import pandas as pd
import json
import options_voting as options
import os

args = options.parser.parse_args()
tscale = 100
tgap = 1. / tscale   

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict():
    """Load dataset file
    """
    df=pd.read_csv("./data/activitynet_annotations/video_info_new.csv")
    json_data= load_json("./data/activitynet_annotations/anet_anno_action.json")
    database=json_data
    train_dict={}
    val_dict={}
    test_dict={}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset=="training":
            train_dict[video_name]=video_new_info
        elif video_subset=="validation":
            val_dict[video_name]=video_new_info
        elif video_subset=="testing":
            test_dict[video_name]=video_new_info
    return train_dict,val_dict,test_dict

def ioa_with_anchors(anchors_min,anchors_max,box_min,box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors=anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

def getBatchList(numVideo,batch_size,shuffle=True):
    """Generate batch list for each epoch randomly
    """
    video_list=range(numVideo)
    batch_start_list=[i*batch_size for i in range(len(video_list)/batch_size)]
    batch_start_list.append(len(video_list)-batch_size)
    if shuffle==True:
        random.shuffle(video_list)
    batch_video_list=[]
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart:(bstart+batch_size)])
    return batch_video_list

def getBatchListTest(video_dict,video_list, batch_size,shuffle=True):
    """Generate batch list during testing
    """
    #video_list=video_dict.keys()
    batch_start_list=[i*batch_size for i in range(len(video_list)/batch_size)]
    batch_start_list.append(len(video_list)-batch_size)
    if shuffle==True:
        random.shuffle(video_list)
    batch_video_list=[]
    for bstart in batch_start_list:
        batch_video_list.append(video_list[bstart:(bstart+batch_size)])
    return batch_video_list

def getBatchData(video_list,data_dict): 
    """Given a video list (batch), get corresponding data
    """
    batch_label_action=[]
    batch_label_start=[]
    batch_label_end=[]
    batch_anchor_feature=[]

    for idx in video_list:
        batch_label_action.append(data_dict["gt_action"][idx])
        batch_label_start.append(data_dict["gt_start"][idx])    
        batch_label_end.append(data_dict["gt_end"][idx]) 
        batch_anchor_feature.append(data_dict["feature"][idx])
        
    batch_label_action=np.array(batch_label_action)
    batch_label_start=np.array(batch_label_start)
    batch_label_end=np.array(batch_label_end)
    batch_anchor_feature=np.array(batch_anchor_feature)
    batch_anchor_feature=np.reshape(batch_anchor_feature,[len(video_list),tscale,-1])
    return batch_label_action,batch_label_start,batch_label_end,batch_anchor_feature
    
def getFullData_windows(dataSet):
    """Load full data in dataset
    """
    train_dict,val_dict,test_dict=getDatasetDict()
    if dataSet=="train":
        video_dict=train_dict
        video_list=video_dict.keys()

    else:
        video_dict=val_dict
        val_list=video_dict.keys()
        video_list=val_list


    batch_bbox=[]
    batch_index=[0]
    batch_anchor_xmin=[]
    batch_anchor_xmax=[]
    batch_anchor_feature=[]
    video_list2 = []
    #for i in range(100):
    for i in range(len(video_list)):
        if i%100==0:
            print "%d / %d %s videos are loaded" %(i,len(video_list),dataSet)
        video_name=video_list[i]
        video_list2.append(video_name)
        #print video_name
        video_info=video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations']
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            batch_bbox.append([tmp_start,tmp_end])
        
        tmp_anchor_xmin=[tgap*i for i in range(tscale)]
        tmp_anchor_xmax=[tgap*i for i in range(1,tscale+1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))    
        batch_anchor_xmax.append(list(tmp_anchor_xmax)) 
        batch_index.append(batch_index[-1]+len(video_labels))
        ######################################################### Load feature for BSN ##########################################
        tmp_df=pd.read_csv("./data/activitynet_feature_cuhk/csv_mean_"+str(tscale)+"/"+video_name+".csv")
        batch_anchor_feature.append(tmp_df.values[:,:])

    num_data=len(batch_anchor_feature)
    batch_label_action=[]
    batch_label_start=[]
    batch_label_end=[]
    
    for idx in range(num_data):
        gt_bbox=np.array(batch_bbox[batch_index[idx]:batch_index[idx+1]])
        #break
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]

        match_score_start = list(gt_xmins)
        match_score_end = list(gt_xmaxs)

        anchor_xmin=batch_anchor_xmin[idx]
        anchor_xmax=batch_anchor_xmax[idx]
        
        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(tgap,0.1*gt_lens)
        
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)
        
        match_score_action=[]
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(np.max(ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))

        batch_label_action.append(match_score_action)
        batch_label_start.append(match_score_start)
        batch_label_end.append(match_score_end)    
    
    dataDict={"gt_action":batch_label_action,"gt_start":batch_label_start,"gt_end":batch_label_end,"feature":batch_anchor_feature}

    return dataDict, video_list2

def getFullData(dataSet):
    """Load full data in dataset
    """
    train_dict,val_dict,test_dict=getDatasetDict()
    if dataSet=="train":
        video_dict=train_dict
    else:
        video_dict=val_dict
    video_list=video_dict.keys()
        
    batch_bbox=[]
    batch_index=[0]
    batch_anchor_xmin=[]
    batch_anchor_xmax=[]
    batch_anchor_feature=[]
    #for i in range(100):
    for i in range(len(video_list)):
        if i%100==0:
            print "%d / %d %s videos are loaded" %(i,len(video_list),dataSet)
        video_name=video_list[i]
        video_info=video_dict[video_name]
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second
        video_labels=video_info['annotations']
        for j in range(len(video_labels)):
            tmp_info=video_labels[j]
            tmp_start=tmp_info['segment'][0]
            tmp_end=tmp_info['segment'][1]
            tmp_start=max(min(1,tmp_start/corrected_second),0)
            tmp_end=max(min(1,tmp_end/corrected_second),0)
            batch_bbox.append([tmp_start,tmp_end])
        
        tmp_anchor_xmin=[tgap*i for i in range(tscale)]
        tmp_anchor_xmax=[tgap*i for i in range(1,tscale+1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))    
        batch_anchor_xmax.append(list(tmp_anchor_xmax)) 
        batch_index.append(batch_index[-1]+len(video_labels))
        tmp_df=pd.read_csv("./data/activitynet_feature_cuhk/csv_mean_"+str(tscale)+"/"+video_name+".csv")
        batch_anchor_feature.append(tmp_df.values[:,:])
    num_data=len(batch_anchor_feature)
    batch_label_action=[]
    batch_label_start=[]
    batch_label_end=[]
    
    for idx in range(num_data):
        gt_bbox=np.array(batch_bbox[batch_index[idx]:batch_index[idx+1]])
        #break
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]
        anchor_xmin=batch_anchor_xmin[idx]
        anchor_xmax=batch_anchor_xmax[idx]
        
        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(tgap,0.1*gt_lens)
        
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1)
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1)
        
        match_score_action=[]
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(np.max(ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))
        match_score_start=[]
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_start_bboxs[:,0],gt_start_bboxs[:,1])))
        match_score_end=[]
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_end_bboxs[:,0],gt_end_bboxs[:,1])))
    
        batch_label_action.append(match_score_action)
        batch_label_start.append(match_score_start)
        batch_label_end.append(match_score_end)    
    
    dataDict={"gt_action":batch_label_action,"gt_start":batch_label_start,"gt_end":batch_label_end,"feature":batch_anchor_feature}
    return dataDict,video_list

  
def getProposalDataTest(video_list,video_dict):
    """Load data during testing
    """
    batch_anchor_xmin=[]
    batch_anchor_xmax=[]
    batch_anchor_feature=[]
    for i in range(len(video_list)):
        video_name=video_list[i]
        #print video_name
        tmp_anchor_xmin=[tgap*i for i in range(tscale)]
        tmp_anchor_xmax=[tgap*i for i in range(1,tscale+1)]
        batch_anchor_xmin.append(list(tmp_anchor_xmin))    
        batch_anchor_xmax.append(list(tmp_anchor_xmax)) 
        tmp_df=pd.read_csv("./data/activitynet_feature_cuhk/csv_mean_"+str(tscale)+"/"+video_name+".csv")
        batch_anchor_feature.append(tmp_df.values[:,:])
    batch_anchor_xmin=np.array(batch_anchor_xmin)
    batch_anchor_xmax=np.array(batch_anchor_xmax)
    batch_anchor_feature=np.array(batch_anchor_feature)
    batch_anchor_feature=np.reshape(batch_anchor_feature,[len(video_list),tscale,-1])
    return batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature


class Dataset():
    def __init__(self, args):

        self.currenttestidx = 0

        self.val_data_dict, self.val_list = getFullData("val")
        self.val_features = self.val_data_dict['feature']


    def load_data_slide_window(self, batch_size = 0, is_training=True):

        if is_training==True:
            features = []
            gtlabels = []
            labels = []
            gtsegments = []
            idx = []

            #################################################### BSN ########################################################
            features_window = np.load('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)
            attention_window = np.load('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)
            r_gt_window  = np.load('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'.npy', allow_pickle=True)

            window_list = range(0,len(features_window))
          
            idxx = random.sample(window_list, batch_size)

            return np.array([features_window[i] for i in idxx]), np.array([r_gt_window[i] for i in idxx]), np.array([attention_window[i] for i in idxx]), idxx#, np.array([regression[i] for i in idxx])

        else: #For Testing     

            test_list = range(0,len(self.val_list))
            idx_save = self.val_list[self.currenttestidx]
            feat = self.val_features[test_list[self.currenttestidx]]
            if args.window_length == 15:
                tdf= pd.read_csv("./outputs/VEM_Train/TEM_action_L15/"+idx_save+".csv")
                attention = tdf.action_sig.values[:]
            else:
                tdf= pd.read_csv("./outputs/VEM_Train/TEM_action_L5/"+idx_save+".csv")
                attention = tdf.action.values[:]

            idx_now = test_list[self.currenttestidx]
            
            if self.currenttestidx == len(test_list)-1:
                done = True; self.currenttestidx = 0
            else:
                done = False; self.currenttestidx += 1
         
            return np.array(feat), idx_save, attention, done


    def load_data_slide_window_val(self, batch_size = 0, is_training=True):

        if is_training==True:
            features = []
            gtlabels = []
            labels = []
            gtsegments = []
            idx = []

            #################################################### BSN ########################################################
            features_window = np.load('./outputs/VEM_Train/features_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'_val.npy', allow_pickle=True)
            attention_window = np.load('./outputs/VEM_Train/attention_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'_val.npy', allow_pickle=True)
            r_gt_window  = np.load('./outputs/VEM_Train/r_window'+str(args.window_length)+'stride'+str(args.window_stride)+'_'+str(args.voting_type)+'_val.npy', allow_pickle=True)

            window_list = range(0,len(features_window))
          
            idxx = random.sample(window_list, batch_size)

            return np.array([features_window[i] for i in idxx]), np.array([r_gt_window[i] for i in idxx]), np.array([attention_window[i] for i in idxx]), idxx#, np.array([regression[i] for i in idxx])


        else: #For Testing     
         
            return 0


