# -*- coding: utf-8 -*-
"""
This TEM code is based on BSN.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import load_dataset as TEM_load_data

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def tem_inference(X_feature,config):
    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net1=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net1=tf.layers.conv1d(inputs=net1,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net2=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net2=tf.layers.conv1d(inputs=net2,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net3=net1+net2

    net=0.1*tf.layers.conv1d(inputs=net3,filters=3,kernel_size=1,strides=1,padding='same')
    scores=tf.nn.sigmoid(net)

    TEM_trainable_variables=tf.trainable_variables()
    return scores,TEM_trainable_variables
    

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        #common information
        self.learning_rates=[0.001]*5 + [0.0001]*15
        self.training_epochs = len(self.learning_rates)
        self.n_inputs = 400
        self.negative_ratio=1
        self.batch_size = 16
        self.num_prop=100


if __name__ == "__main__":
    config = Config()
    
    X_feature = tf.placeholder(tf.float32, shape=(None,config.num_prop,config.n_inputs))
    Y_bbox=tf.placeholder(tf.float32,[None,2])
    Index=tf.placeholder(tf.int32,[config.batch_size+1])
    LR= tf.placeholder(tf.float32)

    scores,TEM_trainable_variables=tem_inference(X_feature,config)
    
    model_saver=tf.train.Saver(var_list=TEM_trainable_variables,max_to_keep=80)
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.global_variables_initializer().run()  
    model_saver.restore(sess,"models/TEM/tem_model_epoch-3")  
    feat_ratio = load_json("./data/features_ratio.json")
    annoDf_train=pd.read_csv("./data/thumos_annotations/val_Annotation.csv")
    annoDf_test=pd.read_csv("./data/thumos_annotations/test_Annotation.csv")
    videoNameList=list(set(annoDf_train.video.values[:])) + list(set(annoDf_test.video.values[:]))
    save_path = "./outputs/TEM_Test/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path2 = "./outputs/VEM_Train/TEM_action/"
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)    
    columns=["frame","action","start","end"]

    for videoName in videoNameList:
        list_snippets,list_data,video_snippet=TEM_load_data.getVideoData(videoName)
        out_scores=sess.run(scores,feed_dict={X_feature:list_data})  
        calc_time_list=np.zeros(len(video_snippet))
        snippet_scores=np.zeros([len(video_snippet),3])
        
        for idx in range(len(list_snippets)):
            snippets=list_snippets[idx]
            for jdx in range(len(snippets)):
                tmp_snippet=snippets[jdx]
                tmp_snippet_index=video_snippet.index(tmp_snippet)
                calc_time_list[tmp_snippet_index]+=1
                snippet_scores[tmp_snippet_index,:]+=out_scores[idx,jdx,:]

        calc_time_list=np.stack([calc_time_list,calc_time_list,calc_time_list],axis=1)
        snippet_scores=snippet_scores/calc_time_list   
        
        snippet_scores=np.concatenate((np.reshape(video_snippet,[-1,1]),snippet_scores),axis=1)
        tmp_df=pd.DataFrame(snippet_scores,columns=columns)  
        tmp_df.to_csv(save_path+videoName+".csv",index=False)

        ############################# rescale action curves for VEM training ##############################

        if videoName[6:10] == 'test':

            ratio_vid = feat_ratio[videoName][0]
            length = feat_ratio[videoName][1]

            x_new = [int(ratio_vid*n) for n in list(range(length))]

            action_df = pd.read_csv(save_path+videoName+".csv")
            #action = action_df.action.values[:]
            action_new = action_df.action.values[:][x_new]
            #action_new = 1/(1 + np.exp(-4*action_df.action.values[:][x_new]))

            columns2=["action"]
            tmp_df_a=pd.DataFrame(action_new,columns=columns2)
            tmp_df_a.to_csv(save_path2+str(videoName)+".csv",index=False)



        
