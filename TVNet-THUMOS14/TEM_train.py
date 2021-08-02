# -*- coding: utf-8 -*-
"""
This TEM code is based on BSN.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import load_dataset as TEM_load_data
import matplotlib.pyplot as plt

    
def intersection_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max):
    """Compute intersection between score a ground truth box and the anchors.
    """
    int_xmin = tf.maximum(anchors_min, box_min)
    int_xmax = tf.minimum(anchors_max, box_max)
    inter_len = tf.maximum(int_xmax - int_xmin, 0.)
    scores = tf.div(inter_len, len_anchors)
    return scores

def loop_condition(idx,
              b_anchors_xmin,b_anchors_xmax,
              b_gbboxes,b_match_scores):   
    """Loop condition of bboxes encode.
    """    
    r = tf.less(idx,tf.shape(b_gbboxes))
    return r[0]
    

def loop_body(idx,
              b_anchors_xmin,b_anchors_xmax,
              b_gbboxes,b_match_scores):
    """Loop body of bboxes encode.
    """    
    box_min = b_gbboxes[idx,0]
    box_max = b_gbboxes[idx,1]

    len_anchors = b_anchors_xmax-b_anchors_xmin
    overlap = intersection_with_anchors(b_anchors_xmin,b_anchors_xmax,len_anchors,box_min,box_max)
    b_match_scores = tf.maximum(overlap,b_match_scores)
    return [idx+1,b_anchors_xmin,b_anchors_xmax,
              b_gbboxes,b_match_scores]


def tem_bboxes_encode(anchors_xmin,anchors_xmax,
                      gbboxes,gIndex,
                      config):
    """Calculate overlap between anchors and ground truth.
    """   
    num_prop = config.num_prop
    batch_size = config.batch_size
    dtype = tf.float32
    batch_match_scores = tf.reshape(tf.constant([]),[-1,num_prop])
    
    for i in range(batch_size):

        shape=(num_prop)
        match_scores=tf.zeros(shape,dtype)
        b_anchors_xmin = anchors_xmin[i]
        b_anchors_xmax = anchors_xmax[i]
        b_gbboxes = gbboxes[Index[i]:Index[i+1]]
        idx=0
        [idx,b_anchors_rx,b_anchors_rw,b_gbboxes,
         match_scores]=tf.while_loop(loop_condition,loop_body,[idx,b_anchors_xmin,b_anchors_xmax,
                                                                     b_gbboxes,match_scores])
        match_scores=tf.reshape(match_scores,[-1,num_prop])
        batch_match_scores=tf.concat([batch_match_scores,match_scores],axis=0)
    return batch_match_scores


def binary_logistic_loss(scores,anchors):
    pmask=tf.cast(scores>0.5,dtype=tf.float32)
    num_positive=tf.reduce_sum(pmask)
    num_entries=tf.cast(tf.shape(scores)[0],dtype=tf.float32)    
    ratio=num_entries/num_positive
    coef_0=0.5*(ratio)/(ratio-1)
    coef_1=coef_0*(ratio-1)
    
    anchors=tf.reshape(anchors,[-1])
    loss=coef_1*pmask*tf.log(anchors+0.00001)+coef_0*(1.0-pmask)*tf.log(1.0-anchors+0.00001)
    loss=-tf.reduce_mean(loss)
    num_sample=[tf.reduce_sum(pmask),ratio] 
    return loss,num_sample

def tem_loss(anchors_action,anchors_start,anchors_end,
             match_scores_action,match_scores_start,match_scores_end,config):

    loss_action,num_sample_action = binary_logistic_loss(match_scores_action,anchors_action)
    loss_start,num_sample_start = binary_logistic_loss(match_scores_start,anchors_start)
    loss_end,num_sample_end= binary_logistic_loss(match_scores_end,anchors_end)
    loss={"loss_action":loss_action,"num_sample_action":num_sample_action,
          "loss_start":loss_start,"num_sample_start":num_sample_start,
          "loss_end":loss_end,"num_sample_end":num_sample_end}
    return loss

def tem_train(X_feature,anchors_xmin,anchors_xmax,Y_bbox,Index,LR,config):

    net=tf.layers.conv1d(inputs=X_feature,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net1=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net1=tf.layers.conv1d(inputs=net1,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net2=tf.layers.conv1d(inputs=net,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)
    net2=tf.layers.conv1d(inputs=net2,filters=512,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)

    net3=net1+net2

    net=0.1*tf.layers.conv1d(inputs=net3,filters=3,kernel_size=1,strides=1,padding='same')
    net=tf.nn.sigmoid(net)

    anchors_action = net[:,:,0]
    anchors_start = net[:,:,1]
    anchors_end = net[:,:,2]

    gt_xmins=Y_bbox[:,0]
    gt_xmaxs=Y_bbox[:,1]
    gt_duration=gt_xmaxs-gt_xmins
     
    gt_duration_boundary=tf.maximum(5.0,0.1*gt_duration)    
    
    gt_start_bboxs=tf.stack((gt_xmins-gt_duration_boundary/2,gt_xmins+gt_duration_boundary/2),axis=1)
    gt_end_bboxs=tf.stack((gt_xmaxs-gt_duration_boundary/2,gt_xmaxs+gt_duration_boundary/2),axis=1)
    match_scores_start=tem_bboxes_encode(anchors_xmin,anchors_xmax,gt_start_bboxs,Index,config)
    match_scores_end=tem_bboxes_encode(anchors_xmin,anchors_xmax,gt_end_bboxs,Index,config)
    match_scores_action=tem_bboxes_encode(anchors_xmin,anchors_xmax,Y_bbox,Index,config)

    match_scores_action=tf.reshape(match_scores_action,[-1])
    match_scores_start=tf.reshape(match_scores_start,[-1])
    match_scores_end=tf.reshape(match_scores_end,[-1])

    loss=tem_loss(anchors_action,anchors_start,anchors_end,
                  match_scores_action,match_scores_start,match_scores_end,config)
    

    TEM_trainable_variables=tf.trainable_variables()

    l2 = 0.005 * sum(tf.nn.l2_loss(tf_var) for tf_var in TEM_trainable_variables)
    cost = 2*loss["loss_action"]+loss["loss_start"]+loss["loss_end"]+l2
    loss['l2'] = l2
    loss['cost'] = cost     
        
    optimizer=tf.train.AdamOptimizer(learning_rate=LR).minimize(cost,var_list=TEM_trainable_variables)

    return optimizer,loss,TEM_trainable_variables
    

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """
    def __init__(self):
        self.learning_rates=[0.001]*5 + [0.0001]*15
        self.training_epochs = len(self.learning_rates)
        self.n_inputs = 400
        self.negative_ratio=1
        self.batch_size = 16
        self.num_prop=100


def plotInfo(axs,info,color):
    axs["loss_action"].set_title("loss_action")
    axs["loss_start"].set_title("loss_start")
    axs["loss_end"].set_title("loss_end")
    axs["l2"].set_title("l2")

    axs["loss_action"].plot(info["loss_action"],color)
    axs["loss_start"].plot(info["loss_start"],color)
    axs["loss_end"].plot(info["loss_end"],color)
    axs["l2"].plot(info["l2"],color)
    plt.pause(0.001)

if __name__ == "__main__":
    config = Config()
    
    X_feature = tf.placeholder(tf.float32, shape=(config.batch_size,config.num_prop,config.n_inputs))
    X_xmin=tf.placeholder(tf.float32,[config.batch_size,config.num_prop])
    X_xmax=tf.placeholder(tf.float32,[config.batch_size,config.num_prop])
    Y_bbox=tf.placeholder(tf.float32,[None,2])
    Index=tf.placeholder(tf.int32,[config.batch_size+1])
    LR= tf.placeholder(tf.float32)

    optimizer,loss,SCNN_trainable_variables=tem_train(X_feature,X_xmin,X_xmax,Y_bbox,Index,LR,config)
    
    model_saver=tf.train.Saver(var_list=SCNN_trainable_variables,max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement =True
    sess=tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()  

    trainDataDict=TEM_load_data.getFullData("val")
    testDataDict=TEM_load_data.getFullData("test")


    train_info={"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
    val_info={"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
                
    info_keys=train_info.keys()
        
    for epoch in range(0,config.training_epochs):
    ## TRAIN ##
        batch_window_list=TEM_load_data.getBatchList(len(trainDataDict["gt_bbox"]),config.batch_size,shuffle=True)
        
        mini_info={"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
        
        for idx in range(len(batch_window_list)):
            batch_index,batch_bbox,batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature=TEM_load_data.getBatchData(batch_window_list[idx],trainDataDict)
            _,out_loss=sess.run([optimizer,loss], feed_dict={X_feature:batch_anchor_feature,
                                                              X_xmin:batch_anchor_xmin,
                                                              X_xmax:batch_anchor_xmax,
                                                              Y_bbox:batch_bbox,
                                                              Index:batch_index,
                                                              LR:config.learning_rates[epoch]})  
            for key in info_keys:
                mini_info[key].append(out_loss[key])

        for key in info_keys:
            train_info[key].append(np.mean(mini_info[key]))
        #plotInfo(axs,train_info,'r')
        model_saver.save(sess,"models/TEM/tem_model_epoch",global_step=epoch)
        
        batch_window_list=TEM_load_data.getBatchList(len(testDataDict["gt_bbox"]),config.batch_size,shuffle=False)
        mini_info={"loss_action":[],"loss_start":[],"loss_end":[],"l2":[]}
        for idx in range(len(batch_window_list)):
            batch_index,batch_bbox,batch_anchor_xmin,batch_anchor_xmax,batch_anchor_feature=TEM_load_data.getBatchData(batch_window_list[idx],testDataDict)
            out_loss=sess.run(loss,feed_dict={X_feature:batch_anchor_feature,
                                                              X_xmin:batch_anchor_xmin,
                                                              X_xmax:batch_anchor_xmax,
                                                              Y_bbox:batch_bbox,
                                                              Index:batch_index,
                                                              LR:config.learning_rates[epoch]})  
            for key in info_keys:
                mini_info[key].append(out_loss[key])

        for key in info_keys:
            val_info[key].append(np.mean(mini_info[key]))

        
