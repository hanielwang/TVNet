import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import tensorflow as tf
import time
import random
import json
import scipy.io as sio
from load_dataset import Dataset
import options_voting as options
from matplotlib import pyplot
import matplotlib.pyplot as plt

plt.switch_backend('agg')
args = options.parser.parse_args()
tf.reset_default_graph()

def _variable_with_weight_decay(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def test(dataset, args, itr):
    feature_seq = tf.placeholder(tf.float32, [1, args.window_length, args.feature_size])

    net=tf.layers.conv1d(inputs=feature_seq,filters=256,kernel_size=3,strides=1,name = "conv1d1",padding='same',activation=None,reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)

    net=tf.layers.conv1d(inputs=net,filters=128,kernel_size=3,strides=1,name = "conv1d2",padding='same',activation=None,reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)

    net=tf.layers.conv1d(inputs=net,filters=256,kernel_size=3,strides=1,name = "conv1d3",padding='same',activation=None,reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1,forget_bias=1,activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)

    h0 = lstm_cell.zero_state(1, np.float32)
 
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, net, initial_state=h0)

    outputs = outputs[:,:,-1]

    outputs = tf.squeeze(outputs)

#################################################################################

    # Initialize everything
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver()
 
    saver.restore(sess,tf.train.latest_checkpoint('./models/VEM/'+str(args.voting_type)+'/L'+str(args.window_length)+'S'+str(args.window_stride)+'/'))

    # Test
    element_logit_stack = []
    instance_logit_stack = []
    label_stack = []
    done = False
    batch_result=[]

    while not done: #test all samples automatically
        features, idx, attention, done = dataset.load_data_slide_window(is_training=False)
        features_length = len(features)
        features_sup = np.multiply(features,np.expand_dims(attention,1))
        acummu_idxs = [[]for n in range(len(features))]
        output_all = []
        for i in range(0, len(features)-int(args.window_length)+1, 1):
            outputt = sess.run([outputs], feed_dict={feature_seq: np.expand_dims(features_sup[i:i+int(args.window_length)], axis=0)})
            vote_all = []
            vote = 0
            for j in range(0,int(args.window_length)):
                sum_pre = 0
                sum_post = 0
                for n in range(0,j+1):
                    if args.voting_type == 'start':
                        sum_pre = sum_pre - outputt[0][n]
                    else:
                        sum_pre = sum_pre + outputt[0][n]

                for n2 in range(j+1,int(args.window_length)):
                    if args.voting_type == 'start':
                        sum_post = sum_post + outputt[0][n2]
                    else:
                        sum_post = sum_post - outputt[0][n2]

                vote = sum_pre + sum_post
                acummu_idxs[i+j].append(vote)
                vote_all.append(vote)  

        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        def accumulate_all_windows(acummu_idxs, features_length, idx, test_stride):
            pred_start_points = np.zeros(features_length)
            acummu_idxs_sum = []
            acummu_sum = 0
            for i in range(0,features_length):
                acummu_sum = np.sum(acummu_idxs[i],axis=0)
                if len(acummu_idxs[i]) == args.window_length:
                    acummu_idxs_sum.append(acummu_sum)
                else:
                    acummu_sum2 = args.window_length*(acummu_sum/len(acummu_idxs[i]))
                    acummu_idxs_sum.append(acummu_sum2)
           
            acummu_idxs_sum = np.array(acummu_idxs_sum)
            acummu_idxs_sum_norm = normalization(acummu_idxs_sum)

            return acummu_idxs_sum_norm

        acummu_idxs_sum_norm = accumulate_all_windows(acummu_idxs, features_length,idx,1)

        save_path = "./outputs/VEM_"+str(args.voting_type)+"_L_"+str(args.window_length)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if args.voting_type == 'start':
            columns=["start"]
            tmp_df=pd.DataFrame(acummu_idxs_sum_norm,columns=columns)
            tmp_df.to_csv(save_path+"/"+idx+".csv",index=False)
           
        else:
            columns=["end"]
            tmp_df=pd.DataFrame(acummu_idxs_sum_norm,columns=columns)
            tmp_df.to_csv(save_path+"/"+idx+".csv",index=False)



if __name__ == "__main__":
   args = options.parser.parse_args()

   dataset = Dataset(args)
   print ('VEM test start....')
   test(dataset, args, 0) 
