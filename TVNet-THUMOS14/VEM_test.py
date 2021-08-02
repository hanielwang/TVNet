import os
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import random
import scipy.io as sio
#from video_dataset_attention import Dataset
from load_dataset import Dataset

import options_voting as options
# from classificationMAP import getClassificationMAP as cmAP
# from detectionMAP import getDetectionMAP as dmAP
from matplotlib import pyplot
import matplotlib.pyplot as plt

plt.switch_backend('agg')
args = options.parser.parse_args()
tf.reset_default_graph()

save_path = './outputs/VEM_Test/L'+str(args.window_length)+'_'+str(args.voting_type)+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def test(dataset, args, itr):
    # Placeholders
    feature_seq = tf.placeholder(tf.float32, [1, args.window_length, args.feature_size])

    # Model
    net=tf.layers.conv1d(inputs=feature_seq,filters=512,kernel_size=3,strides=1,name = "conv1d1",padding='same',activation=None,reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)

    net=tf.layers.conv1d(inputs=net,filters=256,kernel_size=3,strides=1,name = "conv1d2",padding='same',activation=None,reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1,forget_bias=1,activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)

    h0 = lstm_cell.zero_state(1, np.float32)

    outputs, state = tf.nn.dynamic_rnn(lstm_cell, net, initial_state=h0)

    outputs = outputs[:,:,-1]

    # Initialize everything
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint('./models/VEM/'+str(args.voting_type)+'/L'+str(args.window_length)+'S'+str(args.window_stride)+'/'))

    # Test
    done = False
    print ('VEM test start...')
    while not done: #test all vids
        video_name,features,attention,done = dataset.load_data_slide_window(is_training=False) 
        features_length = len(features)
        acummu_idxs = [[]for n in range(len(features))]
        features= np.multiply(features,np.expand_dims(attention,1))

        for i in range(0, len(features)-int(args.window_length), 1):#third: stride

            outputt = sess.run([outputs], feed_dict={feature_seq: np.expand_dims(features[i:i+int(args.window_length)], axis=0)})

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

        def accumulate_all_windows(acummu_idxs, features_length, test_stride):
            pred_start_points = np.zeros(features_length)
            acummu_idxs_sum = []
            acummu_sum = 0
            for i in range(0,features_length):
                acummu_sum = np.sum(acummu_idxs[i],axis=0)
                acummu_idxs_sum.append(acummu_sum)
            
            acummu_idxs_sum = np.array(acummu_idxs_sum)
            acummu_idxs_sum_norm = normalization(acummu_idxs_sum)

            return acummu_idxs_sum_norm

        acummu_idxs_sum_norm = accumulate_all_windows(acummu_idxs, features_length,1)

        if args.voting_type == 'start':
            columns=["start"]
            tmp_df=pd.DataFrame(acummu_idxs_sum_norm,columns=columns)
            tmp_df.to_csv(save_path+'/'+str(video_name)+".csv",index=False)
            #tmp_df.to_csv('./outputs/VEM_Test/L'+str(args.window_length)+'_'+str(args.voting_type)+'/'+str(video_name)+".csv",index=False)

        else:
            columns=["end"]
            tmp_df=pd.DataFrame(acummu_idxs_sum_norm,columns=columns)
            tmp_df.to_csv(save_path+'/'+str(video_name)+".csv",index=False)
            #tmp_df.to_csv('./outputs/VEM_Test/L'+str(args.window_length)+'_'+str(args.voting_type)+'/'+str(video_name)+".csv",index=False)                
    print ('VEM test finished')
if __name__ == "__main__":
   args = options.parser.parse_args()

   dataset = Dataset(args)
   test(dataset, args, 0)   
