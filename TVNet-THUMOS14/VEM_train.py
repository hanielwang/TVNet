import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import time
import random
import scipy.io as sio
from load_dataset import Dataset
import options_voting as options
import os

args = options.parser.parse_args()
tf.reset_default_graph()



def Train_VEM():


    save_path = './models/VEM/'+str(args.voting_type)+'/L'+str(args.window_length)+'S'+str(args.window_stride)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Placeholders
    learning_rate = tf.placeholder(tf.float32)
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    feature_seq = tf.placeholder(tf.float32, [None, args.window_length, args.feature_size])
    gt_r = tf.placeholder(tf.float32)
    attention_window = tf.placeholder(tf.float32)
 
    # Model

    net=tf.layers.conv1d(inputs=feature_seq,filters=512,kernel_size=3,strides=1,name = "conv1d1",padding='same',activation=None, reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)
    net= tf.layers.batch_normalization(net) 

    net=tf.layers.conv1d(inputs=net,filters=256,kernel_size=3,strides=1,name = "conv1d2",padding='same',activation=None, reuse=tf.AUTO_REUSE)
    net=tf.nn.tanh(net)
    net= tf.layers.batch_normalization(net) 

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1,forget_bias=1,activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)
    h0 = lstm_cell.zero_state(batch_size, np.float32)

    outputs, state = tf.nn.dynamic_rnn(lstm_cell, net, initial_state=h0)
    outputs = outputs[:,:,-1]

    # Loss

    gt_r_tan = tf.nn.tanh(gt_r)
    loss = tf.reduce_mean(tf.square(outputs-gt_r_tan))
    apply_gradient_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
       

    # Initialize tensorflow graph
    init = tf.global_variables_initializer()
    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=200)

    dataset = Dataset(args)
    lr=[0.001]*300+[0.0001]*201

    #Start training
    for i in range(0, len(lr)):

        batch_feature_seq, attention, gt_rr, idx= dataset.load_data_slide_window(batch_size=args.batch_size)

        batch_feature_seq = np.multiply(batch_feature_seq,np.expand_dims(attention,2))

        _, cost = sess.run([apply_gradient_op, loss], feed_dict={feature_seq:batch_feature_seq, attention_window:attention, gt_r:gt_rr, learning_rate: lr[i], batch_size:args.batch_size})
            
        print('Iteration: %d, Loss: %.5f' %(i, cost))


        if i % 100 == 0:

            print('Iteration: %d, Loss: %.5f' %(i, cost))
            saver.save(sess, save_path+'/model', global_step=i)
   

def main():
    Train_VEM()

if __name__ == '__main__':

    main()
