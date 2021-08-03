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

def _variable_with_weight_decay(name, shape, wd):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def main():


    save_path = './models/VEM/'+str(args.voting_type)+'/L'+str(args.window_length)+'S'+str(args.window_stride)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Placeholders
    learning_rate = tf.placeholder(tf.float32)

    batch_size = tf.placeholder(tf.int32, [], name='batch_size')

    feature_seq = tf.placeholder(tf.float32, [None, args.window_length, args.feature_size])

    gt_r = tf.placeholder(tf.float32)

    idxx = tf.placeholder(tf.float32)

    ####################################################################
    # Model

    net=tf.layers.conv1d(inputs=feature_seq,filters=256,kernel_size=3,strides=1,name = "conv1d1",padding='same',activation=None)
    net=tf.nn.tanh(net)

    net=tf.layers.conv1d(inputs=net,filters=128,kernel_size=3,strides=1,name = "conv1d2",padding='same',activation=None)
    net=tf.nn.tanh(net)

    net=tf.layers.conv1d(inputs=net,filters=256,kernel_size=3,strides=1,name = "conv1d3",padding='same',activation=None)
    net=tf.nn.tanh(net)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1,forget_bias=1,activation=tf.nn.tanh)

    h0 = lstm_cell.zero_state(batch_size, np.float32)

    outputs0, state = tf.nn.dynamic_rnn(lstm_cell, net, initial_state=h0)

    outputs = outputs0[:,:,-1]
 
    # Loss
    gt_r_tanh = tf.nn.tanh(gt_r)
    loss = tf.reduce_mean(tf.square(outputs-gt_r_tanh))
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
    lr = [0.0001]*800+[0.00001]*201

    #Start training

    for i in range(0, len(lr)):
        # Train
        batch_feature_seq, gt_rr, attention, idx= dataset.load_data_slide_window(batch_size_cls=args.batch_size)
        attention_sig = attention
        batch_feature_seq_sup = np.multiply(batch_feature_seq,np.expand_dims(attention_sig,2))
        _, cost = sess.run([apply_gradient_op, loss], feed_dict={feature_seq:batch_feature_seq_sup, gt_r:gt_rr, learning_rate: lr[i], batch_size:args.batch_size})


        if i % 200== 0:
            print('Iteration: %d, Loss: %.5f' %(i, cost))
            saver.save(sess, save_path+'/model', global_step=i)

if __name__ == "__main__":
    print ('VEM train Start...')
    main()
