
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as sio
import feature
from block1 import Model
import os
from tqdm import tqdm
from scipy import io
import random
import h5py
import time

is_train = False  # True = train, False = test/validation
is_validation = False  # True = validation, False = test
batch_size = 16
iter = 1
learning_rate = 0.0001  # 1e-4
train_bool = True


start_time = time.clock()
# Load data
if is_train == True:
    # Train
    data = h5py.File('../data/Train.mat', 'r')
    train_x = np.transpose(data['Train'])

    data = sio.loadmat(r'../data/labels_gallery.mat')
    gallery_labels = data['gallery_labels']

    data = sio.loadmat(r'../data/labels_probe_Train.mat')
    probe_labels = data['probe_tr_labels']

    data = sio.loadmat(r'../data/acc_func.mat')
    acc_value = data['acc_func']
    # Test
else:
    data = h5py.File('../data/Test.mat', 'r')
    train_x = np.transpose(data['Test'])

    data = sio.loadmat(r'../data/labels_gallery.mat')
    gallery_labels = data['gallery_labels']

    data = sio.loadmat(r'../data/labels_probe_Test.mat')
    probe_labels = data['probe_te_labels']

    data = sio.loadmat(r'../data/acc_func.mat')
    acc_value = data['acc_func']

workers = train_x.shape[0]
probe = train_x.shape[1]
gallery = train_x.shape[2]

# Settings
# x = tf.placeholder(tf.float32, [None, 8])
# y_ = tf.placeholder(tf.float32, [None, 1])
# pos_s = tf.placeholder(tf.float32, [probe, 1])
# pos_l = tf.placeholder(tf.float32, [probe])
#
# model = Model(x, y_, pos_s, pos_l)
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
#
# with tf.variable_scope('model'):
#     global_step = tf.Variable(0, name='global_step', trainable=False)
# opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op1 = opt.minimize(model.loss, global_step=global_step, var_list=model.block1_net_variables)
# correct_prediction1 = tf.equal(tf.cast(model.prediction, tf.float32), tf.cast(y_, tf.float32))
# acc1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
# init = tf.global_variables_initializer()
# sess.run(init)
# Train
if is_train:
    # Get features and labels
    train_x_median = np.mean(train_x, axis=0)
    train_x_max = np.argmax(train_x_median, axis=1)
    fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 3, 3)
    fea.generate_features_3d(train_x, train_x_max, acc_value)
    train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)
    train_label = np.reshape(train_label, [-1, 1])
    train_input = np.reshape(train_input, [-1, 8])
    io.savemat('train_feature.mat', {'feature': train_input})
    io.savemat('train_label.mat', {'label': train_label})

else:
    train_x_median = np.median(train_x, axis=0)
    train_x_max = np.argmax(train_x_median, axis=1)

    fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 3, 3)
    fea.generate_features_3d(train_x, train_x_max, acc_value)
    train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)

    train_label = np.reshape(train_label, [-1, 1])

    io.savemat('test_feature.mat', {'feature': train_input})
    io.savemat('test_label.mat', {'label': train_label})
	
	
end_time = time.clock()
print(end_time - start_time)