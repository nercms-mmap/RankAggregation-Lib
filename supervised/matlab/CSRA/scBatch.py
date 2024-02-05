# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:29:48 2017

@author: Administrator
"""
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
import sys

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

is_train = True  # True = train, False = test/validation
is_validation = False  # True = validation, False = test
batch_size = 15
iter = 1
learning_rate = 0.0001  # 1e-4
train_bool = True
kk = 10
path = '/dat01/fanxinyao/ext-CSRA/aggregate/cuhk03labeled/data/'


def longtailfunc(x):
    # r = -0.001*x+1.001
    r = 1 / (0.1 * (x - 1) + 1)
    # r = 1-np.e**(-0.05)+np.e**(-0.05*x)
    # r = 556.12198*(np.e**((-(np.log(x+1e-5)-12.8)**2)/2*(3.6)**2))/(x+1e-5)	#a
    # r = 42.31545*(np.e**((-(np.log(x+1e-5)-10.4)**2)/2*(3.8)**2))/(x+1e-5)	#c
    # if x<=1:
    # 	r=1
    # else:
    # 	r=0.01
    return r


# Load data
if is_train == True:
    data = h5py.File('../data/train.mat', 'r')
    train_x = np.transpose(data['Train'])

    labels_gallery = sio.loadmat(os.path.join(path, 'labels_gallery.mat'))
    gallery_labels = labels_gallery['gallery_labels']

    labels_probe_Train = sio.loadmat(os.path.join(path, 'labels_probe_Train.mat'))
    probe_labels = labels_probe_Train['probe_tr_labels']

    acc_func = sio.loadmat(os.path.join(path, 'acc_func.mat'))
    acc_value = acc_func['acc_func']

else:
    data = h5py.File('../data/test.mat', 'r')
    train_x = np.transpose(data['Test'])

    labels_gallery = sio.loadmat(os.path.join(path, 'labels_gallery.mat'))
    gallery_labels = labels_gallery['gallery_labels']

    labels_probe_Test = sio.loadmat(os.path.join(path, '/labels_probe_Test.mat'))
    probe_labels = labels_probe_Test['probe_te_labels']

    acc_func = sio.loadmat(os.path.join(path, 'acc_func.mat'))
    acc_value = acc_func['acc_func']

workers = train_x.shape[0]
probe = train_x.shape[1]
gallery = train_x.shape[2]

# Settings
x = tf.placeholder(tf.float32, [None, kk])
# x_input = tf.placeholder(tf.float32, [None, 16])
y_ = tf.placeholder(tf.float32, [None, 1])
# y2_label = tf.placeholder(tf.float32, [None, 1]
pos_s = tf.placeholder(tf.float32, [probe, 1])
pos_l = tf.placeholder(tf.float32, [probe])
model = Model(x, y_, pos_s, pos_l)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.variable_scope('model'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op1 = opt.minimize(model.loss, global_step=global_step, var_list=model.block1_net_variables)
correct_prediction1 = tf.equal(tf.cast(model.prediction, tf.float32), tf.cast(y_, tf.float32))
acc1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
init = tf.global_variables_initializer()
sess.run(init)
# Train
if is_train:
    # Get features and labels
    train_fea_file = '../train_feature.mat'
    train_lab_file = '../train_label.mat'

    # if the train_fea_file or train_lab_file not exist
    if not (os.path.isfile(train_fea_file) and os.path.isfile(train_lab_file)):
        train_x_median = np.median(train_x, axis=0)
        train_x_max = []
        for j in range(probe):
            probe_label = probe_labels[j]
            median_score = train_x_median[j]
            gt_position = np.where(gallery_labels == probe_label)[0]
            sorted_median_position = np.argsort(-median_score)
            max_position = []
            for k in range(len(gt_position)):
                gt = gt_position[k]
                max_p = np.where(sorted_median_position == gt)
                max_position.append(max_p)
            train_x_max.append(max_position)

        train_x_median = np.median(train_x, axis=0)
        train_x_max = np.argmax(train_x_median, axis=1)
        fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 5, 5)
        fea.generate_features_3d(train_x, train_x_max, acc_value)
        train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)
        print('train_inputf:', train_input.shape)
        train_label = np.reshape(train_label, [-1, 1])
        train_input = np.reshape(train_input, [-1, kk])
        print('train_inputl:', train_input.shape)
        io.savemat(train_fea_file, {'feature': train_input})
        io.savemat(train_lab_file, {'label': train_label})

    feature = sio.loadmat(train_fea_file)
    train_input = feature['feature']
    label = sio.loadmat(train_lab_file)
    train_label = label['label']
    print('get train features and labels successfully')

    saver = tf.train.Saver(max_to_keep=None)
    if tf.train.get_checkpoint_state('../block1batch/'):
        ckpt = tf.train.get_checkpoint_state('../block1batch/')
        # ckpt='block1batch/epoch4410'
        print('ckpt:', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('get checkpoint successfully')

    if train_bool:
        n_iter = int(len(train_input) / batch_size)
        weights = np.ones([probe, workers])
        input = np.ones([batch_size, kk])
        label_input = np.ones([batch_size, 1])

        print('train_input size: ', train_input.shape)
        print('train_label size: ', train_label.shape)
        print('n_iter size: ', n_iter)
    while True:
        start = time.clock()
        epoch = int(sess.run(global_step) / n_iter) + 1
        # print('epoch:', epoch)
        sorted_weights = []
        weights_inds = []
        for j in range(probe):
            weight = weights[j]
            sorted_weight = sorted(weight, reverse=True)
            weight_inds = np.argsort(-weight)
            sorted_weights.append(sorted_weight)
            weights_inds.append(weight_inds)

        weighted_scores = []
        for j in range(0, probe):
            new_scores = np.zeros(gallery)
            for k in range(0, workers):
                p = weights_inds[j][k]
                n_score = sorted_weights[j][k] * train_x[p][j]
                new_scores = new_scores + n_score
            weighted_scores.append(new_scores)

        sorted_weighted_scores_id = np.argsort(-np.array(weighted_scores))

        gallery_label = []
        for j in range(0, probe):
            gal_lab = []
            for k in range(gallery):
                g_label = gallery_labels[sorted_weighted_scores_id[j][k]]
                gal_lab.append(g_label)
            # if len(gal_lab) != 5332:
            # 	print(j)

            gallery_label.append(np.array(gal_lab))

        position_scores = []
        for k in range(probe):
            probe_label = probe_labels[k]
            # position = gallery_label[k].index(probe_label)
            position = np.where(gallery_label[k] == probe_label)[0]
            standard_position = (len(position) - 1) / 2
            mean_postion = np.mean(position)
            diff = mean_postion - standard_position
            position_score = np.median(longtailfunc(diff))
            position_scores.append(position_score)
        # if len(list(position_score)) != 1:
        # 	print(k)
        position_scores = np.reshape(np.array(position_scores), [probe, 1])
        position_labels = np.ones(probe)

        loss_all, loss_pos, loss_wgh = sess.run([model.loss, model.loss_pos, model.loss_wgh],
                                                feed_dict={x: input, y_: label_input, pos_s: position_scores,
                                                           pos_l: position_labels})
        loss_all = sess.run([model.loss],
                            feed_dict={x: input, y_: label_input, pos_s: position_scores, pos_l: position_labels})

        weights = []
        rand = []
        random_list = np.array(random.sample(list(range(0, workers * probe)), workers * probe))
        # random_list = np.array(random.sample(list(range(0,len(train_input))) ,len(train_input)))
        for i in range(n_iter):
            rand_index = random_list[i * batch_size: (i + 1) * batch_size]
            input = np.reshape(train_input[rand_index], [-1, kk])
            label_input = np.reshape(train_label[rand_index], [-1, 1])

            prediction, label, _ = sess.run([model.prediction, y_, train_op1],
                                            feed_dict={x: input, y_: label_input, pos_s: position_scores,
                                                       pos_l: position_labels})

            weights.append(prediction)

        sorted_random_list_ind = np.argsort(random_list)
        weights = np.reshape(np.array(weights), [probe * workers])
        n_weights = np.zeros(probe * workers)
        for k in range(probe * workers):
            n_weights[k] = weights[sorted_random_list_ind[k]]

        weights = np.reshape(n_weights, [probe, workers])
        input = np.reshape(train_input, [-1, kk])
        label_input = np.reshape(train_label, [-1, 1])
        end = time.clock()

        print(epoch)
        print(
            'epoch: %d, loss_all: .4%f, loss_position: %.4f, loss_weight: %.4f, weight1: %.4f, weight2: %.4f, position_scores: %.4f, epoch_time: %.4f' % (
            epoch, loss_all[0], loss_pos, loss_wgh, weights[0][0], weights[0][workers - 1],
            sum(position_scores) / probe, end - start))
        # print(sum(position_scores)/probe)
        # print('weights:', weights[0][0],weights[0][workers-1])
        # print(position_labels)

        save_path = '../block1batch'
        if epoch % 10 == 0:
            saver.save(sess, os.path.join(save_path, 'epoch{}'.format(epoch)), write_meta_graph=True)
        if epoch == 200000:
            sys.exit(0)
        tf.reset_default_graph()


# Test
else:
    for i in range(0, iter):
        # Get features and labels
        test_fea_file = './test_feature.mat'
        test_lab_file = './test_label.mat'

        # if the test_fea_file or test_lab_file not exist
        if not (os.path.isfile(test_fea_file) and os.path.isfile(test_lab_file)):
            if i == 0:
                train_x_median = np.median(train_x, axis=0)
                train_x_max = np.argmax(train_x_median, axis=1)
            else:
                train_x_max = np.argmax(weighted_scores, axis=1)

            fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 5, 5)
            fea.generate_features_3d(train_x, train_x_max, acc_value)
            train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)

            train_label = np.reshape(train_label, [-1, 1])

            io.savemat(test_fea_file, {'feature': train_input})
            io.savemat(test_lab_file, {'label': train_label})

        feature = sio.loadmat(test_fea_file)
        train_input = feature['feature']
        label = sio.loadmat(test_lab_file)
        train_label = label['label']
        print('get test features and labels successfully')

        saver = tf.train.Saver()
        # model_file = tf.train.latest_checkpoint('block1batch/')
        model_file = 'block1batch/epoch10'
        epoch = model_file.split('/')[-1]
        print('ckpt:', model_file)
        print('epoch:', epoch)
        saver.restore(sess, model_file)
        prediction = []
        for f in range(0, workers * probe):
            input = np.reshape(train_input[f], [-1, 8])
            label_input = np.reshape(train_label[f], [-1, 1])
            position_scores = np.ones([probe, 1])
            position_labels = np.ones(probe)

            pre, label, _ = sess.run([model.prediction, y_, train_op1],
                                     feed_dict={x: input, y_: label_input, pos_s: position_scores,
                                                pos_l: position_labels})
            prediction.append(pre[0])
        # print(f)
        if is_validation:
            io.savemat('train_block1_label' + str(i) + '.mat', {'label': train_label})
            io.savemat('train_block1_prediciton' + str(i) + '.mat', {'prediction': prediction})
        else:
            io.savemat('testres/test_block1_label' + str(i) + '_' + epoch + '.mat', {'label': train_label})
            io.savemat('testres/test_block1_prediciton' + str(i) + '_' + epoch + '.mat',
                       {'prediction': prediction})
# weighted_scores = []
# for j in range(0,probe):
# 	abilities = []
# 	for k in range(0,workers):
# 		ability = np.array(prediction)[j+k*probe]
# 		abilities.append(ability)
#
# 	ability_sum = sum(abilities)
# 	abilities = abilities/ability_sum
#
# 	score=[]
# 	for k in range(0,workers):
# 		s = train_x[k][j]
# 		score.append(s)
# 	weight = sorted(abilities, reverse = True)
#
# 	ind = np.argsort(np.reshape(-abilities,[1,-1]))
# 	new_scores = np.zeros(gallery)
# 	for k in range(0,workers):
# 		p = ind[0][k]
# 		weighting = np.array(weight)[k]*np.array(score)[p]
# 		new_scores = new_scores+weighting
#
# 	weighted_scores.append(new_scores)
#
# if is_validation:
# 	io.savemat('train_weighted_scores'+str(i)+'.mat', {'weighted_scores': weighted_scores})
# else:
# 	io.savemat('test_weighted_scores'+str(i)+'.mat', {'weighted_scores': weighted_scores})
#
#
# # Evaluation
# loss = 0
# for q in range(0,workers*probe):
# 	p = prediction[q]
# 	l = train_label[q]
# 	lossloss = abs(p-l)
# 	loss += lossloss
# loss = loss/(workers*probe)
# acc = 1 - loss
# print(acc)

sess.close()
