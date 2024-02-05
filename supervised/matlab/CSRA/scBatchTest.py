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

is_train = False# True = train, False = test/validation
is_validation = False #True = validation, False = test
batch_size = 8
iter = 1

learning_rate = 0.0001#1e-4
train_bool = True
# Load data
data = sio.loadmat(r'./Train.mat')
train_x = data['Train']

labels_gallery = sio.loadmat(r'./labels_gallery.mat')
gallery_labels = labels_gallery['gallery_labels']

labels_probe_Train = sio.loadmat(r'./labels_probe_Train.mat')
probe_labels = labels_probe_Train['probe_tr_labels']

acc_func = sio.loadmat(r'./acc_func.mat')
acc_value = acc_func['acc_func']

if is_train == False:
    # Validation
    if is_validation:
        data = sio.loadmat(r'./Train.mat')
        train_x = data['Train']
        
        labels_gallery = sio.loadmat(r'./labels_gallery.mat')
        gallery_labels = labels_gallery['gallery_labels']
        
        labels_probe_Train = sio.loadmat(r'./labels_probe_Train.mat')
        probe_labels = labels_probe_Train['probe_tr_labels']
        
        acc_func = sio.loadmat(r'./acc_func.mat')
        acc_value = acc_func['acc_func']
    # Test
    else:
        data= sio.loadmat(r'./Test.mat')
        train_x = data['Test']
        
        labels_gallery = sio.loadmat(r'./labels_gallery.mat')
        gallery_labels = labels_gallery['gallery_labels']
        
        labels_probe_Test = sio.loadmat(r'./labels_probe_Test.mat')
        probe_labels = labels_probe_Test['probe_te_labels']
        
        acc_value = sio.loadmat(r'./acc_func.mat')
        acc_value = acc_func['acc_func']

workers = train_x.shape[0]
probe = train_x.shape[1]
gallery = train_x.shape[2]

# Settings
x = tf.placeholder(tf.float32, [None, 8])
# x_input = tf.placeholder(tf.float32, [None, 16])
y_ = tf.placeholder(tf.float32,[None, 1])
# y2_label = tf.placeholder(tf.float32, [None, 1]
pos_s = tf.placeholder(tf.float32, [probe,1])
pos_l = tf.placeholder(tf.float32, [probe])
  
model = Model(x,y_,pos_s,pos_l)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

with tf.variable_scope('model'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op1 = opt.minimize(model.loss, global_step=global_step, var_list=model.block1_net_variables)
correct_prediction1 = tf.equal(tf.cast(model.prediction,tf.float32), tf.cast(y_,tf.float32))
acc1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
init = tf.global_variables_initializer()
sess.run(init)
# Train
if is_train:
	# Get features and labels
	# train_x_median = np.median(train_x, axis=0)
	# train_x_max = []
	# for j in range(probe):
	# 	probe_label = probe_labels[j]
	# 	median_score = train_x_median[j]
	# 	gt_position = np.where(gallery_labels == probe_label)[0]
	# 	sorted_median_position = np.argsort(-median_score)
	# 	max_position =[]
	# 	for k in len(gt_position):
	# 		gt = gt_position[k]
	# 		max_p = np.where(sorted_median_position == gt)
	# 		max_position.append(max_p)
	# 	train_x_max.append(max_position)


	# train_x_median = np.median(train_x, axis=0)
	# train_x_max = np.argmax(train_x_median, axis=1)
	# fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 3, 3)
	# fea.generate_features_3d(train_x, train_x_max, acc_value)
	# train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)
	# train_label = np.reshape(train_label,[-1,1])
	# train_input = np.reshape(train_input,[-1,8])
	# io.savemat('train_feature.mat', {'feature': train_input})
	# io.savemat('train_label.mat', {'label': train_label})

	feature = sio.loadmat(r'./train_feature.mat')
	train_input = feature['feature']
	label = sio.loadmat(r'./train_label.mat')
	train_label= label['label']

	saver = tf.train.Saver(max_to_keep=None)
	if tf.train.get_checkpoint_state('block1batch/'):
		saver.restore(sess, 'block1batch')
		print('success')
	if train_bool:
		n_iter = int(len(train_input) / batch_size)
		weights = np.ones([probe,workers])
		input = np.ones([batch_size,8])
		label_input = np.ones([batch_size,1])
	while True:
		epoch = int(sess.run(global_step) / n_iter) + 1
		print('epoch:', epoch)

		sorted_weights = []
		weights_inds = []
		for j in range(probe):
			weight = weights[j]
			sorted_weight = sorted(weight, reverse = True)
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
				# if len(list(new_scores)) != 5332:
				# 	print(k)
			weighted_scores.append(new_scores)
		
		sorted_weighted_scores_id = np.argsort(-np.array(weighted_scores))

		gallery_label = []
		for j in range(0,probe):
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
			standard_position = (len(position)-1) / 2
			mean_postion = np.mean(position)
			diff = mean_postion - standard_position
			position_score = np.median(1 / (0.1 * diff + 1))
			position_scores.append(position_score)
			# if len(list(position_score)) != 1:
			# 	print(k)
		position_scores = np.reshape(np.array(position_scores),[probe,1])
		position_labels = np.ones(probe)
		
		loss_all, loss_pos, loss_wgh = sess.run([model.loss, model.loss_pos, model.loss_wgh], feed_dict={x: input, y_: label_input, pos_s: position_scores, pos_l: position_labels})
		loss_all = sess.run([model.loss], feed_dict={x: input, y_: label_input, pos_s: position_scores, pos_l: position_labels})

		weights = []
		rand =[]
		random_list = np.array(random.sample(list(range(0,workers*probe)) ,workers*probe))
		for i in tqdm(range(n_iter)):
			rand_index = random_list[i * batch_size : (i+1) * batch_size]
			input = np.reshape(train_input[rand_index], [-1, 8])
			label_input = np.reshape(train_label[rand_index], [-1, 1])

			prediction,label,_ = sess.run([model.prediction, y_, train_op1], feed_dict={x: input, y_: label_input, pos_s: position_scores, pos_l: position_labels})

			weights.append(prediction)

		sorted_random_list_ind = np.argsort(random_list)
		weights = np.reshape(np.array(weights),[probe*workers])
		n_weights = np.zeros(probe*workers)
		for k in range(probe*workers):
			n_weights[k] = weights[sorted_random_list_ind[k]]

		weights = np.reshape(n_weights,[probe,workers])
		input = np.reshape(train_input, [-1, 8])
		label_input = np.reshape(train_label, [-1, 1])

		print('epoch:%d,loss_all:%f,loss_position:%f,loss_weight:%f' %(epoch, loss_all[0], loss_pos, loss_wgh))
		print(sum(position_scores)/probe)
		print("weights:", weights[0][0],weights[0][workers-1])
		# print(position_labels)
	
		save_path = 'block1batch'
		if epoch % 100 == 0:
			saver.save(sess, os.path.join(save_path, 'epoch{}'.format(epoch)), write_meta_graph=True)
		if epoch == 300000:
			sys.exit(0)
		tf.reset_default_graph()
 

# Test
else:
	for i in range(0,iter):
		# Get features and labels
		# if i == 0:
		# 	train_x_median = np.median(train_x, axis=0)
		# 	train_x_max = np.argmax(train_x_median, axis=1)
		# else:
		# 	train_x_max = np.argmax(weighted_scores,axis=1)
		#
		# fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 3, 3)
		# fea.generate_features_3d(train_x, train_x_max, acc_value)
		# train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)
		#
		# train_label = np.reshape(train_label,[-1,1])
		#
		# io.savemat('test_feature.mat', {'feature': train_input})
		# io.savemat('test_label.mat', {'feature': train_label})

		# train_x_median = np.median(train_x, axis=0)
		# train_x_max = np.argmax(train_x_median, axis=1)
		# fea = feature.FeatureRepresenter_3D(workers, probe, gallery, 3, 3, 3)
		# fea.generate_features_3d(train_x, train_x_max, acc_value)
		# train_input, train_label = fea.get_features_3d(train_x, gallery_labels, probe_labels, acc_value)
		# train_label = np.reshape(train_label,[-1,1])
		# train_input = np.reshape(train_input,[-1,8])
		# io.savemat('test_feature.mat', {'feature': train_input})
		# io.savemat('test_label.mat', {'label': train_label})

		feature = sio.loadmat(r'./test_feature.mat')
		train_input = feature['feature']
		label = sio.loadmat(r'./test_label.mat')
		train_label= label['label']

		saver = tf.train.Saver()
		model_file = tf.train.latest_checkpoint('block1batch-1/')
		saver.restore(sess, model_file)
		prediction = []
		for f in range(0, workers*probe):
			input = np.reshape(train_input[f], [-1, 8])
			label_input = np.reshape(train_label[f], [-1, 1])
			position_scores = np.ones([probe,1])
			position_labels = np.ones(probe)

			pre, label, _ = sess.run([model.prediction, y_, train_op1],feed_dict={x: input, y_: label_input, pos_s: position_scores,pos_l: position_labels})
			prediction.append(pre[0])
			# print(f)
		if is_validation:
			io.savemat('train_block1_label'+str(i)+'.mat', {'label': train_label})
			io.savemat('train_block1_prediciton'+str(i)+'.mat', {'prediction': prediction})
		else:
			io.savemat('test_block1_label'+str(i)+'.mat', {'label': train_label})
			io.savemat('test_block1_prediciton'+str(i)+'.mat', {'prediction': prediction})
		weighted_scores = []
		for j in range(0,probe):
			abilities = []
			for k in range(0,workers):
				ability = np.array(prediction)[j+k*probe]
				abilities.append(ability)
				
			ability_sum = sum(abilities)
			abilities = abilities/ability_sum
			
			score=[]
			for k in range(0,workers):
				s = train_x[k][j]
				score.append(s)
			weight = sorted(abilities, reverse = True)
			
			ind = np.argsort(np.reshape(-abilities,[1,-1]))
			new_scores = np.zeros(gallery)
			for k in range(0,workers):
				p = ind[0][k]
				weighting = np.array(weight)[k]*np.array(score)[p]
				new_scores = new_scores+weighting
				
			weighted_scores.append(new_scores)
		
		if is_validation:
			io.savemat('train_weighted_scores'+str(i)+'.mat', {'weighted_scores': weighted_scores})
		else:
			io.savemat('test_weighted_scores'+str(i)+'.mat', {'weighted_scores': weighted_scores})

		
		# Evaluation
		loss = 0
		for q in range(0,workers*probe):
			p = prediction[q]
			l = train_label[q]
			lossloss = abs(p-l)
			loss += lossloss
		loss = loss/(workers*probe)
		acc = 1 - loss
		# print(loss)
		print(acc)

sess.close()
