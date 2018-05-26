
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import logging
import time
import cv2 as cv
import os
from net import DPNet

PATH = '../DynamicProjectionData/'

def prepareLog(start_iter, normal_ori_i, datetime = ''):
	if start_iter == 0:
		datetime = time.strftime(r"%Y%m%d_%H%M%S", time.localtime())
		path = PATH + 'train_log/{}_{}'.format(datetime, normal_ori_i)
		outdatapath = path + '/data'
		ckptpath = path + '/ckpt'
		os.mkdir(path)
		os.mkdir(outdatapath)
		os.mkdir(ckptpath)
		name = 'log_' + datetime + '.log'
	else:
		path = PATH + 'train_log/' + datetime
		outdatapath = path + '/data'
		ckptpath = path + '/ckpt'
		name = 'log_{}_{}.log'.format(datetime, start_iter)

		 
	logging.basicConfig(filename = path + '/' + name, level = logging.INFO)

	return outdatapath, ckptpath


def readData(indatapath, outdatapath, data_size, batch_size):

	print('load data')

	normal = np.empty([0, 424, 512, 3], np.float32)
	color = np.empty([0, 424, 512, 3], np.uint8)
	mask = np.empty([0, 424, 512], np.bool)

	for i in range(data_size):
		normal = np.append(normal, [np.load(indatapath + 'normal{}.npy'.format(i))], axis = 0)
		color = np.append(color, [np.load(indatapath + 'color{}.npy'.format(i))], axis = 0)
		mask = np.append(mask, [np.load(indatapath + 'mask{}.npy'.format(i))], axis = 0)

	mask = np.expand_dims(mask, axis = 3)
	color = color.astype(np.float32) / 255.0


	train_size = int(data_size * 0.8)
	test_size = data_size - train_size
	if (test_size % batch_size > 0):
		test_size = int(test_size / batch_size) * batch_size
		train_size = test_size * 4

	if os.path.isfile(outdatapath + '/indices.npy'):
		indices = np.load(outdatapath + '/indices.npy')
	else:
		indices = np.random.permutation(data_size)
		np.save(outdatapath + '/indices.npy', indices)
	
	train_idx, test_idx = indices[: train_size], indices[train_size:]
	train_normal, test_normal = normal[train_idx], normal[test_idx]
	train_color, test_color = color[train_idx], color[test_idx]
	train_mask, test_mask = mask[train_idx], mask[test_idx]

	return train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size


def train():

	print('train')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 11000 + 1, '20180524_140932'


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	indatapath = PATH + 'capture_data_handled_0526/'
	outdatapath, ckptpath = prepareLog(start_iter, normal_ori_i, datetime)

	data_size = 540
	batch_size = 5
	
	train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size = readData(indatapath, outdatapath, data_size, batch_size)
	[size, height, width] = train_normal.shape[0: 3]

	model = DPNet(batch_size, height, width, normal_ori_i)

	logging.info('datapath: ' + indatapath)
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('data_size: {}, train_size: {}, test_size: {}'.format(size, train_size, test_size))
	logging.info('batch_size: {}, learning_rate: {}, lamda: {}'.format(batch_size, model.learning_rate, 1))

	end = size - batch_size

	with tf.Session() as sess:

		train_step, accuracy, accuracy_3, loss_, BRDF_, I_, lr_, lp_ = model.net()

		# print(train_step.name, accuracy.name, accuracy_3.name, loss_.name, BRDF_.name, I_.name)

		if start_iter == 0:
			sess.run(tf.global_variables_initializer())
		else:
			tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))

		for i in range(start_iter, 20000):
			if i % 20 == 0:
				print(i)
			idx = i % end
			if i < 2000:
				lamda = 1
			else:
				lamda = 0

			if i % 100 == 0 or i == 19999:

				# train accuracy
				train_accuracy, train_accuracy_3, train_loss, train_lr, train_lp, train_BRDF, train_ii = sess.run(
					[accuracy, accuracy_3, loss_, lr_, lp_, BRDF_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.lamda: lamda})
				logging.info("{}: train step {}, \ttraining accuracy {}, \t{}, \tloss {}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					train_accuracy, 
					train_accuracy_3, 
					train_loss))
				if i % 300 == 0 or i == 19999:
					train_ii = (train_ii * 255).astype(np.uint8)
					train_ii[train_ii > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/preimg{}_{}.png'.format(i, j), train_ii[j])

				# test accuracy
				test_idx = 0
				result_cnt = 0
				result_sum = np.zeros((5), np.float32)
				while test_idx + batch_size <= test_size:
					result = sess.run(
						[accuracy, accuracy_3, loss_, lr_, lp_, BRDF_, I_], 
						feed_dict = {
							model.normal: test_normal[test_idx: test_idx + batch_size], 
							model.color: test_color[test_idx: test_idx + batch_size], 
							model.mask: test_mask[test_idx: test_idx + batch_size], 
							model.lamda: lamda})
					result_cnt += 1
					result_sum += np.array(result[0: 5])

					test_ii = result[6]
					if i % 2000 == 0 or i == 19999:
						test_ii = (test_ii * 255).astype(np.uint8)
						test_ii[test_ii > 255] = 255
						for j in range(batch_size):
							cv.imwrite(outdatapath + '/testimg{}_{}.png'.format(i, test_idx + j), test_ii[j])

					test_idx += batch_size

				[test_accuracy, test_accuracy_3, test_loss, test_lr, test_lp] = result_sum / result_cnt
				logging.info("{}: test step {}, \ttesting accuracy {}, \t{}, \tloss {}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					test_accuracy, 
					test_accuracy_3, 
					test_loss))

				print("step {}, training accuracy {}, {}, loss {}".format(i, train_accuracy, train_accuracy_3, train_loss))
				print("step {}, testing accuracy {}, {}, loss {}".format(i, test_accuracy, test_accuracy_3, test_loss))

			sess.run(train_step, feed_dict = {

				model.normal: train_normal[idx: idx + batch_size], 
				model.color: train_color[idx: idx + batch_size], 
				model.mask: train_mask[idx: idx + batch_size],
				model.lamda: lamda})

			if i % 1000 == 0 or i == 19999:
				tf.train.Saver().save(sess, ckptpath + '/model_latest')
				tf.train.Saver().save(sess, ckptpath + '/model_{}'.format(i))



if __name__ == '__main__':
	train()