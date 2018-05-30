
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import logging
import time
import cv2 as cv
import os
import matplotlib.pyplot as plt
from net import *

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
		name = 'log_{}_{}.log'.format(datetime[0: len(datetime) - 2], start_iter)

		 
	logging.basicConfig(filename = path + '/' + name, level = logging.INFO)

	return outdatapath, ckptpath


def readData(indatapath, outdatapath, data_size, batch_size, remove_back = False):

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

	if remove_back:
		normal = normal * mask
		color = color * mask

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

def drawPlots(results, outdatapath):

	# train
	plt.figure('plot')
	acc0, = plt.plot(results[:, 0], results[:, 1], 'r')
	acc3, = plt.plot(results[:, 0], results[:, 2], 'g')
	plt.xlabel('iter')
	plt.ylabel('acc')
	plt.ylim((0, 1))
	plt.legend(handles = [acc0, acc3,], labels = ['acc0', 'acc3'])
	plt.savefig(outdatapath + '/../train_acc.png')
	plt.close()

	plt.figure('plot')
	loss, = plt.plot(results[:, 0], results[:, 3], 'b')
	plt.xlabel('iter')
	plt.ylabel('loss')
	plt.ylim((0, 1))
	plt.legend(handles = [loss,], labels = ['loss'])
	plt.savefig(outdatapath + '/../train_loss.png')
	plt.close()

	# test
	plt.figure('plot')
	acc0, = plt.plot(results[:, 0], results[:, 4], 'r')
	acc3, = plt.plot(results[:, 0], results[:, 5], 'g')
	plt.xlabel('iter')
	plt.ylabel('acc')
	plt.ylim((0, 1))
	plt.legend(handles = [acc0, acc3,], labels = ['acc0', 'acc3'])
	plt.savefig(outdatapath + '/../test_acc.png')
	plt.close()

	plt.figure('plot')
	loss, = plt.plot(results[:, 0], results[:, 6], 'b')
	plt.xlabel('iter')
	plt.ylabel('loss')
	plt.ylim((0, 1))
	plt.legend(handles = [loss,], labels = ['loss'])
	plt.savefig(outdatapath + '/../test_loss.png')
	plt.close()


def train():

	print('train')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 14000 + 1, '20180528_183506_0'


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	indatapath = PATH + 'train_data_540/'
	outdatapath, ckptpath = prepareLog(start_iter, normal_ori_i, datetime)

	data_size = 540
	batch_size = 5
	lp_iter = 2000
	# lp_iter = 20000
	
	remove_back = False
	train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size = readData(
		indatapath, outdatapath, data_size, batch_size, remove_back)
	[size, height, width] = train_normal.shape[0: 3]

	model = DPNet(batch_size, height, width, normal_ori_i)

	logging.info('net: 0')
	logging.info('datapath: ' + indatapath)
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('lp_iter: {}'.format(lp_iter))
	logging.info('remove_back: {}'.format(remove_back))
	logging.info('data_size: {}, train_size: {}, test_size: {}'.format(train_size + test_size, train_size, test_size))
	logging.info('batch_size: {}, learning_rate: {}, lamda: {}'.format(batch_size, model.learning_rate, 1))

	end = size - batch_size

	with tf.Session() as sess:

		train_step, accuracy, accuracy_3, loss_, normal_, reflect_, I_, lr_, lp_ = model.net()

		if start_iter == 0:
			sess.run(tf.global_variables_initializer())
		else:
			tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))

		results = np.empty([0, 7], np.float32)

		for i in range(start_iter, 20000):
			if i % 20 == 0:
				print(i)
			idx = i % end
			if i < lp_iter:
				lamda = 1
			else:
				lamda = 0

			if i % 100 == 0 or i == 19999:

				# train accuracy
				train_accuracy, train_accuracy_3, train_loss, train_lr, train_lp, train_nn, train_reflect, train_ii = sess.run(
					[accuracy, accuracy_3, loss_, lr_, lp_, normal_, reflect_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.lamda: lamda})
				logging.info("{}: train step: {}, \ttraining accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}, \t{:.16f}, \t{:.16f}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					train_accuracy, 
					train_accuracy_3, 
					train_loss, 
					train_lr, 
					train_lp))
				if i % 300 == 0 or i == 19999:
					train_nn = ((train_nn + 1) / 2 * 255).astype(np.uint8)
					train_nn[train_nn > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/prenormal{}_{}.png'.format(i, j), train_nn[j][..., ::-1])

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
						[accuracy, accuracy_3, loss_, lr_, lp_, normal_, reflect_, I_], 
						feed_dict = {
							model.normal: test_normal[test_idx: test_idx + batch_size], 
							model.color: test_color[test_idx: test_idx + batch_size], 
							model.mask: test_mask[test_idx: test_idx + batch_size], 
							model.lamda: lamda})
					result_cnt += 1
					result_sum += np.array(result[0: 5])

					test_nn = result[5]
					test_ii = result[7]
					if i % 2000 == 0 or i == 19999:
						test_nn = ((test_nn + 1) / 2 * 255).astype(np.uint8)
						test_nn[test_nn > 255] = 255
						for j in range(batch_size):
							cv.imwrite(outdatapath + '/testnormal{}_{}.png'.format(i, test_idx + j), test_nn[j][..., ::-1])
					
						test_ii = (test_ii * 255).astype(np.uint8)
						test_ii[test_ii > 255] = 255
						for j in range(batch_size):
							cv.imwrite(outdatapath + '/testimg{}_{}.png'.format(i, test_idx + j), test_ii[j])

					test_idx += batch_size

				[test_accuracy, test_accuracy_3, test_loss, test_lr, test_lp] = result_sum / result_cnt
				logging.info("{}: test step : {}, \ttesting accuracy : {:.16f}, \t{:.16f}, \tloss: {:.16f}, \t{:.16f}, \t{:.16f}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					test_accuracy, 
					test_accuracy_3, 
					test_loss, 
					test_lr, 
					test_lp))

				print("step {}, training accuracy {}, {}, loss {}, {}, {}".format(
					i, train_accuracy, train_accuracy_3, train_loss, train_lr, train_lp))
				print("step {}, testing accuracy {}, {}, loss {}, {}, {}".format(
					i, test_accuracy, test_accuracy_3, test_loss, test_lr, test_lp))

				results = np.append(
					results, 
					np.array([[i, train_accuracy, train_accuracy_3, train_loss, test_accuracy, test_accuracy_3, test_loss]]), 
					axis = 0)

			sess.run(train_step, feed_dict = {
				model.normal: train_normal[idx: idx + batch_size], 
				model.color: train_color[idx: idx + batch_size], 
				model.mask: train_mask[idx: idx + batch_size],
				model.lamda: lamda})

			if i % 1000 == 0 or i == 19999:
				tf.train.Saver().save(sess, ckptpath + '/model_latest')
				tf.train.Saver().save(sess, ckptpath + '/model_{}'.format(i))

		drawPlots(results, outdatapath)



def train1():

	print('train')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 11000 + 1, '20180524_140932'


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	data_size = 40
	batch_size = 5

	indatapath = PATH + 'train_data_{}/'.format(data_size)
	outdatapath, ckptpath = prepareLog(start_iter, normal_ori_i, datetime)

	remove_back = False

	train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size = readData(
		indatapath, outdatapath, data_size, batch_size, remove_back)
	[size, height, width] = train_normal.shape[0: 3]

	model = DPNet1(batch_size, height, width, normal_ori_i)

	logging.info('net: 1')
	logging.info('datapath: ' + indatapath)
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('remove_back: {}'.format(remove_back))
	logging.info('data_size: {}, train_size: {}, test_size: {}'.format(train_size + test_size, train_size, test_size))
	logging.info('batch_size: {}, learning_rate: {}, lamda: {}'.format(batch_size, model.learning_rate, 1))

	end = size - batch_size

	with tf.Session() as sess:

		train_step, accuracy, accuracy_3, loss_, loss1_, loss2_, normal_, I_ = model.net()

		if start_iter == 0:
			sess.run(tf.global_variables_initializer())
		else:
			tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))

		for i in range(start_iter, 20000):
			if i % 20 == 0:
				print(i)
			idx = i % end

			if i % 100 == 0 or i == 19999:
				# train accuracy
				train_accuracy, train_accuracy_3, train_loss, tl1, tl2, train_nn, train_ii = sess.run(
					[accuracy, accuracy_3, loss_, loss1_, loss2_, normal_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.keep_prob: 0.5})
				logging.info("{}: train step: {}, \ttraining accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f} \t{:.16f} \t{:.16f}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					train_accuracy, 
					train_accuracy_3, 
					train_loss,
					tl1, tl2))
				if i % 100 == 0 or i == 19999:
					train_nn = ((train_nn + 1) / 2 * 255).astype(np.uint8)
					train_nn[train_nn > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/prenormal{}_{}.png'.format(i, j), train_nn[j][..., ::-1])

					train_ii = (train_ii * 255).astype(np.uint8)
					train_ii[train_ii > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/preimg{}_{}.png'.format(i, j), train_ii[j])

				print("step {}, training accuracy {}, {}, loss {}, {}, {}".format(
					i, train_accuracy, train_accuracy_3, train_loss, tl1, tl2))

				# # test accuracy
				# test_idx = 0
				# result_cnt = 0
				# result_sum = np.zeros((3), np.float32)
				# while test_idx + batch_size <= test_size:
				# 	result = sess.run(
				# 		[accuracy, accuracy_3, loss_, I_], 
				# 		feed_dict = {
				# 			model.normal: test_normal[test_idx: test_idx + batch_size], 
				# 			model.color: test_color[test_idx: test_idx + batch_size], 
				# 			model.mask: test_mask[test_idx: test_idx + batch_size], 
				# 			model.keep_prob: 0.5})
				# 	print(result[2])
				# 	result_cnt += 1
				# 	result_sum += np.array(result[0: 3])

				# 	test_ii = result[3]
				# 	if i % 2000 == 0 or i == 19999:
				# 		test_ii = (test_ii * 255).astype(np.uint8)
				# 		test_ii[test_ii > 255] = 255
				# 		for j in range(batch_size):
				# 			cv.imwrite(outdatapath + '/testimg{}_{}.png'.format(i, test_idx + j), test_ii[j])

				# 	test_idx += batch_size

				# [test_accuracy, test_accuracy_3, test_loss] = result_sum / result_cnt
				# logging.info("{}: test step : {}, \ttesting accuracy : {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
				# 	time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
				# 	i, 
				# 	test_accuracy, 
				# 	test_accuracy_3, 
				# 	test_loss))

				# print("step {}, testing accuracy {}, {}, loss {}".format(
				# 	i, test_accuracy, test_accuracy_3, test_loss))

			sess.run(train_step, feed_dict = {
				model.normal: train_normal[idx: idx + batch_size], 
				model.color: train_color[idx: idx + batch_size], 
				model.mask: train_mask[idx: idx + batch_size],
				model.keep_prob: 0.5})

			if i % 1000 == 0 or i == 19999:
				tf.train.Saver().save(sess, ckptpath + '/model_latest')
				tf.train.Saver().save(sess, ckptpath + '/model_{}'.format(i))


def train2():

	print('train')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 11000 + 1, '20180524_140932'


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	indatapath = PATH + 'train_data_540/'
	outdatapath, ckptpath = prepareLog(start_iter, normal_ori_i, datetime)

	data_size = 540
	batch_size = 5
	
	remove_back = False
	train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size = readData(
		indatapath, outdatapath, data_size, batch_size, remove_back)
	[size, height, width] = train_normal.shape[0: 3]

	model = DPNet2(batch_size, height, width, normal_ori_i)

	logging.info('net: 2')
	logging.info('datapath: ' + indatapath)
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('remove_back: {}'.format(remove_back))
	logging.info('data_size: {}, train_size: {}, test_size: {}'.format(train_size + test_size, train_size, test_size))
	logging.info('batch_size: {}, learning_rate: {}, lamda: {}'.format(batch_size, model.learning_rate, 1))

	end = size - batch_size

	with tf.Session() as sess:

		train_step, accuracy, accuracy_3, loss_, label_, I_ = model.net()

		if start_iter == 0:
			sess.run(tf.global_variables_initializer())
		else:
			tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))

		for i in range(start_iter, 20000):
			if i % 20 == 0:
				print(i)
			idx = i % end

			if i % 100 == 0 or i == 19999:
				# train accuracy
				train_accuracy, train_accuracy_3, train_loss, train_ii = sess.run(
					[accuracy, accuracy_3, loss_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.keep_prob: 0.5})
				logging.info("{}: train step: {}, \ttraining accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
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
				result_sum = np.zeros((3), np.float32)
				while test_idx + batch_size <= test_size:
					result = sess.run(
						[accuracy, accuracy_3, loss_, I_], 
						feed_dict = {
							model.normal: test_normal[test_idx: test_idx + batch_size], 
							model.color: test_color[test_idx: test_idx + batch_size], 
							model.mask: test_mask[test_idx: test_idx + batch_size], 
							model.keep_prob: 0.5})
					print(result[2])
					result_cnt += 1
					result_sum += np.array(result[0: 3])

					test_ii = result[3]
					if i % 2000 == 0 or i == 19999:
						test_ii = (test_ii * 255).astype(np.uint8)
						test_ii[test_ii > 255] = 255
						for j in range(batch_size):
							cv.imwrite(outdatapath + '/testimg{}_{}.png'.format(i, test_idx + j), test_ii[j])

					test_idx += batch_size

				[test_accuracy, test_accuracy_3, test_loss] = result_sum / result_cnt
				logging.info("{}: test step : {}, \ttesting accuracy : {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					test_accuracy, 
					test_accuracy_3, 
					test_loss))

				print("step {}, training accuracy {}, {}, loss {}".format(
					i, train_accuracy, train_accuracy_3, train_loss))
				print("step {}, testing accuracy {}, {}, loss {}".format(
					i, test_accuracy, test_accuracy_3, test_loss))

			sess.run(train_step, feed_dict = {
				model.normal: train_normal[idx: idx + batch_size], 
				model.color: train_color[idx: idx + batch_size], 
				model.mask: train_mask[idx: idx + batch_size],
				model.keep_prob: 0.5})

			if i % 1000 == 0 or i == 19999:
				tf.train.Saver().save(sess, ckptpath + '/model_latest')
				tf.train.Saver().save(sess, ckptpath + '/model_{}'.format(i))



if __name__ == '__main__':
	train()