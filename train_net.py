
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
	
	# indices = np.load(PATH + 'indices/indices0.npy')
	# indices = np.load(PATH + 'indices/indices1.npy')
	np.save(outdatapath + '/indices.npy', indices)

	train_idx, test_idx = indices[: train_size], indices[train_size:]
	train_normal, test_normal = normal[train_idx], normal[test_idx]
	train_color, test_color = color[train_idx], color[test_idx]
	train_mask, test_mask = mask[train_idx], mask[test_idx]

	return train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size

def drawPlots(results, outdatapath, lamda = 1):

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
	plt.ylim((0, lamda))
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
	plt.ylim((0, lamda))
	plt.legend(handles = [loss,], labels = ['loss'])
	plt.savefig(outdatapath + '/../test_loss.png')
	plt.close()


def train():

	print('train')

	start_iter, datetime = 0, ''
	#start_iter, datetime = 2000 + 1, '20180611_181307_0'
	end_iter = 20000


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	data_size = 500
	# data_size = 25
	batch_size = 5

	# lp_iter = 2000
	# lp_iter = 0
	lp_iter = 20000
	
	lamda_default = 1
	# lamda_default = 10

	learning_rate = 1e-2

	indatapath = PATH + 'train_data_{}/'.format(data_size)
	# indatapath = PATH + 'train_data_{}_1/'.format(data_size)
	# indatapath = PATH + 'train_data_500_1/'
	outdatapath, ckptpath = prepareLog(start_iter, normal_ori_i, datetime)

	remove_back = False
	train_normal, test_normal, train_color, test_color, train_mask, test_mask, train_size, test_size = readData(
		indatapath, outdatapath, data_size, batch_size, remove_back)
	[size, height, width] = train_normal.shape[0: 3]

	lightdir = np.array([0, 0, 1])
	model = DPNet(batch_size, height, width, normal_ori_i, lightdir)

	logging.info('net: 0')
	logging.info('datapath: ' + indatapath)
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('lp_iter: {}'.format(lp_iter))
	logging.info('remove_back: {}'.format(remove_back))
	logging.info('data_size: {}, train_size: {}, test_size: {}'.format(train_size + test_size, train_size, test_size))
	logging.info('batch_size: {}, lamda: {}'.format(batch_size, lamda_default))
	logging.info('lightdir: {}'.format(lightdir))

	end = size - batch_size

	with tf.Session() as sess:

		train_step_adam, train_step_gd, accuracy, accuracy_3, loss_, normal_, reflect_, I_, lr_, lp_ = model.net()

		if start_iter == 0:
			sess.run(tf.global_variables_initializer())
		else:
			tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))


		results = np.empty([0, 7], np.float32)

		for i in range(start_iter, end_iter):
			if i % 20 == 0:
				print(i)
			idx = i % end
			# idx = 0
			if i < lp_iter:
				lamda = lamda_default
			else:
				lamda = 0


			if i % 100 == 0 or i == end_iter - 1:

				# train accuracy
				train_accuracy, train_accuracy_3, train_loss, train_lr, train_lp, train_ii = sess.run(
					[accuracy, accuracy_3, loss_, lr_, lp_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.lamda: lamda})
				logging.info("{}: train step: {}, \ttraining accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}, \t{:.16f}, \t{:.16f}, \tlearning rate: {}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					train_accuracy, 
					train_accuracy_3, 
					train_loss, 
					train_lr, 
					train_lp,
					learning_rate))

				if i % 300 == 0 or i == end_iter - 1:
					train_ii[train_ii < 0] = 0
					train_ii[train_ii > 1] = 1
					train_ii = (train_ii * 255).astype(np.uint8)
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/preimg{}_{}.png'.format(i, j), train_ii[j])

				# test accuracy
				test_idx = 0
				result_cnt = 0
				result_sum = np.zeros((5), np.float32)
				while test_idx + batch_size <= test_size:
					result = sess.run(
						[accuracy, accuracy_3, loss_, lr_, lp_, I_, normal_], 
						feed_dict = {
							model.normal: test_normal[test_idx: test_idx + batch_size], 
							model.color: test_color[test_idx: test_idx + batch_size], 
							model.mask: test_mask[test_idx: test_idx + batch_size], 
							model.lamda: lamda})
					result_cnt += 1
					result_sum += np.array(result[0: 5])

					pre_normal = result[6]
					test_ii = result[5]
					if i % 2000 == 0 or i == end_iter - 1:
						pre_normal = ((pre_normal + 1) / 2 * 255).astype(np.uint8)
						pre_normal[pre_normal > 255] = 255
						for j in range(batch_size):
							cv.imwrite(outdatapath + '/prenormal{}_{}.png'.format(i, test_idx + j), pre_normal[j][..., ::-1])

						test_ii[test_ii < 0] = 0
						test_ii[test_ii > 1] = 1
						test_ii = (test_ii * 255).astype(np.uint8)
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
				# if i == 19999:
				# 	print("total: training accuracy {}, {}, loss {}, {}, {}".format(
				# 		train_accuracy_total, train_accuracy_3_total, train_loss_total, train_lr_total, train_lp_total))
				# 	print("total: testing accuracy {}, {}, loss {}, {}, {}".format(
				# 		test_accuracy, test_accuracy_3, test_loss, test_lr, test_lp))

				results = np.append(
					results, 
					np.array([[i, train_accuracy, train_accuracy_3, train_loss, test_accuracy, test_accuracy_3, test_loss]]), 
					axis = 0)

			# learning_rate = 1e-2
			sess.run(train_step_adam, feed_dict = {
				model.normal: train_normal[idx: idx + batch_size], 
				model.color: train_color[idx: idx + batch_size], 
				model.mask: train_mask[idx: idx + batch_size],
				model.lamda: lamda, 
				model.learning_rate: learning_rate})


		# # adaptive adam
		# 	if i < 100:
		# 		learning_rate = 1e-2
		# 	elif i < 500:
		# 		learning_rate = 1e-3
		# 	elif i < 2000:
		# 		learning_rate = 1e-4	
		# 	else:
		# 		learning_rate = 3e-5
		# 	sess.run(train_step_adam, feed_dict = {
		# 		model.normal: train_normal[idx: idx + batch_size], 
		# 		model.color: train_color[idx: idx + batch_size], 
		# 		model.mask: train_mask[idx: idx + batch_size],
		# 		model.lamda: lamda, 
		# 		model.learning_rate: learning_rate})

		# # # adaptive adam + gd
		# 	if i < 500:
		# 		learning_rate = 1e-2
		# 		sess.run(train_step_adam, feed_dict = {
		# 			model.normal: train_normal[idx: idx + batch_size], 
		# 			model.color: train_color[idx: idx + batch_size], 
		# 			model.mask: train_mask[idx: idx + batch_size],
		# 			model.lamda: lamda, 
		# 			model.learning_rate: learning_rate})
		# 	else:
		# 		learning_rate = 1e-2
		# 		sess.run(train_step_gd, feed_dict = {
		# 			model.normal: train_normal[idx: idx + batch_size], 
		# 			model.color: train_color[idx: idx + batch_size], 
		# 			model.mask: train_mask[idx: idx + batch_size],
		# 			model.lamda: lamda, 
		# 			model.learning_rate: learning_rate})

			if i % 1000 == 0 or i == 19999:
				tf.train.Saver().save(sess, ckptpath + '/model_latest')
				tf.train.Saver().save(sess, ckptpath + '/model_{}'.format(i))

		drawPlots(results, outdatapath, lamda_default)



def train1():

	print('train')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 11000 + 1, '20180524_140932'


	normal_ori_i = 0
	normal_ori = ['train', 'depth2normal']

	data_size = 500
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

		train_step, accuracy, accuracy_3, loss_, loss1_, loss2_, normal_, I_, rho_d_, rho_s_, alpha_, f1_, f2_ = model.net()

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
				train_accuracy, train_accuracy_3, train_loss, tl1, tl2, train_nn, train_ii, rho_d, rho_s, alpha, f1, f2 = sess.run(
					[accuracy, accuracy_3, loss_, loss1_, loss2_, normal_, I_, rho_d_, rho_s_, alpha_, f1_, f2_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.keep_prob: 0.5})
				logging.info("{}: train step: {}, \ttraining accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}, \t{:.16f}, \t{:.16f}".format(
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
						# print(train_nn[j])

					train_ii = (train_ii * 255).astype(np.uint8)
					train_ii[train_ii > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/preimg{}_{}.png'.format(i, j), train_ii[j])

					rho_d = (rho_d * 255).astype(np.uint8)
					rho_d[rho_d > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/rho_d{}_{}.png'.format(i, j), rho_d[j])

					# f1 = (f1 * 255).astype(np.uint8)
					# f1[f1 > 255] = 255
					# f2 = (f2 * 255).astype(np.uint8)
					# f2[f2 > 255] = 255
					# for j in range(batch_size):
					# 	cv.imwrite(outdatapath + '/f1{}_{}.png'.format(i, j), f1[j])
					# 	cv.imwrite(outdatapath + '/f2{}_{}.png'.format(i, j), f2[j])

				print("step {}, training accuracy {}, {}, loss {}, {}, {}, rho_s {}, alpha {}".format(
					i, train_accuracy, train_accuracy_3, train_loss, tl1, tl2, rho_s[0], alpha[0]))

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

def train11():

	print('train11')

	start_iter, datetime = 0, ''
	# start_iter, datetime = 11000 + 1, '20180524_140932'


	normal_ori_i = 1
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

		train_step, loss_, l1_, l2_, normal_, I_ = model.net()

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
				train_loss, l1, l2, train_nn, train_ii = sess.run(
					[loss_, l1_, l2_, normal_, I_], 
					feed_dict = {
						model.normal: train_normal[idx: idx + batch_size], 
						model.color: train_color[idx: idx + batch_size], 
						model.mask: train_mask[idx: idx + batch_size], 
						model.keep_prob: 0.5})
				logging.info("{}: train step: {}, \tloss: {:.16f}, \t{:.16f}, \t{:.16f}".format(
					time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
					i, 
					train_loss, l1, l2))
				if i % 200 == 0 or i == 19999:
					train_ii[train_ii < 0] = 0
					train_ii[train_ii > 1] = 1
					train_ii = (train_ii * 255).astype(np.uint8)
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/preimg{}_{}.png'.format(i, j), train_ii[j])

					train_nn = ((train_nn + 1) / 2 * 255).astype(np.uint8)
					train_nn[train_nn > 255] = 255
					for j in range(batch_size):
						cv.imwrite(outdatapath + '/prenormal{}_{}.png'.format(i, j), train_nn[j][..., ::-1])
						# print(train_nn[j])


				print("step {}, loss {} {} {}".format(
					i, train_loss, l1, l2))

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
	# train()
	train11()