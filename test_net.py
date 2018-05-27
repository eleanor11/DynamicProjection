
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import logging
import time
import cv2 as cv
import os
from net import DPNet

PATH = '../DynamicProjectionData/'
SAVE_DIF = True
SAVE_NPY = False
SAVE_IMG = True

def prepareLog(normal_ori_i):
	datetime = time.strftime(r"%Y%m%d_%H%M%S", time.localtime())
	path = PATH + 'test_log/{}_{}'.format(datetime, normal_ori_i)
	outdatapath = path + '/data'
	os.mkdir(path)
	os.mkdir(outdatapath)
	name = 'log_' + datetime + '.log'		 
	logging.basicConfig(filename = path + '/' + name, level = logging.INFO)

	return outdatapath


def readData(indatapath, datasize):

	print('load data')

	normal = np.empty([0, 424, 512, 3], np.float32)
	color = np.empty([0, 424, 512, 3], np.uint8)
	mask = np.empty([0, 424, 512], np.bool)

	for i in range(datasize):
		normal = np.append(normal, [np.load(indatapath + 'normal{}.npy'.format(i))], axis = 0)
		color = np.append(color, [np.load(indatapath + 'color{}.npy'.format(i))], axis = 0)
		mask = np.append(mask, [np.load(indatapath + 'mask{}.npy'.format(i))], axis = 0)

	mask = np.expand_dims(mask, axis = 3)
	color = color.astype(np.float32) / 255.0

	return normal, color, mask


def test():

	print('start...')

	normal_ori = ['train', 'depth2normal']

	path = '20180527_124027_0'
	normal_ori_i = int(path[len(path) - 1])
	ckptpath = PATH + 'train_log/' + path + '/ckpt'
	indatapath = PATH + 'capture_data_handled_0526/'
	outdatapath = prepareLog(normal_ori_i)

	batch_size = 1
	datasize, datasize_trained = 540, 540
	normal, color, mask = readData(indatapath, datasize)
	[size, height, width] = normal.shape[0: 3]

	lightdir = [0.0, 0.0, 1.0]
	model = DPNet(batch_size, height, width, normal_ori_i, lightdir)

	logging.info('datapath: ' + indatapath)
	logging.info('datasize: {} （trained: {}）'.format(size, datasize_trained))
	logging.info('normal: ' + normal_ori[normal_ori_i])
	logging.info('lightdir: {}'.format(lightdir))
	logging.info('modelpath: ' + ckptpath)

	with tf.Session() as sess:

		accuracy_, accuracy_3_, loss_, normal_, BRDF_, I_, lr_, lp_ = model.net('predicting')

		# restore model
		tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))
		
		result_sum = np.array([0.0, 0.0, 0.0])
		result_old = np.array([0.0, 0.0, 0.0])
		result_new = np.array([0.0, 0.0, 0.0])

		for i in range(datasize):
			if normal_ori_i == 0:
				accuracy, accuracy_3, loss, lr, lp, pre_normal, BRDF, img = sess.run(
					[accuracy_, accuracy_3_, loss_, lr_, lp_, normal_, BRDF_, I_], 
					feed_dict = {
						model.normal: normal[i: i + batch_size], 
						model.color: color[i: i + batch_size], 
						model.mask: mask[i: i + batch_size], 
						model.lamda: 1.0}) 
			else:
				accuracy, accuracy_3, loss, lr, lp, BRDF, img = sess.run(
					[accuracy_, accuracy_3_, loss_, lr_, lp_, BRDF_, I_], 
					feed_dict = {
						model.normal: normal[i: i + batch_size], 
						model.color: color[i: i + batch_size], 
						model.mask: mask[i: i + batch_size], 
						model.lamda: 1.0}) 

			result_sum += np.array([accuracy, accuracy_3, loss])
			if i < datasize_trained:
				result_old += np.array([accuracy, accuracy_3, loss])
			else:
				result_new += np.array([accuracy, accuracy_3, loss])

			logging.info("{}: data: {}, \taccuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
				time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), 
				i, 
				accuracy, 
				accuracy_3, 
				loss))

			# save npy
			if SAVE_NPY:
				if normal_ori_i == 0:
					np.save(outdatapath + '/prenormal{}.npy'.format(i), pre_normal[0])
				np.save(outdatapath + '/preBRDF{}.npy'.format(i), BRDF[0])
				np.save(outdatapath + '/preimg{}.npy'.format(i), img[0])

			# save img
			if SAVE_IMG:
				# normal
				if normal_ori_i == 0:
					if SAVE_DIF:
						dif_normal = np.abs(pre_normal[0] - normal[i])
						dif_normal_avg = np.average(dif_normal, axis = 2)
						cv.imwrite(outdatapath + '/difnormal_avg{}.png'.format(i), (dif_normal_avg / 2 * 255).astype(np.uint8))
					pre_normal = ((pre_normal + 1) / 2 * 255).astype(np.uint8)
					pre_normal[pre_normal > 255] = 255
					cv.imwrite(outdatapath + '/prenormal{}.png'.format(i), pre_normal[0])
				
				# # brdf
				# min = np.min(BRDF[0])
				# d = np.max(BRDF[0] - min)
				# BRDF = ((BRDF + min) / d * 255).astype(np.uint8)
				# cv.imwrite(outdatapath + '/prebrdf{}.png'.format(i), BRDF[0])
				# image
				
				if SAVE_DIF:
					dif_img = np.abs(img[0] - color[i] * mask[i])
					dif_img_avg = np.average(dif_img, axis = 2)
					cv.imwrite(outdatapath + '/difimg_avg{}.png'.format(i), (dif_img_avg / np.max(dif_img_avg) * 255).astype(np.uint8))
				img = (img * 255).astype(np.uint8)
				img[img > 255] = 255
				cv.imwrite(outdatapath + '/preimg{}.png'.format(i), img[0])
		
		if datasize_trained < datasize and datasize_trained > 0:
			result_old /= datasize_trained
			result_new /= (datasize - datasize_trained)
			logging.info("{}: old average: accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
				time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), result_old[0], result_old[1], result_old[2]))
			logging.info("{}: new average: accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
				time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), result_new[0], result_new[1], result_new[2]))
			print("old average: accuracy {}, {}, loss {}".format(result_old[0], result_old[1], result_old[2]))
			print("new average: accuracy {}, {}, loss {}".format(result_new[0], result_new[1], result_new[2]))

		result_sum /= datasize
		logging.info("{}: total average: accuracy: {:.16f}, \t{:.16f}, \tloss: {:.16f}".format(
			time.strftime(r"%Y%m%d_%H%M%S", time.localtime()), result_sum[0], result_sum[1], result_sum[2]))
		print("total average: accuracy {}, {}, loss {}".format(result_sum[0], result_sum[1], result_sum[2]))


if __name__ == '__main__':
	test()