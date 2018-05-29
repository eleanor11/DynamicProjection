
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import logging
import time
import cv2 as cv
import os
from net import DPNet

PATH = '../DynamicProjectionData/'

def prepareLog(normal_ori_i):
	datetime = time.strftime(r"%Y%m%d_%H%M%S", time.localtime())
	path = PATH + 'prediction/{}_{}'.format(datetime, normal_ori_i)
	outdatapath = path + '/data'
	os.mkdir(path)
	os.mkdir(outdatapath)
	name = 'log_' + datetime + '.log'		 
	logging.basicConfig(filename = path + '/' + name, level = logging.INFO)

	return outdatapath


def readData(indatapath, datasize, remove_back = False):

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

	if remove_back:
		normal = normal * mask
		color = color * mask

	return normal, color, mask


def predict():

	print('start...')

	normal_ori = ['train', 'depth2normal']

	# path = '20180525_171111_0'
	# path = '20180526_101223_1'
	path = '20180528_113802_0'
	normal_ori_i = int(path[len(path) - 1])
	ckptpath = PATH + 'train_log/' + path + '/ckpt'
	indatapath = PATH + 'train_data_540/'
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

		normal_, reflect_, I_ = model.net('predicting')

		# restore model
		tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))

		for i in range(datasize):
			if normal_ori_i == 0:
				pre_normal, reflect, img = sess.run(
					[normal_, reflect_, I_ ], 
					feed_dict = {
						model.normal: normal[i: i + batch_size], 
						model.color: color[i: i + batch_size], 
						model.mask: mask[i: i + batch_size], 
						model.lamda: 1.0}) 
			else:
				reflect, img = sess.run(
					[reflect_, I_], 
					feed_dict = {
						model.normal: normal[i: i + batch_size], 
						model.color: color[i: i + batch_size], 
						model.mask: mask[i: i + batch_size], 
						model.lamda: 1.0}) 

			# npy
			if normal_ori_i == 0:
				np.save(outdatapath + '/prenormal{}.npy'.format(i), pre_normal[0])
			np.save(outdatapath + '/prereflect{}.npy'.format(i), reflect[0])
			np.save(outdatapath + '/preimg{}.npy'.format(i), img[0])

			# img
			if normal_ori_i == 0:
				# normal
				pre_normal = ((pre_normal + 1) / 2 * 255).astype(np.uint8)
				pre_normal[pre_normal > 255] = 255
				cv.imwrite(outdatapath + '/prenormal{}.png'.format(i), pre_normal[0][..., ::-1])

			# reflectance map
			print(np.max(reflect[0]), np.min(reflect[0]), np.average(reflect[0]))
			min = np.min(reflect[0])
			d = np.max(reflect[0]) - min
			reflect = ((reflect - min) / d * 255).astype(np.uint8)
			cv.imwrite(outdatapath + '/prereflect{}.png'.format(i), reflect[0])

			# img
			img = (img * 255).astype(np.uint8)
			img[img > 255] = 255
			cv.imwrite(outdatapath + '/preimg{}.png'.format(i), img[0])
		
	


if __name__ == '__main__':
	predict()