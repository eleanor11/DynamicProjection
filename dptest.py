import numpy as np 
import cv2 as cv
import os

DATAPATH = '../DynamicProjectionData/'

def testRenderPrediction(datetime, dataset, dataidx, npyidx = -1):
	if npyidx == -1:
		npyidx = dataidx

	normal_ori_i = 0

	path = DATAPATH + 'prediction/' + datetime + '/data/'
	outpath = DATAPATH + 'render_prediction/' + datetime
	if not os.path.isdir(outpath):
		os.mkdir(outpath)

	rawdepth_filter = np.load(DATAPATH + 'train_data_{}_rawdepth/rawdepth_filter{}.npy'.format(dataset, dataidx))
	mask = np.load(DATAPATH + 'train_data_{}/mask{}.npy'.format(dataset, dataidx))
	# pre_normal = np.load(DATAPATH + 'train_data_{}/normal{}.npy')
	pre_normal = np.load(path + 'prenormal{}.npy'.format(npyidx))
	pre_reflect = np.load(path + 'prereflect{}.npy'.format(npyidx))
	pre_img = np.load(path + 'preimg{}.npy'.format(npyidx))



	# # # dataset 40 (1)
	# # datetime = '20180531_192936_0'
	# # path = DATAPATH + 'prediction/' + datetime + '/data/'
	# # outpath = DATAPATH + 'render_prediction/' + datetime
	# # if not os.path.isdir(outpath):
	# # 	os.mkdir(outpath)

	# # rawdepth_filter = np.load(DATAPATH + 'train_data_40_rawdepth/rawdepth_filter1.npy')
	# # mask = np.load(DATAPATH + 'train_data_40/mask1.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_40/normal1.npy')
	# # pre_normal = np.load(path + 'prenormal1.npy')
	# # pre_reflect = np.load(path + 'prereflect1.npy')
	# # pre_img = np.load(path + 'preimg1.npy')

	# # dataset pig (1)
	# datetime = '20180616_152112_0'
	# path = DATAPATH + 'prediction/' + datetime + '/data/'
	# outpath = DATAPATH + 'render_prediction/' + datetime
	# if not os.path.isdir(outpath):
	# 	os.mkdir(outpath)

	# rawdepth_filter = np.load(DATAPATH + 'train_data_pig/rawdepth_filter1.npy')
	# mask = np.load(DATAPATH + 'train_data_pig/mask1.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_pig/normal1.npy')
	# pre_normal = np.load(path + 'prenormal1.npy')
	# pre_reflect = np.load(path + 'prereflect1.npy')
	# pre_img = np.load(path + 'preimg1.npy')

	# # # dataset 540 (452)
	# # datetime = '20180530_193148_0'
	# # path = DATAPATH + 'prediction/' + datetime + '/data/'
	# # outpath = DATAPATH + 'render_prediction/' + datetime
	# # if not os.path.isdir(outpath):
	# # 	os.mkdir(outpath)

	# # rawdepth_filter = np.load(DATAPATH + 'train_data_540_rawdepth/rawdepth_filter452.npy')
	# # mask = np.load(DATAPATH + 'train_data_540/mask452.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_540/normal452.npy')
	# # # pre_normal = np.load(path + 'prenormal452.npy')
	# # pre_reflect = np.load(path + 'prereflect452.npy')
	# # pre_img = np.load(path + 'preimg452.npy')

	# # dataset 600 (373)
	# datetime = '20180606_142314_0'
	# path = DATAPATH + 'prediction/' + datetime + '/data/'
	# outpath = DATAPATH + 'render_prediction/' + datetime
	# if not os.path.isdir(outpath):
	# 	os.mkdir(outpath)

	# rawdepth_filter = np.load(DATAPATH + 'train_data_600_rawdepth/rawdepth_filter373.npy')
	# mask = np.load(DATAPATH + 'train_data_600/mask373.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_600/normal373.npy')
	# pre_normal = np.load(path + 'prenormal373.npy')
	# pre_reflect = np.load(path + 'prereflect373.npy')
	# pre_img = np.load(path + 'preimg373.npy')

	# # dataset 500 (367(0))
	# datetime = '20180626_200208_0'
	# path = DATAPATH + 'prediction/' + datetime + '/data/'
	# outpath = DATAPATH + 'render_prediction/' + datetime
	# if not os.path.isdir(outpath):
	# 	os.mkdir(outpath)

	# rawdepth_filter = np.load(DATAPATH + 'train_data_500_rawdepth/rawdepth_filter367.npy')
	# mask = np.load(DATAPATH + 'train_data_500/mask367.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_500/normal367.npy')
	# pre_normal = np.load(path + 'prenormal0.npy')
	# pre_reflect = np.load(path + 'prereflect0.npy')
	# pre_img = np.load(path + 'preimg0.npy')

	# # dataset 500 (316(0))
	# datetime = '20180626_204232_0'
	# path = DATAPATH + 'prediction/' + datetime + '/data/'
	# outpath = DATAPATH + 'render_prediction/' + datetime
	# if not os.path.isdir(outpath):
	# 	os.mkdir(outpath)

	# rawdepth_filter = np.load(DATAPATH + 'train_data_500_rawdepth/rawdepth_filter316.npy')
	# mask = np.load(DATAPATH + 'train_data_500/mask316.npy')
	# # pre_normal = np.load(DATAPATH + 'train_data_500/normal316.npy')
	# pre_normal = np.load(path + 'prenormal0.npy')
	# pre_reflect = np.load(path + 'prereflect0.npy')
	# pre_img = np.load(path + 'preimg0.npy')


	pre_normal[..., 0] = 0 - pre_normal[..., 0]

	pre_img = np.expand_dims(pre_img, 0)
	pre_normal = np.expand_dims(pre_normal, 0)
	pre_reflect = np.expand_dims(pre_reflect, 0)

	return normal_ori_i, rawdepth_filter, mask, pre_img, pre_normal, pre_reflect


def testCorres(corres, mask):

	# # test color projection
	# corres = np.array([[[(i + j + k * 80) % 256 for k in range(3)] for j in range(512)] for i in range(424)])
	# corres[np.logical_not(mask)] = np.array([0, 0, 0])

	# # test image projection
	# image = cv.imread(DATAPATH + 'data/image.bmp')
	# image = image[..., ::-1]
	# x0, y0, x1, y1 = 120, 205, 290, 335
	# w, h = image.shape[0], image.shape[1]
	# corres[x0: x1, y0: y1] = np.array([[image[int((i - x0) / (x1 - x0) * h), int((y1 - 1 - j) / (y1 - y0) * w)] for j in range(y0, y1)] for i in range(x0, x1)])

					
	# test position projection
	# corres[200: 210, 250: 260] = np.array([255, 0, 0])
	# corres[100: 110, 230: 240] = np.array([255, 0, 0])
	

	return corres