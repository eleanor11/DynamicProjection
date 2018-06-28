
def testRenderPrediction(datetime, dataset, dataidx):
	normal_ori_i = 0

	path = DATAPATH + 'prediction/' + datetime + '/data/'
	outpath = DATAPATH + 'render_prediction/' + datetime
	if not os.path.isdir(outpath):
		os.mkdir(outpath)

	rawdepth_filter = np.load(DATAPATH + 'train_data_{}_rawdepth/rawdepth_filter{}.npy'.format(dataset, dataidx))
	mask = np.load(DATAPATH + 'train_data_{}/mask{}.npy'.format(dataset, dataidx))
	# pre_normal = np.load(DATAPATH + 'train_data_{}/normal{}.npy')
	pre_normal = np.load(path + 'prenormal{}.npy'.format(dataidx))
	pre_reflect = np.load(path + 'prereflect{}.npy'.format(dataidx))
	pre_img = np.load(path + 'preimg{}.npy'.format(dataidx))



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

	return normal_ori_i, pre_img, pre_normal, pre_reflect