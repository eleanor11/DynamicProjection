import tensorflow as tf 
import numpy as np
import math

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W, strides):
	return tf.nn.conv2d(x, W, strides, padding = 'SAME')

def upsample(x, size):
	return tf.image.resize_images(x, size)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def relu(x):
	return tf.nn.relu(x)

def sigmoid(x):
	return tf.nn.sigmoid(x)

def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)

def softmax(x):
	return tf.nn.softmax(x)

def l2_norm(x, dim):
	return tf.nn.l2_normalize(x, axis = dim)

def batch_norm(x):
	[x_mean, x_varia] = tf.nn.moments(x, axes = [0, 1, 2])
	offset = 0
	scale = 0.1
	vari_epsl = 0.0001
	## calculate the batch normalization
	return tf.nn.batch_normalization(x, x_mean, x_varia, offset, scale, vari_epsl)

class DPNet:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir, tf.float32)
		self.oc = 16


	def PSNet(self, x):

		W_conv1 = weight_variable([3, 3, 3, self.oc])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		W_conv2 = weight_variable([3, 3, self.oc, self.oc])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 1, 1, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)

		W_conv3 = weight_variable([3, 3, self.oc, self.oc])
		h_conv3 = conv2d(h_conv2_relu, W_conv3, [1, 1, 1, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)

		W_feature = weight_variable([3, 3, self.oc, int(self.oc / 2)])
		h_feature = conv2d(h_conv2_relu, W_feature, [1, 1, 1, 1])
		h_feature_bn = batch_norm(h_feature)
		h_feature_relu = relu(h_feature_bn)

		# 0: train, 1: self.normal
		if self.normal_ori == 1:
			return h_feature_relu, self.normal, 0

		W_conv4 = weight_variable([3, 3, self.oc, 3])
		h_conv4 = conv2d(h_conv3_relu, W_conv4, [1, 1, 1, 1])
		h_l2_norm = l2_norm(h_conv4, 3)

		return h_feature_relu, h_l2_norm, tf.reduce_sum(self.mask * (self.normal - h_l2_norm) ** 2)


	def IRNet(self, x, feature):

		# add specular term to x

		W_conv1 = weight_variable([3, 3, 4, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		W_conv2 = weight_variable([3, 3, 16, 16])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 1, 1, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)

		W_conv3 = weight_variable([3, 3, 16, 16])
		h_conv3 = conv2d(h_conv2_relu, W_conv3, [1, 1, 1, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)

		h_concat = tf.concat([h_conv3_relu, feature], 3)

		W_conv4 = weight_variable([1, 1, 16 + int(self.oc / 2), 16])
		h_conv4 = conv2d(h_concat, W_conv4, [1, 1, 1, 1])
		h_conv4_bn = batch_norm(h_conv4)
		h_conv4_relu = relu(h_conv4_bn)

		W_conv5 = weight_variable([3, 3, 16, 16])
		h_conv5 = conv2d(h_conv4_relu, W_conv5, [1, 1, 1, 1])
		h_conv5_bn = batch_norm(h_conv5)
		h_conv5_relu = relu(h_conv5_bn)

		W_conv6 = weight_variable([3, 3, 16, 3])
		h_conv6 = conv2d(h_conv5_relu, W_conv6, [1, 1, 1, 1])

		# return h_conv6

		h_sig = sigmoid(h_conv6)
		return h_sig


	def net(self, mode = 'training'):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.lamda = tf.placeholder(tf.float32, name = 'lamda')
		self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

		viewdir = tf.constant([0.0, 0.0, 1.0])

		mat_size = self.size * self.height * self.width
		mat_shape = [self.size, self.height, self.width, 3, 1]
		lightdir_mat = tf.reshape(tf.tile(self.lightdir, [mat_size]), mat_shape)
		viewdir_mat = tf.reshape(tf.tile(viewdir, [mat_size]), mat_shape)

		# PS Net
		# normal: M * H * W * 3
		feature, normal, lp = self.PSNet(self.color)
		
		normal_mat = tf.expand_dims(normal, 3)
		nxl = tf.reshape(tf.matmul(normal_mat, lightdir_mat), [self.size, self.height, self.width, 1])
		mat_t = tf.expand_dims(nxl * normal * 2 - self.lightdir, 3)
		specular = tf.reshape(tf.matmul(mat_t, viewdir_mat), [self.size, self.height, self.width, 1])

		# IR Net
		# reflectance map: M * H * W * 3
		reflect = self.IRNet(tf.concat([self.color, specular], 3), feature)


		I_ = reflect * relu(nxl)

		lr = tf.reduce_sum(self.mask * tf.abs(self.color - I_))
		mask_sum = tf.reduce_sum(self.mask) * 3

		loss_res = lr / mask_sum
		loss_prior = lp / mask_sum
		loss = loss_res + self.lamda * loss_prior
		train_step_adam = tf.train.AdamOptimizer(self.learning_rate, name = 'train_step').minimize(loss)
		train_step_gd = tf.train.GradientDescentOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		delta3 = 3.0 / 256.0
		accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta3), tf.float32) * self.mask) / mask_sum
		delta5 = 5.0 / 256.0
		accuracy_5 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta5), tf.float32) * self.mask) / mask_sum

		accuracy_normal = tf.reduce_sum(tf.cast(tf.equal(self.normal, normal), tf.float32) * self.mask) / mask_sum
		delta3 = 2 * 0.03
		accuracy_n3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.normal - normal), delta3), tf.float32) * self.mask) / mask_sum
		delta5 = 2 * 0.05
		accuracy_n5 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.normal - normal), delta5), tf.float32) * self.mask) / mask_sum

		normalm = normal * self.mask
		reflectm = reflect * self.mask
		Im_ = I_ * self.mask

		if mode == 'training':
			return train_step_adam, train_step_gd, accuracy, accuracy_3, loss, normalm, reflectm, Im_, loss_res, loss_prior
		elif mode == 'testing':
			return accuracy, accuracy_3, accuracy_5, accuracy_n5, loss, normalm, reflectm, Im_, loss_res, loss_prior
		elif mode == 'predicting':
			return normalm, reflectm, Im_


class DPNet1:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir)
		self.learning_rate = 1e-2

	def FCNet(self, x):

		# 424 * 512 * 3 -> 424 * 512 * 16
		W_conv1 = weight_variable([3, 3, 3, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		# 424 * 512 * 16 -> 106 * 128 * 32
		W_conv2 = weight_variable([3, 3, 16, 32])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 2, 2, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)
		h_pool2 = max_pool_2x2(h_conv2_relu)

		# 106 * 128 * 32 -> 27 * 32 * 64
		W_conv3 = weight_variable([3, 3, 32, 64])
		h_conv3 = conv2d(h_pool2, W_conv3, [1, 2, 2, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)
		h_pool3 = max_pool_2x2(h_conv3_relu)

		# 27 * 32 * 64 -> 7 * 8 * 128
		W_conv4 = weight_variable([3, 3, 64, 128])
		h_conv4 = conv2d(h_pool3, W_conv4, [1, 2, 2, 1])
		h_conv4_bn = batch_norm(h_conv4)
		h_conv4_relu = relu(h_conv4_bn)
		h_pool4 = max_pool_2x2(h_conv4_relu)

		h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 8 * 128])

		# fc
		W_fc1 = weight_variable([7 * 8 * 128, 1024])
		b_fc1 = weight_variable([1024])
		h_fc1 = relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
		h_fc1_drop = dropout(h_fc1, self.keep_prob)

		W_fc2 = weight_variable([1024, 6])
		b_fc2 = weight_variable([6])
		h_fc2 = relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		h_sig = sigmoid(h_fc2)

		return h_sig


	def SVNet(self, x, l2 = False):

		# 424 * 512 * 3 -> 424 * 512 * 16
		W_conv1 = weight_variable([3, 3, 3, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		# 424 * 512 * 16 -> 106 * 128 * 32
		W_conv2 = weight_variable([3, 3, 16, 32])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 2, 2, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)
		h_pool2 = max_pool_2x2(h_conv2_relu)

		# 106 * 128 * 32 -> 27 * 32 * 64
		W_conv3 = weight_variable([3, 3, 32, 64])
		h_conv3 = conv2d(h_pool2, W_conv3, [1, 2, 2, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)
		h_pool3 = max_pool_2x2(h_conv3_relu)

		# 27 * 32 * 64 -> 7 * 8 * 128
		W_conv4 = weight_variable([3, 3, 64, 128])
		h_conv4 = conv2d(h_pool3, W_conv4, [1, 2, 2, 1])
		h_conv4_bn = batch_norm(h_conv4)
		h_conv4_relu = relu(h_conv4_bn)
		h_pool4 = max_pool_2x2(h_conv4_relu)

		# 7 * 8 * 128 -> 7 * 8 * 128
		W_conv5 = weight_variable([3, 3, 128, 128])
		h_conv5 = conv2d(h_pool4, W_conv5, [1, 1, 1, 1])
		h_conv5_bn = batch_norm(h_conv5)
		h_conv5_relu = relu(h_conv5_bn)

		# 7 * 8 * 128 -> 27 * 32 * 64
		W_conv6 = weight_variable([3, 3, 128, 64])
		h_conv6 = conv2d(upsample(h_conv5_relu, [14, 16]), W_conv6, [1, 1, 1, 1])
		h_conv6_bn = batch_norm(h_conv6)
		h_conv6_relu = relu(h_conv6_bn)
		h_upsample6 = upsample(h_conv6_relu, [27, 32])

		# 27 * 32 * 64 -> 106 * 128 * 32
		W_conv7 = weight_variable([3, 3, 64, 32])
		h_conv7 = conv2d(upsample(h_upsample6, [53, 64]), W_conv7, [1, 1, 1, 1])
		h_conv7_bn = batch_norm(h_conv7)
		h_conv7_relu = relu(h_conv7_bn)
		h_upsample7 = upsample(h_conv7_relu, [106, 128])

		# 106 * 128 * 32 -> 424 * 512 * 16
		W_conv8 = weight_variable([3, 3, 32, 16])
		h_conv8 = conv2d(upsample(h_upsample7, [212, 256]), W_conv8, [1, 1, 1, 1])
		h_conv8_bn = batch_norm(h_conv8)
		h_conv8_relu = relu(h_conv8_bn)
		h_upsample8 = upsample(h_conv8_relu, [424, 512])

		# 424 * 512 * 16 -> 424 * 512 * 3
		W_conv9 = weight_variable([3, 3, 16, 3])
		h_conv9 = conv2d(h_upsample8, W_conv9, [1, 1, 1, 1])
		h_conv9_bn = batch_norm(h_conv9)
		h_conv9_relu = relu(h_conv9_bn)

		if l2:
			return l2_norm(h_conv9_relu, 3)
		else:
			return h_conv9_relu

	def SVNet1(self, x, l2 = False):

		# 424 * 512 * 3 -> 424 * 512 * 16
		W_conv1 = weight_variable([3, 3, 3, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		# 424 * 512 * 16 -> 212 * 256 * 32
		W_conv2 = weight_variable([3, 3, 16, 32])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 2, 2, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)

		# 212 * 256 * 32 -> 106 * 128 * 64
		W_conv3 = weight_variable([3, 3, 32, 64])
		h_conv3 = conv2d(h_conv2_relu, W_conv3, [1, 2, 2, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)

		# 106 * 128 * 64 -> 53 * 64 * 128
		W_conv4 = weight_variable([3, 3, 64, 128])
		h_conv4 = conv2d(h_conv3_relu, W_conv4, [1, 2, 2, 1])
		h_conv4_bn = batch_norm(h_conv4)
		h_conv4_relu = relu(h_conv4_bn)

		# 53 * 64 * 128 -> 53 * 64 * 128
		W_conv5 = weight_variable([3, 3, 128, 128])
		h_conv5 = conv2d(h_conv4_relu, W_conv5, [1, 1, 1, 1])
		h_conv5_bn = batch_norm(h_conv5)
		h_conv5_relu = relu(h_conv5_bn)

		# 53 * 64 * 128 -> 106 * 128 * 64
		W_conv6 = weight_variable([3, 3, 128, 64])
		h_conv6 = conv2d(upsample(h_conv5_relu, [14, 16]), W_conv6, [1, 1, 1, 1])
		h_conv6_bn = batch_norm(h_conv6)
		h_conv6_relu = relu(h_conv6_bn)
		h_upsample6 = upsample(h_conv6_relu, [27, 32])

		# 106 * 128 * 64 -> 212 * 256 * 32
		W_conv7 = weight_variable([3, 3, 64, 32])
		h_conv7 = conv2d(upsample(h_upsample6, [53, 64]), W_conv7, [1, 1, 1, 1])
		h_conv7_bn = batch_norm(h_conv7)
		h_conv7_relu = relu(h_conv7_bn)
		h_upsample7 = upsample(h_conv7_relu, [106, 128])

		# 212 * 256 * 32 -> 424 * 512 * 16
		W_conv8 = weight_variable([3, 3, 32, 16])
		h_conv8 = conv2d(upsample(h_upsample7, [212, 256]), W_conv8, [1, 1, 1, 1])
		h_conv8_bn = batch_norm(h_conv8)
		h_conv8_relu = relu(h_conv8_bn)
		h_upsample8 = upsample(h_conv8_relu, [424, 512])

		# 424 * 512 * 16 -> 424 * 512 * 3
		W_conv9 = weight_variable([3, 3, 16, 3])
		h_conv9 = conv2d(h_upsample8, W_conv9, [1, 1, 1, 1])
		h_conv9_bn = batch_norm(h_conv9)
		h_conv9_relu = relu(h_conv9_bn)

		if l2:
			return l2_norm(h_conv9_relu, 3)
		else:
			return h_conv9_relu



	def net(self, mode = 'training'):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
		
		# params = self.FCNet(self.color)
		# rho_s, alpha = params[:, 0: 3], params[:, 3: 6]

		# rho_d = self.SVNet(self.color)

		if self.normal_ori == 0:
			# normal = self.SVNet(self.color, l2 = True)
			normal = self.SVNet1(self.color, l2 = True)
		else: 
			normal = self.normal


		# viewdir = tf.constant([0.0, 0.0, 1.0])
		# halfdir = (viewdir + self.lightdir) / tf.reduce_sum((viewdir + self.lightdir) ** 2)

		# mat_size = self.size * self.height * self.width
		# mat_shape = [self.size, self.height, self.width, 3, 1]
		# lightdir_mat = tf.reshape(tf.tile(self.lightdir, [mat_size]), mat_shape)
		# viewdir_mat = tf.reshape(tf.tile(viewdir, [mat_size]), mat_shape)
		# halfdir_mat = tf.reshape(tf.tile(halfdir, [mat_size]), mat_shape)

		# normal_mat = tf.expand_dims(normal, 3)

		# # M * H * W * 1
		# nxl = tf.reshape(tf.matmul(normal_mat, lightdir_mat), [self.size, self.height, self.width, 1])
		# nxv = tf.reshape(tf.matmul(normal_mat, viewdir_mat), [self.size, self.height, self.width, 1])
		# cos_delta = tf.reshape(tf.matmul(normal_mat, halfdir_mat), [self.size, self.height, self.width, 1])
		# tan2_delta = tf.tan(tf.acos(cos_delta)) ** 2
		
		# # M * H * W * 3
		# f1 = rho_d / math.pi

		# alpha2 = alpha ** 2
		# sqrt = ((nxl * nxv) ** 0.5)
		# mask0 = tf.cast(tf.equal(alpha2, 0), tf.float32)
		# tmp1 = rho_s / (4 * math.pi) / (alpha2 + mask0) * (1 - mask0)
		# mask0 = tf.cast(tf.equal(sqrt, 0), tf.float32)
		# tmp1 = tf.tile(tf.reshape(tmp1, [self.size, 1, 1, 3]), [1, self.height, self.width, 1]) / (sqrt + mask0) * (1 - mask0)
		# tmp2 = tan2_delta / tf.tile(tf.reshape(alpha2, [self.size, 1, 1, 3]), [1, self.height, self.width, 1])
		# f2 = tmp1 * (tf.exp(-tmp2))

		# I_ = f1 + f2
		# # I_ = rho_d * relu(nxl)


		mask_sum = tf.reduce_sum(self.mask) * 3

		# l1 = tf.reduce_sum(self.mask * tf.abs(self.color - I_)) / mask_sum
		l2 = tf.reduce_sum(self.mask * tf.abs(self.normal - normal)) / mask_sum
		# loss = l1 + l2
		loss = l2
		# loss = l1 + 1 / l2 ** 0.5
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		# accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		# delta = 3.0 / 256.0
		# accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta), tf.float32) * self.mask) / mask_sum

		normalm = normal * self.mask
		# Im_ = I_ * self.mask

		# rho_d = rho_d * self.mask

		return train_step, loss, normalm

		# if mode == 'training':
		# 	return train_step, accuracy, accuracy_3, loss, l1, l2, normalm, Im_, rho_d, rho_s, alpha, f1, f2
		# elif mode == 'testing':
		# 	return accuracy, accuracy_3, loss, normalm, Im_
		# elif mode == 'predicting':
		# 	return normalm, Im_


class DPNet2:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir)
		self.learning_rate = 1e-2

	# material map net
	def MMNet(self, x, k):

		W_conv1 = weight_variable([3, 3, 3, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		W_conv2 = weight_variable([3, 3, 16, 16])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 1, 1, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)

		W_conv3 = weight_variable([3, 3, 16, 16])
		h_conv3 = conv2d(h_conv2_relu, W_conv3, [1, 1, 1, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)

		W_conv4 = weight_variable([3, 3, 16, k])
		h_conv4 = conv2d(h_conv3_relu, W_conv4, [1, 1, 1, 1])

		# M * H * W * 7 -> M * H * W * 1
		h_softmax = softmax(h_conv4)
		h_max = tf.reshape(tf.argmax(h_softmax, 3), [self.size, self.height, self.width, 1])

		# M * H * W * 1 -> M * H * W * k
		x = tf.tile(tf.constant(np.arange(k).reshape(1, 1, 1, k)), [self.size, self.height, self.width, 1])
		return tf.cast(tf.equal(x, tf.cast(h_max, tf.int32)), tf.float32), h_max


	# material params net
	def MPNet(self, x, k):
		
		# 424 * 512 * 3 -> 424 * 512 * 16
		W_conv1 = weight_variable([3, 3, 3, 16])
		h_conv1 = conv2d(x, W_conv1, [1, 1, 1, 1])
		h_conv1_bn = batch_norm(h_conv1)
		h_conv1_relu = relu(h_conv1_bn)

		# 424 * 512 * 16 -> 106 * 128 * 32
		W_conv2 = weight_variable([3, 3, 16, 32])
		h_conv2 = conv2d(h_conv1_relu, W_conv2, [1, 2, 2, 1])
		h_conv2_bn = batch_norm(h_conv2)
		h_conv2_relu = relu(h_conv2_bn)
		h_pool2 = max_pool_2x2(h_conv2_relu)

		# 106 * 128 * 32 -> 27 * 32 * 64
		W_conv3 = weight_variable([3, 3, 32, 64])
		h_conv3 = conv2d(h_pool2, W_conv3, [1, 2, 2, 1])
		h_conv3_bn = batch_norm(h_conv3)
		h_conv3_relu = relu(h_conv3_bn)
		h_pool3 = max_pool_2x2(h_conv3_relu)

		# 27 * 32 * 64 -> 7 * 8 * 128
		W_conv4 = weight_variable([3, 3, 64, 128])
		h_conv4 = conv2d(h_pool3, W_conv4, [1, 2, 2, 1])
		h_conv4_bn = batch_norm(h_conv4)
		h_conv4_relu = relu(h_conv4_bn)
		h_pool4 = max_pool_2x2(h_conv4_relu)

		h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 8 * 128])

		# fc
		W_fc1 = weight_variable([7 * 8 * 128, 1024])
		b_fc1 = weight_variable([1024])
		h_fc1 = relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
		h_fc1_drop = dropout(h_fc1, self.keep_prob)

		W_fc2 = weight_variable([1024, k * 7])
		b_fc2 = weight_variable([k * 7])
		h_fc2 = relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		return tf.reshape(h_fc2, [self.size, k, 7])

	def net(self, mode = 'training', k = 5):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

		normal = self.normal
		material_map, material_label = self.MMNet(self.color, k)
		params = self.MPNet(self.color, k)
		rho_d, rho_s, alpha = params[..., 0: 3], params[..., 3: 6], params[..., 6]
		
		viewdir = tf.constant([0.0, 0.0, 1.0])
		halfdir = (viewdir + self.lightdir) / tf.reduce_sum((viewdir + self.lightdir) ** 2)

		mat_size = self.size * self.height * self.width
		mat_shape = [self.size, self.height, self.width, 3, 1]
		lightdir_mat = tf.reshape(tf.tile(self.lightdir, [mat_size]), mat_shape)
		viewdir_mat = tf.reshape(tf.tile(viewdir, [mat_size]), mat_shape)
		halfdir_mat = tf.reshape(tf.tile(halfdir, [mat_size]), mat_shape)


		normal_mat = tf.expand_dims(normal, 3)

		# M * H * W * 1
		nxl = tf.reshape(tf.matmul(normal_mat, lightdir_mat), [self.size, self.height, self.width, 1])
		nxv = tf.reshape(tf.matmul(normal_mat, viewdir_mat), [self.size, self.height, self.width, 1])
		cos_delta = tf.reshape(tf.matmul(normal_mat, halfdir_mat), [self.size, self.height, self.width, 1])
		tan2_delta = tf.tan(tf.acos(cos_delta)) ** 2
		
		# M * H * W * 3
		tile_shape = [1, self.height, self.width, 1, 1]
		rho_d_mat = tf.tile(tf.reshape(rho_d, [self.size, 1, 1, k, 3]), tile_shape)
		rho_s_mat = tf.tile(tf.reshape(rho_s, [self.size, 1, 1, k, 3]), tile_shape)
		alpha_mat = tf.tile(tf.reshape(alpha, [self.size, 1, 1, k, 1]), tile_shape)
		material_map = tf.reshape(material_map, [self.size, self.height, self.width, 1, k])
		material_rho_d = tf.reshape(tf.matmul(material_map, rho_d_mat), [self.size, self.height, self.width, 3])
		material_rho_s = tf.reshape(tf.matmul(material_map, rho_s_mat), [self.size, self.height, self.width, 3])
		material_alpha = tf.reshape(tf.matmul(material_map, alpha_mat), [self.size, self.height, self.width, 1])

		f1 = material_rho_d / math.pi

		alpha2 = material_alpha ** 2
		sqrt = ((nxl * nxv) ** 0.5)
		mask0 = tf.cast(tf.equal(alpha2, 0), tf.float32)
		tmp1 = material_rho_s / (4 * math.pi) / (alpha2 + mask0) * (1 - mask0)
		mask0 = tf.cast(tf.equal(sqrt, 0), tf.float32)
		tmp1 = tmp1 / (sqrt + mask0) * (1 - mask0)
		tmp2 = tan2_delta / alpha2
		f2 = tmp1 * tf.exp(-tmp2)

		I_ = f1 + f2

		mask_sum = tf.reduce_sum(self.mask) * 3

		loss = tf.reduce_sum(self.mask * tf.abs(self.color - I_)) / mask_sum
		train_step = tf.train.GradientDescentOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		delta = 3.0 / 256.0
		accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta), tf.float32) * self.mask) / mask_sum

		labelm = tf.cast(material_label, tf.float32) * self.mask
		Im_ = I_ * self.mask


		if mode == 'training':
			return train_step, accuracy, accuracy_3, loss, labelm, Im_
		elif mode == 'testing':
			return accuracy, accuracy_3, loss, labelm, Im_
		elif mode == 'prediction':
			return labelm, Im_
