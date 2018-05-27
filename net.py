import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data



class DPNet:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir)
		self.oc = 16
		self.learning_rate = 1e-2


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	def relu(self, x):
		return tf.nn.relu(x)

	def l2_norm(self, x, dim):
		return tf.nn.l2_normalize(x, axis = dim)

	def batch_norm(self, x):
		[x_mean, x_varia] = tf.nn.moments(x, axes = [0, 1, 2])
		offset = 0
		scale = 0.1
		vari_epsl = 0.0001
		## calculate the batch normalization
		return tf.nn.batch_normalization(x, x_mean, x_varia, offset, scale, vari_epsl)

	def PSNet(self, x):

		W_conv1 = self.weight_variable([3, 3, 3, self.oc])
		h_conv1 = self.conv2d(x, W_conv1)
		h_conv1_bn = self.batch_norm(h_conv1)
		h_conv1_relu = self.relu(h_conv1_bn)

		W_conv2 = self.weight_variable([3, 3, self.oc, self.oc])
		h_conv2 = self.conv2d(h_conv1_relu, W_conv2)
		h_conv2_bn = self.batch_norm(h_conv2)
		h_conv2_relu = self.relu(h_conv2_bn)

		W_conv3 = self.weight_variable([3, 3, self.oc, self.oc])
		h_conv3 = self.conv2d(h_conv2_relu, W_conv3)
		h_conv3_bn = self.batch_norm(h_conv3)
		h_conv3_relu = self.relu(h_conv3_bn)

		W_feature = self.weight_variable([3, 3, self.oc, int(self.oc / 2)])
		h_feature = self.conv2d(h_conv2_relu, W_feature)
		h_feature_bn = self.batch_norm(h_feature)
		h_feature_relu = self.relu(h_feature_bn)

		# 0: train, 1: self.normal
		if self.normal_ori == 1:
			return h_feature_relu, self.normal, 0

		W_conv4 = self.weight_variable([3, 3, self.oc, 3])
		h_conv4 = self.conv2d(h_conv3_relu, W_conv4)
		h_l2_norm = self.l2_norm(h_conv4, 3)

		return h_feature_relu, h_l2_norm, tf.reduce_sum(self.mask * (self.normal - h_l2_norm) ** 2)


	def IRNet(self, x, feature):

		# add specular term to x

		W_conv1 = self.weight_variable([3, 3, 4, 16])
		h_conv1 = self.conv2d(x, W_conv1)
		h_conv1_bn = self.batch_norm(h_conv1)
		h_conv1_relu = self.relu(h_conv1_bn)

		W_conv2 = self.weight_variable([3, 3, 16, 16])
		h_conv2 = self.conv2d(h_conv1_relu, W_conv2)
		h_conv2_bn = self.batch_norm(h_conv2)
		h_conv2_relu = self.relu(h_conv2_bn)

		W_conv3 = self.weight_variable([3, 3, 16, 16])
		h_conv3 = self.conv2d(h_conv2_relu, W_conv3)
		h_conv3_bn = self.batch_norm(h_conv3)
		h_conv3_relu = self.relu(h_conv3_bn)

		h_concat = tf.concat([h_conv3_relu, feature], 3)

		W_conv4 = self.weight_variable([1, 1, 16 + int(self.oc / 2), 16])
		h_conv4 = self.conv2d(h_concat, W_conv4)
		h_conv4_bn = self.batch_norm(h_conv4)
		h_conv4_relu = self.relu(h_conv4_bn)

		W_conv5 = self.weight_variable([3, 3, 16, 3])
		h_conv5 = self.conv2d(h_conv4_relu, W_conv5)
		h_conv5_bn = self.batch_norm(h_conv5)
		h_conv5_relu = self.relu(h_conv5_bn)

		W_conv6 = self.weight_variable([3, 3, 3, 3])
		h_conv6 = self.conv2d(h_conv5_relu, W_conv6)

		return h_conv6


	def net(self, mode = 'training'):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.lamda = tf.placeholder(tf.float32, name = 'lamda')
		# self.color_concate = tf.placeholder(tf.float32, [1, self.height, self.width, self.size * 3])
		
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
		# BRDF: M * H * W * 3
		BRDF = self.IRNet(tf.concat([self.color, specular], 3), feature)


		I_ = BRDF * self.relu(nxl)

		lr = tf.reduce_sum(self.mask * tf.abs(self.color - I_))
		mask_sum = tf.reduce_sum(self.mask) * 3

		loss_res = lr / mask_sum
		loss_prior = lp / mask_sum
		loss = loss_res + self.lamda * loss_prior
		train_step = tf.train.AdamOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		delta = 3.0 / 256.0
		accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta), tf.float32) * self.mask) / mask_sum

		normalm = normal * self.mask
		BRDFm = BRDF * self.mask
		Im_ = I_ * self.mask

		if mode == 'training':
			return train_step, accuracy, accuracy_3, loss, BRDFm, Im_, loss_res, loss_prior
		else:
			return accuracy, accuracy_3, loss, normalm, BRDFm, Im_, loss_res, loss_prior


class DPNet1:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir)
		self.oc = 16
		self.learning_rate = 1e-2


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	def relu(self, x):
		return tf.nn.relu(x)

	def l2_norm(self, x, dim):
		return tf.nn.l2_normalize(x, axis = dim)

	def batch_norm(self, x):
		[x_mean, x_varia] = tf.nn.moments(x, axes = [0, 1, 2])
		offset = 0
		scale = 0.1
		vari_epsl = 0.0001
		## calculate the batch normalization
		return tf.nn.batch_normalization(x, x_mean, x_varia, offset, scale, vari_epsl)


	def net(self, mode = 'training'):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.lamda = tf.placeholder(tf.float32, name = 'lamda')
		# self.color_concate = tf.placeholder(tf.float32, [1, self.height, self.width, self.size * 3])
		
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
		# BRDF: M * H * W * 3
		BRDF = self.IRNet(tf.concat([self.color, specular], 3), feature)


		I_ = BRDF * self.relu(nxl)

		lr = tf.reduce_sum(self.mask * tf.abs(self.color - I_))
		mask_sum = tf.reduce_sum(self.mask) * 3

		loss_res = lr / mask_sum
		loss_prior = lp / mask_sum
		loss = loss_res + self.lamda * loss_prior
		train_step = tf.train.AdamOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		delta = 3.0 / 256.0
		accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta), tf.float32) * self.mask) / mask_sum

		normalm = normal * self.mask
		Im_ = I_ * self.mask

		# rename
		accuracy = tf.identity(accuracy, name = 'accuracy')
		accuracy_3 = tf.identity(accuracy_3, name = 'accuracy_3')
		loss = tf.identity(loss, name = 'loss')
		BRDF = tf.identity(BRDF, name = 'BRDF')
		Im_ = tf.identity(Im_, name = 'Ipre')

		if mode == 'training':
			return train_step, accuracy, accuracy_3, loss, BRDF, Im_, loss_res, loss_prior
		else:
			return accuracy, accuracy_3, loss, normalm, BRDF, Im_, loss_res, loss_prio


class DPNet1:
	def __init__(self, size, height, width, normal_ori, lightdir = [0.0, 0.0, 1.0]):
		self.size = size
		self.height = height
		self.width = width
		self.normal_ori = normal_ori
		self.lightdir = tf.constant(lightdir)
		self.oc = 16
		self.learning_rate = 1e-2


	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	def relu(self, x):
		return tf.nn.relu(x)

	def l2_norm(self, x, dim):
		return tf.nn.l2_normalize(x, axis = dim)

	def batch_norm(self, x):
		[x_mean, x_varia] = tf.nn.moments(x, axes = [0, 1, 2])
		offset = 0
		scale = 0.1
		vari_epsl = 0.0001
		## calculate the batch normalization
		return tf.nn.batch_normalization(x, x_mean, x_varia, offset, scale, vari_epsl)


	def net(self, mode = 'training'):

		print('net')

		self.normal = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'normal')
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3], name = 'color')
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1], name = 'mask')
		self.lamda = tf.placeholder(tf.float32, name = 'lamda')
		# self.color_concate = tf.placeholder(tf.float32, [1, self.height, self.width, self.size * 3])
		
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
		# BRDF: M * H * W * 3
		BRDF = self.IRNet(tf.concat([self.color, specular], 3), feature)


		I_ = BRDF * self.relu(nxl)

		lr = tf.reduce_sum(self.mask * tf.abs(self.color - I_))
		mask_sum = tf.reduce_sum(self.mask) * 3

		loss_res = lr / mask_sum
		loss_prior = lp / mask_sum
		loss = loss_res + self.lamda * loss_prior
		train_step = tf.train.AdamOptimizer(self.learning_rate, name = 'train_step').minimize(loss)

		accuracy = tf.reduce_sum(tf.cast(tf.equal(self.color, I_), tf.float32) * self.mask) / mask_sum
		delta = 3.0 / 256.0
		accuracy_3 = tf.reduce_sum(tf.cast(tf.less(tf.abs(self.color - I_), delta), tf.float32) * self.mask) / mask_sum

		normalm = normal * self.mask
		Im_ = I_ * self.mask

		# rename
		accuracy = tf.identity(accuracy, name = 'accuracy')
		accuracy_3 = tf.identity(accuracy_3, name = 'accuracy_3')
		loss = tf.identity(loss, name = 'loss')
		BRDF = tf.identity(BRDF, name = 'BRDF')
		Im_ = tf.identity(Im_, name = 'Ipre')

		if mode == 'training':
			return train_step, accuracy, accuracy_3, loss, BRDF, Im_, loss_res, loss_prior
		else:
			return accuracy, accuracy_3, loss, normalm, BRDF, Im_, loss_res, loss_prio
