import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 



class DPNet:
	def __init__(self, size, height, width):
		self.mode = 'Training'
		self.size = size
		self.height = height
		self.width = width
		self.oc = 16


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
		[x_mean, x_varia] = tf.nn.moments(x, axes = 0)
		offset = 0
		scale = 0.1
		vari_epsl = 0.0001
		## calculate the batch normalization
		return tf.nn.batch_normalization(x, x_mean, x_varia, offset,scale,vari_epsl)

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

		W_conv4 = self.weight_variable([3, 3, self.oc, 3])
		h_conv4 = self.conv2d(h_conv3_relu, W_conv4)
		h_l2_norm = self.l2_norm(h_conv4, 2)

		return h_feature_relu, h_l2_norm

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

	def net(self):

		print('net')


		self.depth = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1])
		self.mask = tf.placeholder(tf.float32, [self.size, self.height, self.width, 1])
		self.color = tf.placeholder(tf.float32, [self.size, self.height, self.width, 3])
		# self.color_concate = tf.placeholder(tf.float32, [1, self.height, self.width, self.size * 3])
		
		lightdir = tf.constant([0.0, 0.0, -1.0])
		viewdir = tf.constant([0.0, 0.0, -1.0])

		mat_size = self.size * self.height * self.width
		mat_shape = [self.size, self.height, self.width, 3, 1]
		lightdir_mat = tf.reshape(tf.tile(lightdir, [mat_size]), mat_shape)
		viewdir_mat = tf.reshape(tf.tile(viewdir, [mat_size]), mat_shape)

		# PS Net
		# normal: H * W * 3
		feature, normal = self.PSNet(self.color)
		
		normal_mat = tf.expand_dims(normal, 3)
		nxl = tf.reshape(tf.matmul(normal_mat, lightdir_mat), [self.size, self.height, self.width, 1])
		mat_t = tf.expand_dims(nxl * normal * 2 - lightdir, 3)
		specular = tf.reshape(tf.matmul(mat_t, viewdir_mat), [self.size, self.height, self.width, 1])

		# IR Net
		# BRDF: M * H * W * 3
		BRDF = self.IRNet(tf.concat([self.color, specular], 3), feature)


		I_ = BRDF * self.relu(nxl)

		# TODO: loss
		loss_res = tf.reduce_sum(self.mask * (self.color - I_)) / (self.size * tf.reduce_sum(self.mask) * 3)
		# loss_prior = 
		train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_res)

		# TODO: prediction
		prediction = tf.equal(self.color, I_)
		accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


		return train_step, accuracy


def readData():

	# depth: M * W * H * 1, color: M * W * H * 3

	print('load data')

	datapath = '../DynamicProjectionData/capture_data_' + '2/'

	number = 50

	depth = np.array([np.load(datapath + 'depth{}.npy'.format(0))])
	color = np.array([np.load(datapath + 'color{}.npy'.format(0))])
	mask = np.array([np.load(datapath + 'mask{}.npy'.format(0))], np.bool)
	# colori = np.load(datapath + 'color{}.npy'.format(0))
	# color = np.array([colori])
	# color_concate = np.array(colori)

	for i in range(1, number):
		depth = np.append(depth, [np.load(datapath + 'depth{}.npy'.format(i))], axis = 0)
		color = np.append(color, [np.load(datapath + 'color{}.npy'.format(i))], axis = 0)
		mask = np.append(mask, [np.load(datapath + 'mask{}.npy'.format(i))], axis = 0)
		# colori = np.load(datapath + 'color{}.npy'.format(i))
		# color = np.append(color, [colori])
		# color_concate = np.append(color_concate, colori, axis = 2)

	depth = np.expand_dims(depth, axis = 3)
	mask = np.expand_dims(mask, axis = 3)

	return depth, color, mask


def train():

	print('train')
	
	depth, color, mask = readData()
	[size, height, width] = depth.shape[0: 3]

	batch_size = 1

	model = DPNet(batch_size, height, width)

	with tf.Session() as sess:

		train_step, accuracy = model.net()

		sess.run(tf.global_variables_initializer())

		for i in range(20000):
			idx = i % 40

			if i % 100 == 0:
				train_accuracy = sess.run(accuracy, feed_dict = {model.depth: depth[idx: idx + batch_size], model.color: color[idx: idx + batch_size], model.mask: mask[idx: idx + batch_size]})
				print("step {}, training accuracy {}".format(i, train_accuracy))
			sess.run(train_step, feed_dict = {model.depth: depth[idx: idx + batch_size], model.color: color[idx: idx + batch_size], model.mask: mask[idx: idx + batch_size]})



if __name__ == '__main__':
	train()