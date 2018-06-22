from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2 as cv
import numpy as np
import tensorflow as tf
from gl.glrender import GLRenderer
import gl.glm as glm
import ctypes
import copy
import time
from numpy.linalg import inv
import os
from net import DPNet

MODE = 2
# 0: record new background and capture new data by Kinect
# 1: use background data, but capture new data by Kinect
# 2: use data for all, no Kinect

SAVE = False
SAVEALL = False
REALTIME = False

DATAPATH = '../DynamicProjectionData/'
SUB = 'data/data_body/'
SUBIN = 'data/data_body_0523/0/'
SUBOUT = 'data/data_body_0524_pig_0610_3/'

class DynamicProjection(object):
	def __init__(self):
		print("Start initializing dp...")

		self.time = time.time()
		self.index = 0

		# init kinect
		self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)

		self.znear = 0.2
		self.zfar = 3.0


		# init camera and projector params
		self.cx = 263.088
		self.cy = 207.137
		self.fx = 365.753
		self.fy = 365.753
		
		self.t = [ 
            -7.517246646433162,
			0.43210832349956646,
			-3.6190140901470293,
			0.9668409943035755,
			-0.15296828300679596,
			10.71404717765025,
			-4.1966722473037015,
			1.5030783569335278,
			-0.07908378457766652,
			0.996679151785893,
			-6.519932108201286

		]

		self.pwidth = 1024
		self.pheight = 768

		tex = cv.imread(DATAPATH + 'texture3.png')
		self.render = GLRenderer(b'projection', (self.pwidth, self.pheight), tex)

		self.mvp = glm.ortho(-1, 1, -1, 1, -1, 1000)

		self.depthback_origin = np.zeros([424 * 512], np.float32)
		self.colorback_origin = np.zeros([1080 * 1920 * 4], np.float32)

		self.depthback = np.zeros([424, 512, 1], np.uint8)
		self.colorback = np.zeros([424, 512, 4], np.uint8)

		#  init camera
		
		self.cap = cv.VideoCapture(0)  
		self.cap.set(3, 1280)
		self.cap.set(4, 960)
		self.cap.set(10, 0.0)		# brightness
		self.cap.set(11, 0.0)
		self.cap.set(12, 58.0)
		self.cap.set(13, 0.0)
		self.cap.set(15, -6.0)		# exposure
		self.cap.set(17, 4600)


	def c2d(self, rawdepth, rawcolor):
		# map color frame to depth space

		rgbd = np.zeros([424 * 512, 4], np.uint8)

		color = rawcolor.reshape([1080 * 1920, 4])
		dsp = np.zeros([1080 * 1920 * 2], np.float32)
		self.kinect._mapper.MapColorFrameToDepthSpace(
			424 * 512, 
			rawdepth.ctypes.data_as(ctype.POINTER(ctypes.c_ushort)), 
			1080 * 1920, 
			dsp.ctypes.data_as(ctypes.POINTER(_DepthSpacePoint)))

		x, y = dsp[::2], dsp[1::2]
		infmask = np.logical_or(np.isinf(x), np.isinf(y))
		x[infmask], y[infmask] = -1, -1
		x, y = (x + 0.5).astype(np.int32), (y + 0.5).astype(np.int32)
		validmask = np.logical_and(np.logical_and(x >= 0, x < 512), np.logical_and(y >= 0, y < 424))
		x, y = x[validmask], y[validmask]

		# new rgbd with size 424 * 512
		rgbd[y * 512 + x, :3] = color[validmask, :3]

		return rgbd

	def d2c(self, rawdepth, rawcolor):
		# map depth frame to color space

		rgbd = np.zeros([424 * 512, 4], np.uint8)

		color = rawcolor.reshape([1080, 1920, 4])
		csp = np.zeros([424 * 512 * 2], np.float32)
		self.kinect._mapper.MapDepthFrameToColorSpace(
			424 * 512, 
			rawdepth.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), 
			424 * 512, 
			csp.ctypes.data_as(ctypes.POINTER(_ColorSpacePoint)))

		x, y = csp[::2], csp[1::2]
		infmask = np.logical_or(np.isinf(x), np.isinf(y))
		x[infmask], y[infmask] = -1, -1
		x, y = (x + 0.5).astype(np.int32), (y + 0.5).astype(np.int32)
		validmask = np.logical_and(np.logical_and(x >= 0, x < 1920), np.logical_and(y >= 0, y < 1080))
		x, y = x[validmask], y[validmask]

		# new rgbd with size 424 * 512
		rgbd[validmask, :3] = color[y, x, :3]

		return rgbd

	def depth2gray(self, rawdepth, part = False):
		depth = copy.copy(rawdepth)
		# for those has no depth data, set as far
		if part:
			depth[self.depthback_origin - depth < 50] = self.zfar * 1000
		depth[depth == 0] = self.zfar * 1000
		depth = depth / 1000
		# for those out of range, set as far
		depth[depth > self.zfar] = self.zfar
		depth[depth < self.znear] = self.zfar
		depth = ((self.zfar - depth) / (self.zfar - self.znear) * 255).astype(np.uint8)

		return depth

	def preprocess(self, rawdepth, rawcolor):
		rgbd = np.zeros([424 * 512, 4], np.uint8)

		d2c = True

		if d2c:

			# map depth frame to color space
			if MODE < 2: 
				rgbd = self.d2c(rawdepth, rawcolor)
				if SAVE:
					np.save(DATAPATH + SUB + "rgbd.npy", rgbd)
			else: 
				rgbd = np.load(DATAPATH + SUBIN + "rgbd.npy")
				rgbd = rgbd.reshape([424 * 512, 4])

		else:

			# map color frame to depth sapce
			if MODE < 2:
				rgbd = self.c2d(rawdepth, rawcolor)
				if SAVE: 
					np.save(DATAPATH + SUB + "rgbd.npy", rgbd)
			else: 
				rgbd = np.load(DATAPATH + SUBIN + "rgbd.npy")
				rgbd = rgbd.reshape([424 * 512, 4])


		# trun raw depth into gray image
		# with all the objects in it
		# black (0) means background
		rgbd[:, 3] = self.depth2gray(rawdepth)

		# trun raw depth into gray image
		# remove the background and other stable object
		depth_part = self.depth2gray(rawdepth, True)


		rgbd = rgbd.reshape([424, 512, 4])
		depth_part = depth_part.reshape([424, 512])
		return rgbd, depth_part


	def recordBackground(self, num_frame):
		print("Record {} frames as background...".format(num_frame))

		if MODE == 0:

			depth_cnt = np.zeros([424 * 512], np.uint8)
			color_cnt = np.zeros([1080 * 1920 * 4], np.uint8)

			for i in range(num_frame):
				while 1:
					x = 0
					if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame():
						depth_frame = self.kinect.get_last_depth_frame()
						color_frame = self.kinect.get_last_color_frame()
						depth_cnt[depth_frame > 0] += 1
						color_cnt[color_frame > 0] += 1
						self.depthback_origin += depth_frame
						self.colorback_origin += color_frame
						break

			depth_mask = depth_cnt > 0
			self.depthback_origin[depth_mask] /= depth_cnt[depth_mask]
			color_mask = color_cnt > 0
			self.colorback_origin[color_mask] /= color_cnt[color_mask]

			rgbd = self.d2c(self.depthback_origin, self.colorback_origin)
			rgbd[:, 3] = self.depth2gray(self.depthback_origin)

			back = rgbd.reshape([424, 512, 4])
			self.depthback = back[:, :, 3]
			self.colorback = back[:, :, 0: 3]

			if SAVE:
				np.save(DATAPATH + SUB + 'depthback_origin.npy', self.depthback_origin)
				np.save(DATAPATH + SUB + 'colorback_origin.npy', self.colorback_origin)

		else: 

			self.depthback_origin = np.load(DATAPATH + SUBIN + 'depthback_origin.npy')
			self.colorback_origin = np.load(DATAPATH + SUBIN + 'colorback_origin.npy')


	def depth2normal(self, rawdepth, mask):
		proj = np.zeros([424, 512, 3], np.float32)

		rawdepth = rawdepth.reshape([424, 512])[mask]
		u = np.array([[i for i in range(511, -1, -1)]] * 424)[mask]
		v = np.array([[i for j in range(512)] for i in range(424)])[mask]

		Z = rawdepth / 1000
		X = (u - self.cx) * Z / self.fx
		Y = (self.cy - v) * Z / self.fy

		t = self.t
		denom = t[8] * X + t[9] * Y + t[10] * Z + 1
		x = (t[0] * X + t[1] * Y + t[2] * Z + t[3]) / denom * 2 - 1
		y = 1 - (t[4] * X + t[5] * Y + t[6] * Z + t[7]) / denom * 2

		proj[mask, 0], proj[mask, 1], proj[mask, 2] = x, y, denom

		kernel_cul = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype = np.uint8)
		cul = cv.erode(mask.astype(np.uint8), kernel_cul, iterations = 1, borderValue = 0)
		cul_cmask = cul.astype(np.bool)
		cul_umask = cv.filter2D(cul, -1, np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)
		cul_lmask = cv.filter2D(cul, -1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)

		kernel_cdr = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype = np.uint8)
		cdr = cv.erode(mask.astype(np.uint8), kernel_cdr, iterations = 1, borderValue = 0)
		cdr_cmask = cdr.astype(np.bool)
		cdr_dmask = cv.filter2D(cdr, -1, np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)
		cdr_rmask = cv.filter2D(cdr, -1, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)

		num = np.sum(cul) + np.sum(cdr)

		p0 = np.concatenate([proj[cul_cmask], proj[cdr_cmask]], 0)
		p2 = np.concatenate([proj[cul_umask], proj[cdr_dmask]], 0)
		p1 = np.concatenate([proj[cul_lmask], proj[cdr_rmask]], 0)

		position = np.array([[[i, j] for j in range(512)] for i in range(424)])
		pos0 = np.concatenate([position[cul_cmask], position[cdr_cmask]], 0)
		pos1 = np.concatenate([position[cul_umask], position[cdr_dmask]], 0)
		pos2 = np.concatenate([position[cul_lmask], position[cdr_rmask]], 0)

		# surface normal
		surface_normals = np.zeros([num, 3], np.float32)
		surface_normals = np.cross(p1 - p0, p2 - p1)

		vertex_normals = np.zeros((424, 512, 3), np.float32)
		for i in range(num):
			vertex_normals[pos0[i, 0], pos0[i, 1], 0: 3] += surface_normals[i]
			vertex_normals[pos1[i, 0], pos1[i, 1], 0: 3] += surface_normals[i]
			vertex_normals[pos2[i, 0], pos2[i, 1], 0: 3] += surface_normals[i]

		norm = np.expand_dims(np.linalg.norm(vertex_normals, axis = 2), 2)
		norm[norm == 0.0] = 1.0
		vertex_normals = vertex_normals / norm

		vertex_normals[..., 0] = 0 - vertex_normals[..., 0]

		return vertex_normals


	def projectLight(self):

		vertices = np.array([[-1, -1, 0], [3, -1, 0], [-1, 3, 0]], np.float32)
		colors = np.ones([3, 3], np.float32) * 0.5

		self.render.draw(vertices, colors, None, None, self.mvp.T, 0)


	def project(self, rawdepth, corres, mask, normal_ori_i, pre_normal, pre_reflect):
		proj = np.zeros([424, 512, 3], np.float32)

		rawdepth = rawdepth.reshape([424, 512])[mask]
		u = np.array([[i for i in range(511, -1, -1)]] * 424)[mask]
		v = np.array([[i for j in range(512)] for i in range(424)])[mask]

		Z = rawdepth / 1000
		X = (u - self.cx) * Z / self.fx
		Y = (self.cy - v) * Z / self.fy

		t = self.t
		denom = t[8] * X + t[9] * Y + t[10] * Z + 1
		x = (t[0] * X + t[1] * Y + t[2] * Z + t[3]) / denom * 2 - 1
		y = 1 - (t[4] * X + t[5] * Y + t[6] * Z + t[7]) / denom * 2

		proj[mask, 0], proj[mask, 1], proj[mask, 2] = x, y, denom

		kernel_cul = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype = np.uint8)
		cul = cv.erode(mask.astype(np.uint8), kernel_cul, iterations = 1, borderValue = 0)
		cul_cmask = cul.astype(np.bool)
		cul_umask = cv.filter2D(cul, -1, np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)
		cul_lmask = cv.filter2D(cul, -1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)

		kernel_cdr = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype = np.uint8)
		cdr = cv.erode(mask.astype(np.uint8), kernel_cdr, iterations = 1, borderValue = 0)
		cdr_cmask = cdr.astype(np.bool)
		cdr_dmask = cv.filter2D(cdr, -1, np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)
		cdr_rmask = cv.filter2D(cdr, -1, np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]), borderType = cv.BORDER_CONSTANT).astype(np.bool)

		num = np.sum(cul) + np.sum(cdr)

		p0 = np.concatenate([proj[cul_cmask], proj[cdr_cmask]], 0)
		p2 = np.concatenate([proj[cul_umask], proj[cdr_dmask]], 0)
		p1 = np.concatenate([proj[cul_lmask], proj[cdr_rmask]], 0)

		vertices = np.zeros([num * 3, 3], np.float32)
		vertices[0::3, :], vertices[1::3, :], vertices[2::3, :] = p0, p1, p2


		position = np.array([[[i, j] for j in range(512)] for i in range(424)])
		pos0 = np.concatenate([position[cul_cmask], position[cdr_cmask]], 0)
		pos1 = np.concatenate([position[cul_umask], position[cdr_dmask]], 0)
		pos2 = np.concatenate([position[cul_lmask], position[cdr_rmask]], 0)

		if normal_ori_i == 1:
			# calculate surface normal
			surface_normals = np.zeros([num, 3], np.float32)
			surface_normals = np.cross(p1 - p0, p2 - p1)

			vertex_normals = np.zeros((424, 512, 3), np.float32)
			for i in range(num):
				vertex_normals[pos0[i, 0], pos0[i, 1], 0: 3] += surface_normals[i]
				vertex_normals[pos1[i, 0], pos1[i, 1], 0: 3] += surface_normals[i]
				vertex_normals[pos2[i, 0], pos2[i, 1], 0: 3] += surface_normals[i]
		else:
			vertex_normals = pre_normal
		n0 = np.concatenate([vertex_normals[cul_cmask], vertex_normals[cdr_cmask]], 0)
		n2 = np.concatenate([vertex_normals[cul_umask], vertex_normals[cdr_dmask]], 0)
		n1 = np.concatenate([vertex_normals[cul_lmask], vertex_normals[cdr_rmask]], 0)
		normals = np.zeros((num * 3, 3), np.float32)
		normals[0::3, :], normals[1::3, :], normals[2::3, :] = n0, n1, n2

		pre_reflect = pre_reflect[..., ::-1]
		r0 = np.concatenate([pre_reflect[cul_cmask], pre_reflect[cdr_cmask]], 0)
		r2 = np.concatenate([pre_reflect[cul_umask], pre_reflect[cdr_dmask]], 0)
		r1 = np.concatenate([pre_reflect[cul_lmask], pre_reflect[cdr_rmask]], 0)
		reflects = np.zeros((num * 3, 3), np.float32)
		reflects[0::3, :], reflects[1::3, :], reflects[2::3, :] = r0, r1, r2

		corres = corres[..., ::-1]
		c0 = np.concatenate([corres[cul_cmask], corres[cdr_cmask]], 0)
		c2 = np.concatenate([corres[cul_umask], corres[cdr_dmask]], 0)
		c1 = np.concatenate([corres[cul_lmask], corres[cdr_rmask]], 0)
		c0, c1, c2 = c0 / 255, c1 / 255, c2 / 255
		colors = np.zeros([num * 3, 3], np.float32)
		colors[0::3, :], colors[1::3, :], colors[2::3, :] = c0, c1, c2

		# # normal to color
		# norm = np.expand_dims(np.linalg.norm(vertex_normals, axis = 2), 2)
		# norm[norm == 0] = 1.0
		# vertex_normals = vertex_normals / norm
		# vertex_normals = ((vertex_normals + 1) / 2 * 255).astype(np.uint8)
		# cv.imshow('norm', vertex_normals)

		self.render.draw(vertices, colors, normals, reflects, self.mvp.T)


	def getRawData(self):
		rawdepth = np.load(DATAPATH + SUBIN + 'rawdepth.npy')
		rawcolor = np.load(DATAPATH + SUBIN + 'rawcolor.npy')
		rawinfrared = np.load(DATAPATH + SUBIN + 'rawinfrared.npy')
		cameraColor = np.load(DATAPATH + SUBIN + 'cameraColor.npy')

		return True, rawdepth, rawcolor, rawinfrared, cameraColor


	def getRawDataWithKinect(self, save):

		flag = False

		rawdepth = np.zeros((424 * 512, 1))
		rawcolor = np.zeros((1080 * 1092 * 4))
		rawinfrared = np.zeros((424 * 512, 1))
		cameraColor = np.zeros((1280 * 960 * 3))

		if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame() and self.kinect.has_new_infrared_frame():
			
			rawdepth = self.kinect.get_last_depth_frame()
			rawcolor = self.kinect.get_last_color_frame()
			rawinfrared = self.kinect.get_last_infrared_frame()

			if save:
				np.save(DATAPATH + SUB + 'rawdepth.npy', rawdepth)
				np.save(DATAPATH + SUB + 'rawcolor.npy', rawcolor)
				np.save(DATAPATH + SUB + 'rawinfrared.npy', rawinfrared)

			flag = True


		ret, cameraColor = self.cap.read()
		if save:
			np.save(DATAPATH + SUB + 'cameraColor.npy', cameraColor)


		return flag, rawdepth, rawcolor, rawinfrared, cameraColor

	def calculateRT(self, sub = '_ll/'):
		path = '../DynamicProjectionData/capture_images'
		path = path + sub + 'good/npy/'

		R_ir = np.load(path + 'R_ir.npy')
		T_ir = np.load(path + 'T_ir.npy')
		R_rgb = np.load(path + 'R_rgb.npy')
		T_rgb = np.load(path + 'T_rgb.npy')
		H_ir = np.load(path + 'H_ir.npy')
		H_rgb = np.load(path + 'H_rgb.npy')

		R_plus = np.array([
			[1, 0, 0], 
			[0, 1, 0], 
			[0, 0, 1]		
			])
		T_plus = np.array([0, 0, 0])

		if os.path.isfile(path + 'R_plus.npy') and os.path.isfile(path + 'T_plus.npy'):
			R_plus = np.load(path + 'R_plus.npy')
			T_plus = np.load(path + 'T_plus.npy')

		R = np.matmul(R_rgb, inv(R_ir))
		T = T_rgb - np.matmul(R, T_ir)

		R_final = np.matmul(H_rgb, np.matmul(R, inv(H_ir)))
		T_final = np.matmul(H_rgb, T)

		R_final = np.matmul(R_final, R_plus)
		T_final = T_final + T_plus

		return R_final, T_final

	def calibrateKinectCamera(self, R, T, base_p_irs, camera, depth, infrared):

		depth = np.flip(depth.reshape([424, 512]), 1)
		depth = depth.reshape([424 * 512])

		infrared = infrared.reshape((424, 512))
		infrared = self.depth2gray(infrared)
		infrared = np.flip(255 - infrared, 1)

		cali = np.zeros((424, 512, 3), np.uint8)
		base_depth = np.zeros((424 * 512, 3), np.float32)
		for i in range(3):
			cali[:, :, i] = copy.copy(infrared)
			base_depth[:, i] = depth

		p_irs = np.transpose(base_p_irs * base_depth)
		p_rgbs = np.transpose(np.matmul(R, p_irs)) + T
		
		x, y = (p_rgbs[:, 0] / p_rgbs[:, 2]).astype(int), (p_rgbs[:, 1] / p_rgbs[:, 2]).astype(int)
		mask = np.logical_and(np.logical_and(x >= 0, x < camera.shape[0]), np.logical_and(y >= 0, y < camera.shape[1]))
		x, y = x[mask], y[mask]
			
		mask = mask.reshape([424, 512])
		cali[mask] = camera[x, y]

		return cali


	def colorCalibration(self):

		# idx = 0
		# idx = -10
		idx = 502

		while idx < 0:
			idx += 1
			self.getRawDataWithKinect(False)

		while idx < 256 * 3:
			ch = cv.waitKey(1)
			if ch == 27:
				break

			print(idx)

			c = 'r'
			if idx >= 256:
				c = 'g'
			if idx >= 512:
				c = 'b'

			if MODE < 2:
				flag, rawdepth, rawcolor, cameraColor = self.getRawDataWithKinect(SAVE)  
			else:
				flag, rawdepth, rawcolor, cameraColor = self.getRawData()

			if flag:

				rgbd, depth_part = self.preprocess(rawdepth, rawcolor)
				depth = rgbd[:, :, 3]
				color = rgbd[:, :, 0: 3]
				mask = depth_part >  0
				mask[120: 290, 205: 335] = True
				corres = np.zeros([424, 512, 3], np.uint8)
				corres[mask] = np.array([255, 255, 255])

				# projection
				image = cv.imread('{}calibration_color/im_{}_{}.bmp'.format(DATAPATH, c, idx % 256))
				cv.imshow('image', image)
				# bgr to rgb
				image = image[..., ::-1]

				x0, y0, x1, y1 = 120, 205, 290, 335
				w, h = image.shape[0], image.shape[1]
				corres[x0: x1, y0: y1] = np.array([[image[i - x0, j - y0] for j in range(y0, y1)] for i in range(x0, x1)])
				# np.save('{}corres/corres_{}_{}.npy'.format(DATAPATH, c, idx % 256), corres)

				self.project(rawdepth, corres, mask)

			t = 2
			if idx == 0:
				t = 5

			# wait 2 seconds
			t0 = time.time()
			while time.time() - t0 < t:
				_ = 0

			# capture image for 1 second
			t0 = time.time()
			while time.time() - t0 < 1:
				if MODE < 2:
					ret, cameraColor = self.cap.read()
				else:
					cameraColor = np.load(DATAPATH + SUB + 'cameraColor.npy')
					
				# cv.imshow('color', cameraColor)
				cv.imwrite('{}capture_color/capture_{}_{}.png'.format(DATAPATH, c, idx % 256), cameraColor)
				# cv.imwrite('color{}.bmp'.format(idx), cameraColor)

			# wait 1 second
			t0 = time.time()
			while time.time() - t0 < 1:
				_ = 0


			idx = idx + 1

	def captureTrainData(self, gid, gnum, gdelay, cd_dirname):
		path = DATAPATH + cd_dirname
		while os.path.isfile(path + 'a_image_{}_color.png'.format(gid * gnum)):
			gid += 1
		rawdepth_b = self.depthback_origin.reshape([424, 512])
		np.save(path + 'rawdepth_b_{}.npy'.format(gid), rawdepth_b)
		# rawdepth_b = np.load(path + 'rawdepth_b_0.npy')

		RED = np.array([0, 0, 255], np.uint8)
		YELLOW = np.array([0, 255, 255], np.uint8)
		GREEN = np.array([0, 255, 0], np.uint8)

		hint = np.zeros([256, 256, 3], np.uint8)
		hint[:, :] = GREEN
		cv.imshow('hint', hint)
		cv.waitKey(3000)

	
		print('capture data')
		idx = start = gid * gnum
		num = idx + gnum + gdelay

		while idx < num:
			print(idx)

			t0 = time.time()
			while time.time() - t0 < 5:
				cv.imshow('hint', hint)
				cv.waitKey(1)

				if time.time() - t0 >= 2:
					hint[:, :] = YELLOW
				if time.time() - t0 >= 3:
					hint[:, :] = RED
				if MODE < 2:
					flag, rawdepth, rawcolor, rawinfrared, cameraColor = self.getRawDataWithKinect(False)
				else:
					flag, rawdepth, rawcolor, rawinfrared, cameraColor = self.getRawData()
				if flag:

					mask = np.abs(rawdepth.reshape([424, 512]) - rawdepth_b) > 50
					self.projectLight()

					cv.imshow('camera', cameraColor)
					cv.waitKey(1)

					cv.imwrite(path + 'a_image_{}_color.png'.format(idx), rawcolor.reshape([1080, 1920, 4])[:, :, 0: 3])
					cv.imwrite(path + 'a_image_{}_camera.png'.format(idx), cameraColor)

					np.save(path + 'camera{}.npy'.format(idx), cameraColor)
					np.save(path + 'color{}.npy'.format(idx), rawcolor)
					np.save(path + 'infrared{}.npy'.format(idx), rawinfrared)
					np.save(path + 'depth{}.npy'.format(idx), rawdepth)

			idx += 1
			hint[:, :] = GREEN



	def filter(self, depth, selection = 0):
		if selection == 0:
			return depth
		elif selection == 1:
			kernel = np.ones((5, 5), np.float32) / 25
			return cv.filter2D(depth, -1, kernel)
		elif selection == 2:
			return cv.medianBlur(depth, 5)
		elif selection == 3:
			return cv.GaussianBlur(depth, (5, 5), 0)

	def smooth(self, mask, not_mask, rawdepth, depth_part):
		kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
		mask_erode = cv.erode(mask.astype(np.uint8), kernel).astype(np.bool)
		mask_edge = mask ^ mask_erode
		depth_erode = np.zeros((424, 512), np.uint8)
		depth_erode[mask_erode] = depth_part[mask_erode]

		# 1d
		# rawdepth_filter = copy.copy(rawdepth)
		# rawdepth_filter = self.filter(rawdepth_filter, 3)
		# rawdepth_filter = rawdepth_filter.reshape([424, 512])

		# 2d
		rawdepth_filter = copy.copy(rawdepth.reshape([424, 512]))
		rawdepth_filter = self.filter(rawdepth_filter, 2)

		rawdepth_filter[not_mask] = 0
		rawdepth_filter[mask_edge] = rawdepth.reshape([424, 512])[mask_edge]
		depth_filter = self.depth2gray(rawdepth_filter, False)
		rawdepth_filter = rawdepth_filter.reshape(rawdepth.shape)

		return rawdepth_filter


	def initNet(self, path, sess):
		normal_ori_i = int(path[len(path) - 1])
		lightdir = [0.0, 0.0, 1.0]
		batch_size, height, width = 1, 424, 512
		model = DPNet(batch_size, height, width, normal_ori_i, lightdir)
		normal_, reflect_, I_ = model.net('predicting')
		ckptpath = DATAPATH + 'train_log/' + path + '/ckpt'
		tf.train.Saver().restore(sess, tf.train.latest_checkpoint(ckptpath))
		if normal_ori_i == 0:
			return model, normal_ori_i, [normal_, reflect_, I_]
		else:
			return model, normal_ori_i, [reflect_, I_]

	def run(self):

		run = True

		# do color Calibration
		# self.colorCalibration()
		# run = False

		# # capture train data
		# gid, gnum, gdelay = 0, 10, 0
		# cd_dirname = 'capture_data_origin_0531/'
		# self.captureTrainData(gid, gnum, gdelay, cd_dirname)
		# run = False

		
		print('start...')

		if run:
			# for calibration between camera and kinect
			R, T = self.calculateRT()
			base_p_irs = np.array([np.array([i, j, 1.0]) for i in range(424) for j in range(512)], np.float32)

			# init net
			sess = tf.Session()
			model, normal_ori_i, content = self.initNet('20180609_234840_0', sess)
			
			self.time = time.time()

			# 0: virtual scene with BRDF, 1: light, 2: casual
			if REALTIME:
				projection_mode = 1
			else:
				projection_mode = 2

		while run:
			ch = cv.waitKey(1)
			if ch == 27:
				break

			if MODE < 2:
				flag, rawdepth, rawcolor, rawinfrared, cameraColor = self.getRawDataWithKinect(SAVE)  
			else:
				flag, rawdepth, rawcolor, rawinfrared, cameraColor = self.getRawData()

			if flag:

				rgbd, depth_part = self.preprocess(rawdepth, rawcolor)
				depth = rgbd[:, :, 3]
				color = rgbd[:, :, 0: 3]
				
				if SAVE:
					cv.imwrite(DATAPATH + SUB + 'depth.png', depth)
					cv.imwrite(DATAPATH + SUB + 'color.png', color)
					cv.imwrite(DATAPATH + SUB + 'cameraColor.png', cameraColor)
					np.save(DATAPATH + SUB + 'depth_part.npy', depth_part)

				# cv.imshow('depth', depth)
				# cv.imshow('color', color)
				# cv.imshow('cameraColor', cameraColor)
				# cv.imshow('depth_part', depth_part)


				if projection_mode > 0:
					mask = depth_part >  0
					not_mask = depth_part <= 0

					# smooth rawdepth map
					rawdepth_filter = self.smooth(mask, not_mask, rawdepth, depth_part)

					# TODO: segmentation


					# BRDF reconstruction
					# normal_ori_i = 1
					# pre_reflect = np.ones([424, 512, 3], np.float32)
					# pre_normal = None

					normal_ori_i = 0
					normal = self.depth2normal(rawdepth_filter, mask)

					if normal_ori_i == 0:
						pre_normal, pre_reflect, pre_img = sess.run(
							content, 
							feed_dict = {
								model.normal: [normal], 
								model.color: [color], 
								model.mask: [np.expand_dims(mask, 2)], 
								model.lamda: 1.0
							})
						pre_normal[..., 0] = 0 - pre_normal[..., 0]

						# cv.imshow('pre_img', (pre_img[0] * 255).astype(np.uint8))
						# cv.imshow('pre_normal', ((pre_normal[0][..., ::-1] + 1) / 2 * 255).astype(np.uint8))
					else:
						pre_normal == None
						pre_reflect, pre_img = sess.run(
							content, 
							feed_dict = {
								model.normal: [normal], 
								model.color: [color], 
								model.mask: [np.expand_dims(mask, 2)], 
								model.lamda: 1.0
							})


				# test render prediction

					# normal_ori_i = 0

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

					# pre_normal[..., 0] = 0 - pre_normal[..., 0]

					# pre_img = np.expand_dims(pre_img, 0)
					# pre_normal = np.expand_dims(pre_normal, 0)
					# pre_reflect = np.expand_dims(pre_reflect, 0)

				# test render prediction end


					# calibration between kinect and camera
					cali = self.calibrateKinectCamera(R, T, base_p_irs, cameraColor, rawdepth, rawinfrared)
					# cv.imshow('cali', cali)
					if SAVE:
						np.save(DATAPATH + SUB + 'cali.npy', cali)

					# TODO: color compensation


					# render content
					corres = np.zeros([424, 512, 3], np.uint8)
					corres[mask] = np.array([255, 255, 255])
					texture = cv.imread(DATAPATH + 'texture2.png')
					# corres[mask] = texture[mask]
					# corres[mask] = pre_img[0][mask] * 255
					# cv.imshow('corres', corres)
					if SAVE:
						np.save(DATAPATH + SUB + 'corres.npy', corres)


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

				# tests end


				if time.time() - self.time > 5:

					# save proj data with projection on body
					if SAVEALL and projection_mode != 1:
						print('record...')
						if not os.path.isdir(DATAPATH + SUBOUT):
							os.mkdir(DATAPATH + SUBOUT)
						path = '{}{}'.format(DATAPATH + SUBOUT, self.index)
						while os.path.isdir(path):
							self.index += 1
							path = '{}{}'.format(DATAPATH + SUBOUT, self.index)
						os.mkdir(path)

						cv.imwrite(path + '/depth.png', depth)
						cv.imwrite(path + '/color.png', color)
						cv.imwrite(path + '/cameraColor.png', cameraColor)
						cv.imwrite(DATAPATH + SUBOUT + 'cameraColor{}.png'.format(self.index), cameraColor)
						cv.imwrite(path + '/prenormal.png', ((pre_normal[0][..., ::-1] + 1) / 2 * 255).astype(np.uint8))
						cv.imwrite(path + '/preimg.png', (pre_img[0] * 255).astype(np.uint8))

						np.save(path + '/depthback_origin.npy', self.depthback_origin)
						np.save(path + '/colorback_origin.npy', self.colorback_origin)
						np.save(path + '/rgbd.npy', rgbd)
						np.save(path + '/rawdepth.npy', rawdepth)
						np.save(path + '/rawcolor.npy', rawcolor)
						np.save(path + '/rawinfrared.npy', rawinfrared)

						self.index += 1
						print(self.index)

					if REALTIME:
						projection_mode = 1 - projection_mode
						if projection_mode == 0:
							print('project virtual scene')
						else:
							print('project light')

					self.time = time.time()

				if projection_mode == 1:
					self.projectLight()
				else:
					self.project(rawdepth_filter, corres, mask, normal_ori_i, pre_normal[0], pre_reflect[0])


			


if __name__ == '__main__':
	core = DynamicProjection()
	core.recordBackground(60)
	core.run()






