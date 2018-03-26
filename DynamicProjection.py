from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2 as cv
import numpy as np
from gl.glrender import GLRenderer
import gl.glm as glm
import ctypes
import copy
import time

MODE = 2
# 0: record new background and capture new data by Kinect
# 1: use background data, but capture new data by Kinect
# 2: use data for all, no Kinect

SAVE = True

DATAPATH = '../DynamicProjectionData/'


class DynamicProjection(object):
	def __init__(self):
		print("Start initializing dp...")

		# init kinect
		self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

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

		self.render = GLRenderer(b'projection', (self.pwidth, self.pheight))

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
		self.cap.set(15, -6.0)		# exposure



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
					np.save("data/rgbd.npy", rgbd)
			else: 
				rgbd = np.load("data/rgbd.npy")

		else:

			# map color frame to depth sapce
			if MODE < 2:
				rgbd = self.c2d(rawdepth, rawcolor)
				if SAVE: 
					np.save("data/rgbd.npy", rgbd)
			else: 
				rgbd = np.load("data/rgbd.npy")


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


	def record_background(self, num_frame):
		print("Record {} frames as background...".format(num_frame))

		if MODE == 0:

			depth_cnt = np.zeros([424 * 512], np.uint8)
			color_cnt = np.zeros([1080 * 1920 * 4], np.uint8)

			for i in range(num_frame):
				while 1:
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

			np.save('data/depthback_origin.npy', self.depthback_origin)
			np.save('data/colorback_origin.npy', self.colorback_origin)

		else: 

			self.depthback_origin = np.load('data/depthback_origin.npy')
			self.colorback_origin = np.load('data/colorback_origin.npy')


	def project(self, rawdepth, corres, mask):
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

		
		corres[:, 0], corres[:, 2] = corres[:, 2], corres[:, 0]

		c0 = np.concatenate([corres[cul_cmask], corres[cdr_cmask]], 0)
		c2 = np.concatenate([corres[cul_umask], corres[cdr_dmask]], 0)
		c1 = np.concatenate([corres[cul_lmask], corres[cdr_rmask]], 0)
		c0, c1, c2 = c0 / 255, c1 / 255, c2 / 255

		colors = np.zeros([num * 3, 3], np.float32)
		colors[0::3, :], colors[1::3, :], colors[2::3, :] = c0, c1, c2


		# np.save('vertices.npy', vertices)

		self.render.draw(vertices, colors, self.mvp.T)


	def getRawData(self):
		rawdepth = np.load('data/rawdepth.npy')
		rawcolor = np.load('data/rawcolor.npy')
		cameraColor = np.load('data/cameraColor.npy')

		return True, rawdepth, rawcolor, cameraColor


	def getRawDataWithKinect(self, save):
		flag = False

		rawdepth = np.zeros((424 * 512, 1))
		rawcolor = np.zeros((1080 * 1092 * 4))
		cameraColor = np.zeros((1280 * 960 * 3))

		if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame():
			
			rawdepth = self.kinect.get_last_depth_frame()
			rawcolor = self.kinect.get_last_color_frame()

			if save:
				np.save('data/rawdepth.npy', rawdepth)
				np.save('data/rawcolor.npy', rawcolor)

			flag = True


		ret, cameraColor = self.cap.read()
		if save:
			np.save('data/cameraColor.npy', cameraColor)


		return flag, rawdepth, rawcolor, cameraColor


	def colorCalibration(self):

		idx = 0

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
				np.save('{}corres/corres_{}_{}.npy'.format(DATAPATH, c, idx % 256), corres)


				self.project(rawdepth, corres, mask)


			idx = idx + 1



	def run(self):

		run = True

		# # do color Calibration
		self.colorCalibration()
		run = False


		while run:
			ch = cv.waitKey(1)
			if ch == 27:
				break

			if MODE < 2:
				flag, rawdepth, rawcolor, cameraColor = self.getRawDataWithKinect(SAVE)  
			else:
				flag, rawdepth, rawcolor, cameraColor = self.getRawData()

			if flag:

				rgbd, depth_part = self.preprocess(rawdepth, rawcolor)
				depth = rgbd[:, :, 3]
				color = rgbd[:, :, 0: 3]
				mask = depth_part >  0

				# test position projection
				# color[200: 210, 250: 260] = np.array([255, 0, 0])
				# color[100: 110, 230: 240] = np.array([255, 0, 0])

				cv.imshow('depth', depth)
				cv.imshow('color', color)
				cv.imshow('cameraColor', cameraColor)
				# cv.imshow('depth_part', depth_part)

				if SAVE:
					cv.imwrite('data/depth.png', depth)
					cv.imwrite('data/color.png', color)
					cv.imwrite('data/cameraColor.png', cameraColor)

				corres = np.zeros([424, 512, 3], np.uint8)
				corres[mask] = np.array([255, 255, 255])


				# # test color projection
				# corres = np.array([[[(i + j + k * 80) % 256 for k in range(3)] for j in range(512)] for i in range(424)])
				# corres[np.logical_not(mask)] = np.array([0, 0, 0])



				# test image projection
				image = cv.imread('data/image.bmp')
				image = image[..., ::-1]

				x0, y0, x1, y1 = 120, 205, 290, 335
				w, h = image.shape[0], image.shape[1]
				corres[x0: x1, y0: y1] = np.array([[image[int((i - x0) / (x1 - x0) * h), int((y1 - 1 - j) / (y1 - y0) * w)] for j in range(y0, y1)] for i in range(x0, x1)])


				# test position projection
				# corres[200: 210, 250: 260] = np.array([255, 0, 0])
				# corres[100: 110, 230: 240] = np.array([255, 0, 0])

				if SAVE:
					np.save('data/corres.npy', corres)
					np.save('data/depth_part.npy', depth_part)

				self.project(rawdepth, corres, mask)


			


if __name__ == '__main__':
	core = DynamicProjection()
	core.record_background(60)
	core.run()






