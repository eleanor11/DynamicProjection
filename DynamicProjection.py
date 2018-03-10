from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2 as cv
import numpy as np
from gl.glrender import GLRenderer
import gl.glm as glm
import ctypes
import copy

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
            -1.390668693623873,
-0.011007482582276377,
-0.8667484528337432,
0.6619667407488129,
-0.03976989021546296,
1.7952502894980042,
-1.2026607080803946,
0.6863956127031143,
-0.0337235987568367,
0.004993447985183048,
-1.5366058015071355

		]

		self.pwidth = 1024
		self.pheight = 768

		self.render = GLRenderer(b'projection', (self.pwidth, self.pheight))

		self.mvp = glm.ortho(-1, 1, -1, 1, -1, 1000)

		self.depthback_origin = np.zeros([424 * 512], np.float32)
		self.colorback_origin = np.zeros([1080 * 1920 * 4], np.float32)

		self.depthback = np.zeros([424, 512, 1], np.uint8)
		self.colorback = np.zeros([424, 512, 4], np.uint8)

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
			rgbd = self.d2c(rawdepth, rawcolor)

		else:

			# map color frame to depth sapce
			rgbd = self.c2d(rawdepth, rawcolor)
			


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

		# depth_cnt = np.zeros([424 * 512], np.uint8)
		# color_cnt = np.zeros([1080 * 1920 * 4], np.uint8)

		# for i in range(num_frame):
		# 	while 1:
		# 		if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame():
		# 			depth_frame = self.kinect.get_last_depth_frame()
		# 			color_frame = self.kinect.get_last_color_frame()
		# 			depth_cnt[frame_depth > 0] += 1
		# 			color_cnt[frame_color > 0] += 1
		# 			self.depthback_origin += depth_frame
		# 			self.colorback_origin += color_frame
		# 			break

		# depth_mask = depth_cnt > 0
		# self.depthback_origin[depth_mask] /= depth_cnt[depth_mask]
		# color_mask = color_cnt > 0
		# self.colorback_origin[color_mask] /= color_cnt[color_mask]

		# rgbd = d2c(self.depthback_origin, self.colorback_origin)
		# rgbd[:, 3] = depth2gray(self.depthback_origin)

		# back = rgbd.reshape([424, 512, 4])
		# self.depthback = back[:, :, 3]
		# self.colorback = back[:, :, 0: 3]

		# np.save('data/depthback_origin', depthback_origin)
		# np.save('data/colorback_origin', colorback_origin)

		self.depthback_origin = np.load('data/depthback_origin.npy')
		# self.colorback_origin = np.load('data/colorback_origin.npy')


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


	def run(self):
		while 1:
			ch = cv.waitKey(1)
			if ch == 27:
				break

			# if self.kinect.has_new_depth_frame() and self.kinect.has_new_color_frame():
			if True:
				# rawdepth = self.kinect.get_last_depth_frame()
				# rawcolor = self.kinect.get_last_color_frame()

				# np.save('data/rawdepth.npy', rawdepth)
				# np.save('data/rawcolor.npy', rawcolor)

				rawdepth = np.load('data/rawdepth.npy')
				rawcolor = np.load('data/rawcolor.npy')

				rgbd, depth_part = self.preprocess(rawdepth, rawcolor)
				depth = rgbd[:, :, 3]
				color = rgbd[:, :, 0: 3]
				mask = depth_part >  0

				cv.imshow('depth', depth)
				cv.imshow('color', color)

				corres = np.zeros([424, 512, 3], np.uint8)
				corres[mask] = np.array([255, 255, 255])


				self.project(rawdepth, corres, mask)


if __name__ == '__main__':
	core = DynamicProjection()
	core.record_background(60)
	core.run()





