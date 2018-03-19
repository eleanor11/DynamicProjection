from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import copy
import numpy as np

import cv2 as cv

def depth2gray(d):
	depth = copy.copy(d)

	depth = depth / 1000

	znear = 0.2
	zfar = 2.0
	depth[depth > zfar] = zfar
	depth[depth < znear] = znear
	depth = ((zfar - depth) / (zfar - znear) * 255).astype(np.uint8)

	return depth


def RunKinect():
	kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Infrared)
	while 1:
		cv.waitKey(1)
		if kinect.has_new_depth_frame() and kinect.has_new_color_frame() and kinect.has_new_infrared_frame():
			depth = kinect.get_last_depth_frame()
			color = kinect.get_last_color_frame()
			infrared = kinect.get_last_infrared_frame()


			color = color.reshape([1080, 1920, 4])
			color = color[:, :, :3]
			cv.imshow('color', color)


			depth = depth.reshape([424, 512])
			infrared = infrared.reshape([424, 512])
			depth = depth2gray(depth)
			infrared = depth2gray(infrared)
			cv.imshow('depth', depth)
			cv.imshow('infrared', infrared)


if __name__ == '__main__':

    RunKinect()