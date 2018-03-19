from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import cv2 as cv

def RunKinect():
	kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
	if kinect.has_new_depth_frame() and kinect.has_new_color_frame():
		depth = self.kinect.get_last_depth_frame()
		color = self.kinect.get_last_color_frame()

		color = color.reshape([1080, 1920, 4])
		cv.imshow('color', color)


if __name__ == '__main__':

	RunKinect()