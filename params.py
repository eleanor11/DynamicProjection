import numpy as np


# when kinect and projector are connected
# MODE = 0 or 1, PROJECTION_MODE = True
# when kinect and projector are not connected
# MODE = 2, PROJECTION_MODE = False


# 0: record new background and capture new data by Kinect
# 1: use background data, but capture new data by Kinect
# 2: use off-line data for all
MODE = 2

# ------------------------------------------------

PROJECTION_MODE = False


# ------------------------------------------------


# # SUBIN is the path to load the off-line data

# SUBIN = 'data/data_pig_0629_origin/'
# SUBIN = 'data/data_body_0629_origin/'
# SUBIN = 'data/data_body_0629_2_origin/'
# SUBIN = 'data/data_pig_0629_2_origin/'
# SUBIN = 'data/data_body_empty_origin/'
# SUBIN = 'data/data_pig_0630_origin/'

# SUBIN = 'data/data_body_origin/'
SUBIN = 'data/data_body_origin_1/'
# SUBIN = 'data/data_bear_origin/'
# SUBIN = 'data/data_bear_origin_3306/'
# SUBIN = 'data/data_hand_origin/'

# ------------------------------------------------

# # SUBOUT is the path to save the off-line data

# SUBOUT = 'data/data_pig/'
# SUBOUT = 'data/data_bear/'
# SUBOUT = 'data/data_pig_0629_origin/'
# SUBOUT = 'data/data_body_0629_origin/'
# SUBOUT = 'data/data_body_0629_2_origin/'
# SUBOUT = 'data/data_body_empty_origin/'
# SUBOUT = 'data/data_pig_0630_origin/'

# SUBOUT = 'data/data_body_origin/'
SUBOUT = 'data/data_body_origin_1/'
# SUBOUT = 'data/data_bear_origin/'
# SUBOUT = 'data/data_bear_origin_3306/'
# SUBOUT = 'data/data_hand_origin/'

# ------------------------------------------------


# # SUBALL is the path to save all data

# SUBALL = 'data/data_body_0701/'
# SUBALL = 'data/data_pig_0630_6/'
# SUBALL = 'data/data_bear_0630_1/'
# SUBALL = 'data/data_bear_0630_2/'
# SUBALL = 'data/data_bear_0630_3/'
SUBALL = 'data/data_body_1114/'

# ------------------------------------------------

# 0: no reconstruction
# 1: reconstruction of real scene
# 2: use off-line data (rawdepth, mask, pre_normal, pre_reflect, pre_img)
RECONSTRUCTION_MODE = 2

# ------------------------------------------------

# # SUB_BRDF is the path to save off-line BRDF data

# SUB_BRDF = 'data0/data_pig_0629/0/'
SUB_BRDF = 'data0/data_body_0629_21/0/'


# ------------------------------------------------

# TEXTUREFILE = 'texture4.png'
TEXTUREFILE = ''

TEXTUREFILE_LIGHT = 'texture_light.png'

# ------------------------------------------------

# 0: predicted, not realtime
# 1: lighting & predicted
# 2: lighting & predicted & lambertian
# 3: lighting & predicted & lambertian, change illumination
# 4: lighting & predicted & lambertian & color lighting, change light color 3 * 256
# 5: lighting & predicted & lambertian & color lighting, change illumination
# 6: lighting & predicted point, not realtime
# 7: lighting & predicted point, realtime
# 8: lighting & predicted point & default point, realtime
# 9: lighting & predicted point & designed point, change illumination

REALTIME_MODE = 8

REALTIME_LIMIT = 5
PROJECTION_TYPE = ['lighting', 'predicted', 'lambertian', 'colorlighting', 'predicted_point', 'defalut_point', 'designed_params_point']

# ------------------------------------------------

# LightPositions = np.array([
# 	[1.0, 0.0, 0.5], 
# 	[0.0, 0.0, 1.0], 
# ])

# LightPositions = np.array([
# 	[0.0, 0.0, 1.0], 
# 	[1.0, 0.0, 1.0], 
# 	[1.0, 0.0, 0.5], 
# 	[1.0, 0.0, 0.2]
# ])
# LightPositions = np.array([
# 	[0.0, 0.0, 1.0], 
# 	[1.0, 0.0, 0.2], 
# ])
# LightPositions = np.array([
# 	[0.0, 0.0, 1.0], 
# 	[0.5, 0.0, 1.0], 
# 	[1.0, 0.0, 1.0], 
# 	[1.0, 0.0, 0.8], 
# 	[1.0, 0.0, 0.5], 
# 	[1.0, 0.0, 0.2], 
# 	[1.0, 0.0, 0.0],
# 	[0.0, 0.0, 0.0],
# ])
LightPositions = np.array([
	[-1.0, 0.0, 1.0], 
	[-0.5, 0.0, 1.0], 
	[0.0, 0.0, 1.0], 
	[0.5, 0.0, 1.0], 
	[1.0, 0.0, 1.0], 
	[1.0, 0.5, 1.0], 
	[1.0, 1.0, 1.0],
])
# LightPositions = np.array([
# 	[-1.0, 0.0, 1.0], 
# 	[-0.5, 0.0, 1.0], 
# 	[0.0, 0.0, 1.0], 
# 	[0.5, 0.0, 1.0], 
# 	[1.0, 0.0, 1.0], 
# 	[1.0, 0.5, 1.0], 
# 	[1.0, 1.0, 1.0],
# 	[0.5, 0.75, 1.0],
# 	[0.0, 0.5, 1.0],
# 	[-0.5, 0.25, 1.0],
# ])

# LightPositions = np.array([
# 	[-1.0, -0.5, 1.0], 
# 	[-0.75, -0.5, 1.0], 
# 	[-0.5, -0.25, 1.0], 
# 	[-0.25, -0.25, 1.0], 
# 	[0.0, 0.0, 1.0], 
# 	[0.25, -0.25, 1.0], 
# 	[0.5, -0.5, 1.0], 
# 	[0.5, -1.0, 1.0], 
# ])


# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.0, 1.0, 1.0], 
# 	[0.0, 0.0, 1.0], 
# ])
# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[1.0, 1.0, 0.0], 
# 	[0.0, 1.0, 0.0], 
# # ])
# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.0, 1.0, 1.0], 
# 	[0.5, 1.0, 0.0], 
# ])
# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.0, 1.0, 0.0], 
# 	[0.5, 1.0, 0.0],
# 	[1.0, 1.0, 0.0],  
# 	[1.0, 0.5, 0.0], 
# ])
# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.0, 1.0, 1.0], 
# 	[1.0, 0.0, 1.0], 
# 	[1.0, 1.0, 0.0], 
# 	[0.0, 1.0, 0.0],
# ])
# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.5, 1.0, 1.5], 
# 	[1.0, 0.5, 1.0], 
# 	[1.0, 1.0, 0.5], 
# 	[0.5, 1.0, 0.5],
# ])
LightColors = np.array([
	[1.0, 1.0, 1.0], 
	[0.0, 1.0, 1.0], 
	[1.0, 0.0, 1.0], 
	[1.0, 1.0, 0.0],
	[0.5, 1.0, 1.0], 
	[1.0, 0.5, 1.0], 
	[1.0, 1.0, 0.5],  
	[1.0, 0.5, 0.5],
	[0.5, 1.0, 0.5],
	[0.5, 0.5, 1.0],
])

# LightColors = np.array([
# 	[1.0, 1.0, 1.0], 
# 	[0.5, 1.0, 1.5], 
# 	[1.0, 0.5, 1.0], 
# 	[1.0, 1.0, 0.5], 
# 	[0.5, 1.0, 0.5],
# 	[0.25, 1.0, 1.5], 
# 	[1.0, 0.25, 1.0], 
# 	[1.0, 1.0, 0.25], 
# 	[0.5, 1.0, 0.25],
# ])