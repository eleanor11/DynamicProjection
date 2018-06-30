from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import cv2 as cv
import PIL.Image as im

PROJECTION_MODE = False

# 0: default	
# 1: shader1(lambert) 	
# 2: shader2(reflect * normal)
# SHADER = 0
# SHADER = 1
SHADER = 2
# # lightPosition = np.array([0.0, 0.0, 1.0])
# lightPosition = np.array([-1.0, 0.0, 0.0])
# # lightPosition = np.array([1.0, 2.0, 0.0])

# lightPosition = np.array([1.0, 0.0, 1.0])
lightPosition = np.array([0.0, 0.0, 1.0])
# lightPosition = np.array([1.0, 2.0, 0.0])
# lightPosition = np.array([1.0, 0.0, 0.2])
# lightPosition = np.array([1.0, 0.0, 0.5])

def LoadProgram(shaderPathList):
	shaderTypeMapping = {
		'vs': GL_VERTEX_SHADER, 
		'gs': GL_GEOMETRY_SHADER, 
		'fs': GL_FRAGMENT_SHADER
	}
	shaderTypeList = [shaderTypeMapping[shaderType] for shaderPath in shaderPathList for shaderType in shaderTypeMapping if shaderPath.endswith(shaderType)]
	shaders = []
	for i in range(len(shaderPathList)):
		shaderPath = shaderPathList[i]
		shaderType = shaderTypeList[i]

		with open(shaderPath, 'r') as f:
			shaderData = f.read()

		shader = glCreateShader(shaderType)
		glShaderSource(shader, shaderData)
		glCompileShader(shader)
		shaders.append(shader)

	program = glCreateProgram()
	for shader in shaders:
		glAttachShader(program, shader)
	glLinkProgram(program)
	for shader in shaders:
		glDetachShader(program, shader)
		glDeleteShader(shader)
	return program

def LoadTexture(image):

	[w, h] = image.shape[0:2]
	# print(w, h)

	texture = glGenTextures(1)
	glActiveTexture(GL_TEXTURE0)
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, image)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

	return texture

class GLRenderer(object):
	def __init__(self, name, size, tex_image):
		self.width, self.height = size

		glutInit()
		displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL
		glutInitDisplayMode(displayMode)
		glutInitWindowPosition(0, 0)
		glutInitWindowSize(self.width, self.height)
		if PROJECTION_MODE:
			glutEnterGameMode()
		else:
			self.window = glutCreateWindow(name)
		glEnable(GL_CULL_FACE)
		glEnable(GL_DEPTH_TEST)
		glDepthFunc(GL_LESS)

		self.vertexArr = glGenVertexArrays(1)
		glBindVertexArray(self.vertexArr)
		self.vertexBuf = glGenBuffers(1)
		self.colorBuf = glGenBuffers(1)

		self.shader = SHADER

		self.lightPosition = lightPosition
		self.lightColor = np.array([1.0, 1.0, 1.0], np.float32)

		# shader 1
		self.normalBuf = glGenBuffers(1)
		# shader 2
		self.reflectBuf = glGenBuffers(1)
		self.texture = LoadTexture(tex_image)
		self.uvBuf = glGenBuffers(1)
		glClearColor(0.0, 0.0, 0.0, 0.0)


		self.program = []
		shaderPathList = [os.path.join('gl', sh) for sh in ['default.vs', 'default.gs', 'default.fs']]
		# shaderPathList = [os.path.join('gl', sh) for sh in ['default.vs', 'default.fs']]
		self.program.append(LoadProgram(shaderPathList))
		shaderPathList = [os.path.join('gl', sh) for sh in ['test.vs', 'test.fs']]
		self.program.append(LoadProgram(shaderPathList))
		shaderPathList = [os.path.join('gl', sh) for sh in ['test_brdf.vs', 'test_brdf.fs']]
		self.program.append(LoadProgram(shaderPathList))

		for i in range(3):
			self.mvpMatrix = glGetUniformLocation(self.program[i], 'MVP')

		# shader 1
		self.kd_ = glGetUniformLocation(self.program[1], 'kd')
		self.ld_ = glGetUniformLocation(self.program[1], 'ld')
		self.lightPosition_1 = glGetUniformLocation(self.program[1], 'lightPosition')
		self.lightColor_1 = glGetUniformLocation(self.program[1], 'lightColor')
		# self.texture = glGetUniformLocation(self.program[1], 'myTexture')

		# shader 2
		self.lightPosition_2 = glGetUniformLocation(self.program[2], 'lightPosition')
		self.lightColor_2 = glGetUniformLocation(self.program[2], 'lightColor')
		self.texture_ = glGetUniformLocation(self.program[2], 'myTexture')


	def display(self):

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glutSolidSphere(0.2, 30, 30)
		glutSwapBuffers()

		rgb = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, outputType = None)
		cv.imwrite('rgb.png', rgb)

	def drawshape(self):

		glutDisplayFunc(self.display)

		mat_ambient = [0.8, 0.8, 0.8, 1.0]
		mat_diffuse = [0.8, 0.0, 0.8, 1.0]
		mat_diffuse = [0.8, 0.8, 0.8, 1.0]
		mat_specular = [1.0, 0.0, 1.0, 1.0]
		mat_specular = [1.0, 1.0, 1.0, 1.0]
		mat_shininess = [50.0]

		light_position = [1.0, 1.0, 1.0, 0.0]

		glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
		glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
		glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

		glLightfv(GL_LIGHT0, GL_POSITION, light_position)

		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glDepthFunc(GL_LESS)
		glEnable(GL_DEPTH_TEST)

		glutMainLoop()


	def draw(self, vertices, colors, normals, reflects, uv, mvp, shader = SHADER):
		self.shader = shader
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glUseProgram(self.program[self.shader])
		glUniformMatrix4fv(self.mvpMatrix, 1, GL_FALSE, mvp)

		if self.shader == 1:
			glUniform3fv(self.kd_, 1, np.array((0.9, 0.9, 0.9), np.float32))
			glUniform3fv(self.ld_, 1, np.array((1.0, 1.0, 1.0), np.float32))
			glUniform3fv(self.lightPosition_1, 1, self.lightPosition)
			glUniform3fv(self.lightColor_1, 1, self.lightColor)
			# glUniform1i(self.texture_, 0)
		elif self.shader == 2:
			glUniform3fv(self.lightPosition_2, 1, self.lightPosition)
			glUniform3fv(self.lightColor_2, 1, self.lightColor)
			glUniform1i(self.texture_, 0)

		# print('lightPosition ', self.lightPosition)
		# print('lightColor ', self.lightColor)

		glEnableVertexAttribArray(0)
		glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuf)
		glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
		glVertexAttribPointer(
			0, 
			vertices.shape[1], 
			GL_FLOAT, 
			GL_FALSE, 
			0, 
			None
		)

		glEnableVertexAttribArray(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.colorBuf)
		glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
		glVertexAttribPointer(
			1, 
			colors.shape[1], 
			GL_FLOAT, 
			GL_FALSE, 
			0, 
			None
		)

		# SHADER 1, 2
		if self.shader > 0:
			glEnableVertexAttribArray(2)
			glBindBuffer(GL_ARRAY_BUFFER, self.normalBuf)
			glBufferData(GL_ARRAY_BUFFER, normals, GL_STATIC_DRAW)
			glVertexAttribPointer(
				2, 
				normals.shape[1],
				GL_FLOAT, 
				GL_FALSE, 
				0, 
				None
			)

		# SHADER 2
		if self.shader > 1:
			glEnableVertexAttribArray(3)
			glBindBuffer(GL_ARRAY_BUFFER, self.reflectBuf)
			glBufferData(GL_ARRAY_BUFFER, reflects, GL_STATIC_DRAW)
			glVertexAttribPointer(
				3, 
				reflects.shape[1],
				GL_FLOAT, 
				GL_FALSE, 
				0, 
				None
			)
			glEnableVertexAttribArray(4)
			glBindBuffer(GL_ARRAY_BUFFER, self.uvBuf)
			glBufferData(GL_ARRAY_BUFFER, uv, GL_STATIC_DRAW)
			glVertexAttribPointer(
				4, 
				uv.shape[1],
				GL_FLOAT, 
				GL_FALSE, 
				0, 
				None
			)


		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
		# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
		glDrawArrays(GL_TRIANGLES, 0, vertices.shape[0])

		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(1)
		if self.shader > 0:
			glDisableVertexAttribArray(2)
		if self.shader > 1:
			glDisableVertexAttribArray(3)
			glDisableVertexAttribArray(4)
		glUseProgram(0)
		glutSwapBuffers()


		rgb = glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_BYTE, outputType = None)
		z = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, outputType = None)


		# rgb = np.flip(np.flip(rgb, 0), 1)

		h, w = rgb.shape[0], rgb.shape[1]
		rgb = rgb.reshape([h * w, 3])
		rgb = rgb.reshape([w, h, 3])
		rgb = np.flip(rgb, 0)

		z = np.flip(np.flip(z, 0), 1)


		return rgb, z







