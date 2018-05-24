from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import numpy as np
import PIL.Image as im

PROJECTION_MODE = False

# 0: default	1: test
SHADER = 0


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



def LoadTexture(fileName):

	image = im.open(fileName)

	w, h = image.size[0], image.size[1]
	data = image.tobytes("raw", "RGBX", 0, -1)

	texture = glGenTextures(1)
	glBindTexture(GL_TEXTURE_2D, texture)
	glTexImage2D(GL_TEXTURE_2D, 0, 3, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

	return texture

class GLRenderer(object):
	def __init__(self, name, size, toTexture = False):
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
		self.normalBuf = glGenBuffers(1)
		glClearColor(0.0, 0.0, 0.0, 0.0)

		self.toTexture = toTexture
		if toTexture:
			self.texColor = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, self.texColor)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

			self.texDepth = glGenTextures(1)
			glBindTexture(GL_TEXTURE_2D, self.texDepth)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
			glTexImage2D(GL_TEXTURE_2D, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DRPTH_COMPONENT, GL_FLOAT, None)

			self.frameBuf = glGenFramebuffers(1)
			glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuf)
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texColor, 0)
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.texDepth, 0)

		if SHADER == 0:
			shaderPathList = [os.path.join('gl', sh) for sh in ['default.vs', 'default.gs', 'default.fs']]
		elif SHADER == 1:
			shaderPathList = [os.path.join('gl', sh) for sh in ['test.vs', 'test.fs']]
		self.program = LoadProgram(shaderPathList)
		self.mvpMatrix = glGetUniformLocation(self.program, 'MVP')

		if SHADER == 1:
			self.kd = glGetUniformLocation(self.program, 'kd')
			self.ld = glGetUniformLocation(self.program, 'ld')
			self.lightPosition = glGetUniformLocation(self.program, 'lightPosition')
			self.lightColor = glGetUniformLocation(self.program, 'lightColor')


		# glEnableVertexAttribArray(2)
		# self.texture = LoadTexture("data/image.bmp")
		# glUniform1i(glGetUniformLocation(self.program, 'Texture'), self.texture)
		# glActiveTexture(GL_TEXTURE1)
		# glEnable(GL_TEXTURE_2D)



	def draw(self, vertices, colors, normals, mvp):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glUseProgram(self.program)
		glUniformMatrix4fv(self.mvpMatrix, 1, GL_FALSE, mvp)

		if SHADER == 1:
			glUniform3fv(self.kd, 1, np.array((0.9, 0.9, 0.9), np.float32))
			glUniform3fv(self.ld, 1, np.array((1.0, 1.0, 1.0), np.float32))
			glUniform3fv(self.lightPosition, 1, np.array((1.0, 1.0, 1.0), np.float32))
			glUniform3fv(self.lightColor, 1, np.array((1.0, 1.0, 1.0), np.float32))

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


		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
		# glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
		glDrawArrays(GL_TRIANGLES, 0, vertices.shape[0])

		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(1)
		glDisableVertexAttribArray(2)
		glUseProgram(0)
		glutSwapBuffers()


		rgb = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, outputType = None)
		z = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT, outputType = None)
		rgb = np.flip(np.flip(rgb, 0), 1)
		z = np.flip(np.flip(z, 0), 1)

		return rgb, z







