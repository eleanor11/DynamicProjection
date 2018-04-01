#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;

uniform mat4 MVP;

out vec3 vColor;
out vec2 uv;

void main() {
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1);
	// vColor = vertexColor;
	vColor = normalize(vertexNormal);
	// vColor = texture2D(texture, vec3(uvw.xy + texture))
	// uv = vertexUV;
}