#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec3 vertexReflect;

uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform mat4 MVP;

out vec3 vColor;

void main() {
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1);

	vec3 normal = normalize(vertexNormal);
	vec3 lightDir = normalize(lightPosition);

	float NdotL = max(dot(normal, lightDir), 0.0);
	vColor = vertexReflect * NdotL * lightColor * vertexColor;
}



