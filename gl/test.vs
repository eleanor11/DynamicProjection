#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;

out vec3 lighIntensity;

uniform vec4 lightPosition;
uniform vec3 kd;
uniform vec3 ld;

uniform mat4 MVP;

out vec3 vColor;
out vec2 uv;

void main() {
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1);

	// // vColor = texture2D(texture, vec3(uvw.xy + texture))
	// // uv = vertexUV;

	vec3 normal = normalize(vertexNormal);
	vec3 lightDir = normalize(lightPosition);

	float NdotL = max(dot(normal, lightDir), 0.0);
	vColor = kd * ld * vertexColor;


	// vColor = vertexColor;

}