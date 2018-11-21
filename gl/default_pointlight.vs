#version 330 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec3 vertexReflect;
layout(location = 4) in vec2 vertexUV;

uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform mat4 MVP;
uniform vec3 factors;

out vec3 vColor;
out vec3 vUV;

void main() {
	
	gl_Position = MVP * vec4(vertexPosition, 1);

	float dist = (lightPosition.x - vertexPosition.x) * (lightPosition.x - vertexPosition.x) + (lightPosition.y - vertexPosition.y) * (lightPosition.y - vertexPosition.y);
	float attenuation = 1.0 / (factors.z * dist * dist + factors.y * dist + factors.x);

	vec3 normal = normalize(vertexNormal);
	vec3 lightDir = normalize(lightPosition);


	float NdotL = max(dot(normal, lightDir), 0.0);
	if (vertexReflect.x + 1 < 0.0001) {
		vColor = lightColor;
		vUV = vec3(vertexUV, 1.0);
	}
	else {
		vColor = NdotL * vertexColor * lightColor * attenuation;
		vUV = vec3(vertexUV, 0.0);
	}


}