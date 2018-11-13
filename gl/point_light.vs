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

	// float dist = length(lightPosition - vertexPosition);
	float dist = (lightPosition.x - vertexPosition.x) * (lightPosition.x - vertexPosition.x) + (lightPosition.y - vertexPosition.y) * (lightPosition.y - vertexPosition.y);
	float attenuation = 1.0 / (factors.z * dist * dist + factors.y * dist + factors.x);
	// attenuation = 1.0;
	

	vec3 normal = normalize(vertexNormal);
	vec3 lightDir = normalize(lightPosition);
 
	float NdotL = max(dot(normal, lightDir), 0.0);
	if (vertexReflect.x + 1 < 0.0001) {
		vColor = vec3(1.0, 1.0, 1.0);
		vUV = vec3(vertexUV.x, vertexUV.y, 1.0);
	}
	else {
		vColor = vertexReflect * NdotL * lightColor * vertexColor * attenuation;
		vUV = vec3(vertexUV.x, vertexUV.y, 0.0);
	}

	// vUV = vertexUV;
	// vColor = normal * 0.5 + 0.5;

	// float a = 1.0;
	// float b = 0.5;

	// float k = attenuation;
	// float k = dist;

	// if (k > a) {
	//	vColor = vec3(0.0, 0.0, 1.0);
	// }
	// else if (k > b){
	// 	vColor = vec3(1.0, 0.0, 0.0);
	// }
	// else {
	// 	vColor = vec3(0.0, 1.0, 0.0);
	// }

}



