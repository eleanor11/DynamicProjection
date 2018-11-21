#version 330 core


in vec3 vColor;
out vec4 outColor;


void main() {

	outColor = vec4(vColor, 1.0) * 0.5;
	// outColor = vec4(1.0, 0.5, 0.0, 1.0);

}