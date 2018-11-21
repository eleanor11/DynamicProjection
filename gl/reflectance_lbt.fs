#version 330 core


in vec3 vColor;
in vec2 vUV;
out vec4 outColor;

uniform sampler2D myTexture;

void main() {

	// outColor = vec4(vColor, 1.0);
	outColor = vec4(vColor * texture(myTexture, vUV).rgb, 1.0f) * 0.5;
	// outColor = vec4(texture(myTexture, vUV).rgb, 1.0f);
	// outColor = vec4(1.0, 0.5, 0.0, 1.0);

}