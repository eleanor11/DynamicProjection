#version 330 core


in vec3 vColor;
in vec3 vUV;
out vec4 outColor;

uniform sampler2D myTexture;
uniform sampler2D myTextureLight;

void main() {

	if (vUV.z == 0.0) {
		outColor = vec4(vColor * texture(myTexture, vUV.xy).rgb * 0.5, 1.0f);
	}
	else {
		outColor = vec4(vColor, texture(myTextureLight, vUV.xy).r);
	}

}