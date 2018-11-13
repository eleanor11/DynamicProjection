#version 330 core


in vec3 vColor;
in vec3 vUV;
in int flag;
out vec4 outColor;

uniform sampler2D myTexture;
uniform sampler2D myTextureLight;

void main() {

	// outColor = vec4(vColor, 1.0);
	// outColor = vec4(1.0, 0.5, 0.0, 1.0);
	// outColor = vec4(texture(myTexture, vUV).rgb, 1.0f);

	if (vUV.z == 0.0) {
		outColor = vec4(vColor * texture(myTexture, vUV.xy).rgb * 0.5, 1.0f);
	}
	else {
		outColor = vec4(vColor, texture(myTextureLight, vUV.xy).r);
	}

}