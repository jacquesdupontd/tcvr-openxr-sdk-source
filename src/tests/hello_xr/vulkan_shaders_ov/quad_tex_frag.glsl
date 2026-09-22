#version 450
layout(push_constant) uniform PC { mat4 uMvp; vec4 uTint; };
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColor;
void main() { oColor = texture(uTex, vUv) * uTint; }
