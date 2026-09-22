#version 450
// Arcade screen quad (screen presentation of every game).
layout(push_constant) uniform PC { mat4 uMvp; vec2 uAim; vec2 uSrcSize; int uAimVisible; int uCalibrating; int uFilter; float uSharpen; };
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUv;
layout(location = 0) out vec2 vUv;
void main() { vUv = aUv; gl_Position = uMvp * vec4(aPos, 1.0); }
