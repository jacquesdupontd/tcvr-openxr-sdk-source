#version 450
// AppSW motion vectors (26/09): the vertex stage's CurrNDC - PrevNDC, written as is (RGBA16F, w unused).
layout(location = 10) in vec3 vMv;
layout(location = 0) out vec4 oMv;
void main() { oMv = vec4(vMv, 0.0); }
