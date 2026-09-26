#version 450
// AppSW motion vectors (26/09): the vertex stage's CurrNDC - PrevNDC, written as is (RGBA16F, w unused), times a live
// per-axis factor (debug.tcvr.appsw_mvScale / appsw_mvFlipY: the runtime's convention is not documented).
layout(location = 10) in vec3 vMv;
layout(location = 0) out vec4 oMv;
layout(push_constant) uniform PC { vec4 uMvScale; } pc;
void main() { oMv = vec4(vMv * pc.uMvScale.xyz, 0.0); }
