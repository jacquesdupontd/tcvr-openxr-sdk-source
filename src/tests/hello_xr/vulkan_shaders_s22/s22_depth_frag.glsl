#version 450
// Depth map: 3D polygons in the board's own projection, camera depth per texel (for the aim and world sprites).
layout(location = 0) out vec4 oDepth;
void main() { oDepth = vec4(1.0 / gl_FragCoord.w, 0.0, 0.0, 0.0); }
