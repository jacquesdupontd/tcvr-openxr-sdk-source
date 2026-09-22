#version 450
layout(location = 0) out vec2 vUv;
void main() {
    // Generate fullscreen triangle from vertex index (0, 1, 2)
    // 0: (-1, -1), 1: (3, -1), 2: (-1, 3)
    vec2 pos = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
    vUv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
