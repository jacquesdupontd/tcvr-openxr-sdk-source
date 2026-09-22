#version 450
void main() {
    vec2 p = vec2(float((gl_VertexIndex & 1) * 4) - 1.0, float((gl_VertexIndex >> 1) * 4) - 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
