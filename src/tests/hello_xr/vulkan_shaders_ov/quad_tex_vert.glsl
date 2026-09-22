#version 450
// World-space textured quad (menu, later the crosshair): corners from the vertex index, MVP pushed.
layout(push_constant) uniform PC { mat4 uMvp; vec4 uTint; };
layout(location = 0) out vec2 vUv;
void main() {
    const vec2 c[6] = vec2[6](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5));
    vec2 p = c[gl_VertexIndex];
    vUv = vec2(p.x + 0.5, 0.5 - p.y);
    gl_Position = uMvp * vec4(p, 0.0, 1.0);
}
