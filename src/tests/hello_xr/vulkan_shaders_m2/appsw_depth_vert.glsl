#version 450
// AppSW depth (26/09): the REAL distance of the main view's polygons, for the headset's reprojection only. The colour
// pass keeps the game's painter order as its depth (m2.depthOrder=1) -- sent to the headset, that rank made every
// pixel slide by its drawing order when the head moved (ghosting). Same vertex placement as m2_vert (raw path).
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aParam;
layout(push_constant) uniform PC {
    mat4 uMvp;
    vec2 uFocus;
} pc;
void main() {
    gl_Position = pc.uMvp * vec4(aPos.x / max(pc.uFocus.x, 1e-6), aPos.y / max(pc.uFocus.y, 1e-6), aParam.x, 1.0);
}
