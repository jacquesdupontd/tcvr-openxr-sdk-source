#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 2 vertex stage (verbatim port of kSceneVs; Vulkan clip depth is 0..1).
#include "s22_common.glsl"
invariant gl_Position;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aTex;
layout(location = 2) in vec2 aMeta;   // zoom, primitive index
layout(location = 0) out vec3 vTex;
layout(location = 1) out float vOriginalDepth;
layout(location = 2) out vec2 vScreen;
layout(location = 3) flat out int vPrim;
void main() {
    vec2 ScreenSize = ScreenOut.xy;
    int p = int(aMeta.y + 0.5);
    vec4 t0 = P(p, 0);   // kind, direct, cx, cy
    float sx, sy, w;
    int Immersive = Flags.x;
    if (Immersive != 0 && (t0.x > 0.5 || t0.y > 0.5)) {
        sx = t0.x > 0.5 ? aPos.x : t0.z + aPos.x;
        sy = t0.x > 0.5 ? aPos.y : t0.w - aPos.y;
        vScreen = vec2(sx, sy);
        vec4 t9 = P(p, 9);   // flipy, world-sprite flag, camera zoom
        if (t0.x > 0.5 && t9.y > 0.0) {
            vec4 t10 = P(p, 10);   // sprite node centre, screen px
            float d = texelFetch(DepthMap, ivec2(clamp(t10.xy * DepthInfo.x, vec2(0.0), DepthInfo.yz - 1.0)), 0).r;
            if (d > 0.0) {
                d = max(d, DepthInfo.w);
                float zoom = max(t9.z, 1e-6);
                vec3 cam = vec3((sx - ScreenSize.x * 0.5) * d / zoom, (ScreenSize.y * 0.5 - sy) * d / zoom, d);
                gl_Position = ImmersiveMvp * vec4(cam, 1.0);
                gl_Position.z -= Bias.x * float(p) * gl_Position.w;
                vTex = aTex; vOriginalDepth = d; vPrim = p;
                return;
            }
        }
        vec2 plane = vec2(sx / ScreenSize.x - 0.5, 0.5 - sy / ScreenSize.y);
        gl_Position = HudMvp * vec4(plane, 0.0, 1.0);
        vTex = t0.x > 0.5 ? aTex : aTex + 0.5;
        vOriginalDepth = aPos.z;
        vPrim = p;
        return;
    } else if (t0.x > 0.5) {
        sx = aPos.x; sy = aPos.y; w = 1.0; vTex = aTex;
    } else if (t0.y > 0.5) {
        sx = t0.z + aPos.x; sy = t0.w - aPos.y; w = 1.0 / max(aPos.z, 1e-7); vTex = aTex + 0.5;
    } else if (Immersive != 0) {
        float zoom = max(abs(aMeta.x), 1e-6);
        float cameraX = (aPos.x + (t0.z - ScreenSize.x * 0.5) * aPos.z) / zoom;
        float cameraY = (aPos.y + (ScreenSize.y * 0.5 - t0.w) * aPos.z) / zoom;
        vScreen = vec2(t0.z + aPos.x / max(aPos.z, 1e-6), t0.w - aPos.y / max(aPos.z, 1e-6));
        gl_Position = ImmersiveMvp * vec4(cameraX, cameraY, aPos.z, 1.0);
        gl_Position.z -= Bias.x * float(p) * gl_Position.w;
        vTex = aTex + 0.5;
        vOriginalDepth = aPos.z;
        vPrim = p;
        return;
    } else {
        float z = max(aPos.z, 1e-6);
        float x = aPos.x - Bias.y * aMeta.x;
        sx = t0.z + x / z; sy = t0.w - aPos.y / z;
        if (Bias.z > 0.0) sx += Bias.y * aMeta.x / Bias.z;
        w = z; vTex = aTex + 0.5;
    }
    vOriginalDepth = (t0.y > 0.5) ? (1.0 / max(aPos.z, 1e-7)) : aPos.z;
    vPrim = p;
    vScreen = vec2(sx, sy);
    vec2 ndc = vec2(sx / ScreenSize.x * 2.0 - 1.0, sy / ScreenSize.y * 2.0 - 1.0);   // Vulkan: y down = board y
    float zn = w / (w + 5000.0);
    gl_Position = vec4(ndc * w, zn * w, w);
}
