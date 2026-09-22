#version 450
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aParam;
layout(location = 2) in uint aPrim;

layout(set = 0, binding = 0, std140) uniform M2Uniforms {
    mat4 uMvp;
    mat4 uHudMvp;
    vec2 uViewport;
    vec2 uFocus;
    vec2 uCrtc;
    ivec4 uMainClip;
    ivec2 uMainCenter;
    float uEyeOffset;
    float uConvergence;
    float uDepthBias;
    float uPrimCount;
    int uImmersive;
    int uDepthOrder;
    int uRaw;
    int uScale;
    int uStipple;
    int uPass;
    int uPrepassClass;
    float uEdgeFade;
    float uHorizonRow;
    vec3 uSky;
    vec3 uGround;
    int uAniso;
    int uFilterMode;
    int uMipBias;
    int uAlphaCoverage;
    float uContrast;
    float uBright;
    int uTestStage;
    int uCountOverdraw;
};

struct Prim {
    uint first_vertex, vertex_count;
    int  clip_l, clip_t, clip_r, clip_b;
    uint textured, translucent, checker;
    uint colorbase, lumabase;
    uint luma;
    int  texlod;
    uint texsheet;
    uint texwidth, texheight, texx, texy;
    uint texwrapx, texwrapy, texmirrorx, texmirrory;
    uint utex, utexminlod, utexx, utexy;
    int  center_x, center_y;
    uint zsort;
};

layout(std430, set = 0, binding = 1) readonly buffer Prims { Prim prims[]; };

layout(location = 0) out vec3 vParam;
layout(location = 1) flat out uint vPrim;
layout(location = 2) flat out uint vSecondary;
layout(location = 3) out vec2 vBoard;

void main() {
    vPrim = aPrim;
    vSecondary = 0u;
    vBoard = aPos;
    if (uImmersive != 0) {
        Prim p = prims[aPrim];
        float zr = max(aParam.x, 1e-6);
        vec2 xs = (uRaw != 0) ? vec2(uCrtc.x + float(p.center_x) + aPos.x / zr,
                                     (384.0 - float(p.center_y)) + uCrtc.y - aPos.y / zr)
                              : aPos;
        vBoard = xs;
        if (p.center_x != uMainCenter.x || p.center_y != uMainCenter.y ||
            abs(p.clip_l - uMainClip.x) > 2 || abs(p.clip_t - uMainClip.y) > 2 ||
            abs(p.clip_r - uMainClip.z) > 2 || abs(p.clip_b - uMainClip.w) > 2) {
            vSecondary = 1u;
            vec2 plane = vec2(xs.x / uViewport.x - 0.5, 0.5 - xs.y / uViewport.y);
            gl_Position = uHudMvp * vec4(plane, 0.0, 1.0);
            gl_Position.z += uDepthBias * 5.0 * float(aPrim) * gl_Position.w;
            if (uDepthOrder != 0) gl_Position.z = ((float(aPrim) + 0.5) / max(uPrimCount, 1.0)) * gl_Position.w;
            vParam = (uRaw != 0) ? vec3(1.0 / zr, aParam.y / (8.0 * zr), aParam.z / (8.0 * zr)) : aParam;
            return;
        }
        float z = (uRaw != 0) ? aParam.x
                 : (p.textured != 0u) ? (1.0 / max(aParam.x, 1e-7)) : max(aParam.x, 1e-7);
        float X = (uRaw != 0) ? aPos.x / max(uFocus.x, 1e-6) : (xs.x - uCrtc.x - float(p.center_x)) * z / max(uFocus.x, 1e-6);
        float Y = (uRaw != 0) ? aPos.y / max(uFocus.y, 1e-6) : ((384.0 - float(p.center_y)) + uCrtc.y - xs.y) * z / max(uFocus.y, 1e-6);
        gl_Position = uMvp * vec4(X, Y, z, 1.0);
        gl_Position.z += uDepthBias * float(p.zsort) * gl_Position.w;
        if (uDepthOrder != 0) gl_Position.z = ((float(aPrim) + 0.5) / max(uPrimCount, 1.0)) * gl_Position.w;
        vParam = (uRaw != 0) ? vec3(aParam.y / 8.0, aParam.z / 8.0, z)
                             : vec3(aParam.y * z, aParam.z * z, z);
        return;
    }
    float ooz = (prims[aPrim].textured != 0u) ? aParam.x : (1.0 / max(aParam.x, 1e-7));
    float xs = aPos.x - uEyeOffset * uFocus.x * ooz;
    if (uConvergence > 0.0) xs += uEyeOffset * uFocus.x / uConvergence;
    vec2 ndc = vec2((xs / uViewport.x) * 2.0 - 1.0,
                    (aPos.y / uViewport.y) * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vParam = aParam;
}
