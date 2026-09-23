#version 450
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColorOut;
vec4 oColor;

layout(set = 0, binding = 0) uniform sampler2D uLayer;

layout(push_constant) uniform PlanePushConstants {
    mat4 uHudMvp;
    vec2 uOutSize;
    int uKeyZero;
};

void main_body() {
    vec2 ndc = gl_FragCoord.xy / uOutSize * 2.0 - 1.0;
    vec4 c0 = uHudMvp[0], c1 = uHudMvp[1], c3 = uHudMvp[3];
    float a11 = c0.x - ndc.x * c0.w, a12 = c1.x - ndc.x * c1.w, b1 = ndc.x * c3.w - c3.x;
    float a21 = c0.y - ndc.y * c0.w, a22 = c1.y - ndc.y * c1.w, b2 = ndc.y * c3.w - c3.y;
    float det = a11 * a22 - a12 * a21;
    if (abs(det) < 1e-12) discard;
    float u = (b1 * a22 - a12 * b2) / det, v = (a11 * b2 - a21 * b1) / det;
    if (u * c0.w + v * c1.w + c3.w <= 0.0) discard;
    bool inside = abs(u) <= 0.5 && abs(v) <= 0.5;
    if (uKeyZero != 0 && !inside) discard;
    vec2 uv = (uKeyZero != 0) ? clamp(vec2(u + 0.5, 0.5 - v), 0.0, 1.0) : vec2(fract(u + 0.5), clamp(0.5 - v, 0.0, 1.0));
    vec4 c = texture(uLayer, uv);
    if (uKeyZero != 0) {
        float a = clamp(c.a, 0.0, 1.0);
        if (a < 0.02) discard;
        oColor = vec4(clamp(c.rgb / max(a, 0.02), 0.0, 1.0), a);
        return;
    }
    oColor = vec4(c.rgb, 1.0);
}

// Arcade colours are display values (CRT-referred). The eye image is an _SRGB attachment that encodes what we
// write: decode first, so the stored byte is the arcade's (true colours, as the GL build and MAME show) without
// the costly MUTABLE_FORMAT raw views (they disable framebuffer compression: -3 ms on Sega Rally, 23/09).
void main() {
    main_body();
    vec3 d = clamp(oColor.rgb, 0.0, 1.0);
    oColorOut = vec4(d * (d * (d * (d * -0.23012586 + 0.73328061) + 0.44219034) + 0.05115943), oColor.a);   // sRGB->linear, 4 FMA, <= 1.2/255 after re-encoding (fitted 23/09)
}
