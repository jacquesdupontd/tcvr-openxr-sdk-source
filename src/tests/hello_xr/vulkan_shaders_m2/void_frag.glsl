#version 450
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColorOut;
vec4 oColor;

layout(push_constant) uniform VoidPushConstants {
    mat4 uInvMvp;
    vec4 uSky;
    vec4 uGround;
    vec2 uOutSize;
    float uLift;       // brightness curve exponent (menu LUMINOSITE)
};

void main_body() {
    vec2 ndc = gl_FragCoord.xy / uOutSize * 2.0 - 1.0;
    vec4 a = uInvMvp * vec4(ndc, -1.0, 1.0);
    vec4 b = uInvMvp * vec4(ndc,  1.0, 1.0);
    vec3 dir = b.xyz / b.w - a.xyz / a.w;
    float up = dir.y / max(length(dir), 1e-6);
    oColor = vec4(mix(uGround.xyz, uSky.xyz, smoothstep(-0.03, 0.03, up)), 1.0);
}

// Arcade colours are display values (CRT-referred). The eye image is an _SRGB attachment that encodes what we
// write: decode first, so the stored byte is the arcade's (true colours, as the GL build and MAME show) without
// the costly MUTABLE_FORMAT raw views (they disable framebuffer compression: -3 ms on Sega Rally, 23/09).
void main() {
    main_body();
    vec3 d = clamp(oColor.rgb, 0.0, 1.0);
    if (uLift > 0.0 && uLift != 1.0) d = pow(d, vec3(uLift));   // menu LUMINOSITE (25/09)
    oColorOut = vec4(d * (d * (d * (d * -0.23012586 + 0.73328061) + 0.44219034) + 0.05115943), oColor.a);   // sRGB->linear, 4 FMA, <= 1.2/255 after re-encoding (fitted 23/09)
}
