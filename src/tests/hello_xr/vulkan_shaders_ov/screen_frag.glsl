#version 450
// Port of the GLES screen path: Catmull-Rom reconstruction of the arcade framebuffer (the GLES
// "upscale" pass, done here directly while drawing the quad), plus the Time Crisis crosshair and the
// calibration target, same shapes and colours as ScreenFragmentShaderGlsl.
layout(push_constant) uniform PC { mat4 uMvp; vec2 uAim; vec2 uSrcSize; int uAimVisible; int uCalibrating; int uFilter; float uSharpen; };
layout(set = 0, binding = 0) uniform sampler2D uScreen;
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColor;

vec3 src(vec2 uv) { return texture(uScreen, uv).rgb; }

// Catmull-Rom bicubic, nine bilinear taps (identical to the GLES UpscaleFragmentShaderGlsl).
vec3 catmullRom(vec2 uv) {
    vec2 position = uv * uSrcSize;
    vec2 centre = floor(position - 0.5) + 0.5;
    vec2 f = position - centre;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 middle = (centre + w2 / w12) / uSrcSize;
    vec2 first = (centre - 1.0) / uSrcSize;
    vec2 last = (centre + 2.0) / uSrcSize;
    vec3 r = vec3(0.0);
    r += src(vec2(first.x,  first.y))  * w0.x  * w0.y;
    r += src(vec2(middle.x, first.y))  * w12.x * w0.y;
    r += src(vec2(last.x,   first.y))  * w3.x  * w0.y;
    r += src(vec2(first.x,  middle.y)) * w0.x  * w12.y;
    r += src(vec2(middle.x, middle.y)) * w12.x * w12.y;
    r += src(vec2(last.x,   middle.y)) * w3.x  * w12.y;
    r += src(vec2(first.x,  last.y))   * w0.x  * w3.y;
    r += src(vec2(middle.x, last.y))   * w12.x * w3.y;
    r += src(vec2(last.x,   last.y))   * w3.x  * w3.y;
    return clamp(r, 0.0, 1.0);
}

void main() {
    vec3 color;
    if (uFilter == 3 && uSrcSize.x > 0.0) {
        color = catmullRom(vUv);
        if (uSharpen > 0.0) {
            vec2 st = 1.0 / uSrcSize;
            vec3 blur = src(vUv + vec2(st.x, 0.0)) + src(vUv - vec2(st.x, 0.0)) + src(vUv + vec2(0.0, st.y)) + src(vUv - vec2(0.0, st.y));
            color = clamp(color + uSharpen * (color - blur * 0.25), 0.0, 1.0);
        }
    } else {
        color = src(vUv);
    }
    if (uAimVisible != 0) {
        vec2 d = vUv - uAim;
        d.x *= 1.333333;
        float radial = length(d);
        bool ring = radial > 0.011 && radial < 0.015;
        bool vertical = abs(d.x) < 0.002 && abs(d.y) < 0.024;
        bool horizontal = abs(d.y) < 0.002 && abs(d.x) < 0.024;
        if (ring || vertical || horizontal) color = vec3(1.0, 0.12, 0.08);
    }
    if (uCalibrating != 0) {
        vec2 d = (vUv - vec2(0.5)) * vec2(1.333333, 1.0);
        float r = length(d);
        if ((r > 0.021 && r < 0.029) || (abs(d.x) < 0.003 && abs(d.y) < 0.04) || (abs(d.y) < 0.003 && abs(d.x) < 0.04))
            color = vec3(0.1, 1.0, 1.0);
    }
    oColor = vec4(color, 1.0);
}
