#version 450
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColorOut;
vec4 oColor;

layout(set = 0, binding = 0) uniform sampler2D uLayer;
// Depth of the 3D in the board's own image (arcade eye-space z of the nearest main-view surface, 0 = nothing), 124 x 96
// over the 496 x 384 board picture, rasterised on the CPU every frame (RasterArcadeDepth).
layout(set = 0, binding = 1) uniform sampler2D uArcDepth;

layout(push_constant) uniform PlanePushConstants {
    mat4 uHudMvp;
    vec2 uOutSize;
    int uKeyZero;
    float uUvScaleX;   // 496/512: only the arcade's columns (the texture's last 16 are empty; they drew black bands, 23/09)
    float uClipRow;    // > 0: back layer cut below this board row, the ground colour shows there (gun games)
    float uNoTile;     // 1: a menu page: inside the arcade frame only
    float uLift;       // brightness curve exponent (menu LUMINOSITE)
    float uZNear;      // nearest depth marched (arcade units)
    vec4 uEyeArc;      // eye in the board camera's space; w = depth of uHudMvp's plane (0 = off)
    vec4 uProj;        // board projection: centre x, centre y, focus x, focus y
};

// The board picture PROJECTED onto the 3D (28/09, one rule for every 2D pixel: Guillaume, "soit tout bug, soit rien"):
// march the eye ray from the nearest depth to infinity until it passes behind the 3D the board drew at that point of
// its picture; the 2D pixel there is the one this eye sees. A HUD letter lies on what it covers, a spark on its impact,
// a piece of 2D scenery on the 3D scenery -- the same for every game, no element, no threshold.
vec2 boardAt(vec3 E, vec3 D, float w) {   // board coords of the ray point at depth 1/w (w -> 0: infinity)
    float z = 1.0 / max(w, 1e-9);
    float s = (z - E.z) / D.z;
    vec3 P = (w > 1e-9) ? E + s * D : D;
    return vec2(uProj.x + uProj.z * P.x / P.z, uProj.y - uProj.w * P.y / P.z);
}
bool projected(float u, float v, out vec2 board) {
    float zr = uEyeArc.w;
    vec3 E = uEyeArc.xyz;
    float bx = u * 496.0 + 248.0, by = 192.0 - v * 384.0;
    vec3 P1 = vec3(zr * (bx - uProj.x) / uProj.z, zr * (uProj.y - by) / uProj.w, zr);
    vec3 D = P1 - E;
    if (D.z <= 1e-6) return false;
    float wNear = 1.0 / max(uZNear, E.z + 1e-3);
    const int N = 24;
    float wPrev = wNear;
    for (int k = 0; k <= N; ++k) {
        float w = wNear * (1.0 - float(k) / float(N));
        vec2 b = boardAt(E, D, w);
        float d = texture(uArcDepth, b / vec2(496.0, 384.0)).r;
        if (d > 0.0 && 1.0 / max(w, 1e-9) >= d && b.x >= 0.0 && b.y >= 0.0 && b.x <= 496.0 && b.y <= 384.0) {
            float lo = wPrev, hi = w;   // lo: in front of the surface, hi: behind it
            for (int i = 0; i < 6; ++i) {
                float m = 0.5 * (lo + hi);
                vec2 bm = boardAt(E, D, m);
                float dm = texture(uArcDepth, bm / vec2(496.0, 384.0)).r;
                if (dm > 0.0 && 1.0 / max(m, 1e-9) >= dm) hi = m; else lo = m;
            }
            board = boardAt(E, D, hi);
            return true;
        }
        wPrev = w;
    }
    board = boardAt(E, D, 0.0);   // nothing behind: the 2D pixel at infinity
    return true;
}


void main_body() {
    vec2 ndc = gl_FragCoord.xy / uOutSize * 2.0 - 1.0;
    vec4 c0 = uHudMvp[0], c1 = uHudMvp[1], c3 = uHudMvp[3];
    float a11 = c0.x - ndc.x * c0.w, a12 = c1.x - ndc.x * c1.w, b1 = ndc.x * c3.w - c3.x;
    float a21 = c0.y - ndc.y * c0.w, a22 = c1.y - ndc.y * c1.w, b2 = ndc.y * c3.w - c3.y;
    float det = a11 * a22 - a12 * a21;
    if (abs(det) < 1e-12) discard;
    float u = (b1 * a22 - a12 * b2) / det, v = (a11 * b2 - a21 * b1) / det;
    if (u * c0.w + v * c1.w + c3.w <= 0.0) discard;
    if (uKeyZero != 0 && uEyeArc.w > 0.0) {
        vec2 b;
        if (!projected(u, v, b)) discard;
        u = (b.x - 248.0) / 496.0; v = (192.0 - b.y) / 384.0;
    }
    bool inside = abs(u) <= 0.5 && abs(v) <= 0.5;
    if ((uKeyZero != 0 || uNoTile > 0.5) && !inside) discard;
    // Back layer beyond the arcade frame: repeated MIRRORED (identity inside the frame), so the edges meet
    // without a seam; a plain repeat put the image's left edge against its right one (a hard line in the sky).
    float t = u + 0.5, m = fract(t * 0.5) * 2.0;
    vec2 uv = (uKeyZero != 0) ? clamp(vec2(u + 0.5, 0.5 - v), 0.0, 1.0) : vec2(m <= 1.0 ? m : 2.0 - m, clamp(0.5 - v, 0.0, 1.0));
    // Stay half a texel inside the real columns: the bilinear filter at the last one otherwise blends in the
    // first empty (black) column -- a thin dark line at every mirror seam of the sky.
    float halfTexel = 0.5 / float(textureSize(uLayer, 0).x);
    uv.x = clamp(uv.x * uUvScaleX, halfTexel, uUvScaleX - halfTexel);
    if (uClipRow > 0.0 && uKeyZero == 0 && uv.y * 384.0 > uClipRow) discard;
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
    if (uLift > 0.0 && uLift != 1.0) d = pow(d, vec3(uLift));   // menu LUMINOSITE (25/09)
    oColorOut = vec4(d * (d * (d * (d * -0.23012586 + 0.73328061) + 0.44219034) + 0.05115943), oColor.a);   // sRGB->linear, 4 FMA, <= 1.2/255 after re-encoding (fitted 23/09)
}
