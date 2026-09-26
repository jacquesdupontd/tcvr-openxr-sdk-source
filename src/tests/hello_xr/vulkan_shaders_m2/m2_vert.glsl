#version 450
// The cut-out colour pass tests depth EQUAL against its own pre-pass: positions must be bit-identical.
invariant gl_Position;
out float gl_ClipDistance[4];
// APPSW_DEPTH (26/09): the same placement for everything (HUD and overlays on their plane, scenery at its distance),
// but the REAL depth: the headset's reprojection needs distances, not the painter rank of the colour pass.
// APPSW_MV keeps the colour pass's visibility (painter rank as depth): the vector of a pixel is the one of the polygon
// the player SEES there, not of the nearest one.
#ifdef APPSW_DEPTH
#define DEPTH_ORDER_ON false
#else
#define DEPTH_ORDER_ON (uDepthOrder != 0)
#endif
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aParam;
layout(location = 2) in uint aPrim;

layout(set = 0, binding = 0, std140) uniform M2Uniforms {
    mat4 uMvp;
    mat4 uHudMvp;
    vec2 uViewport;
    vec2 uFocus;
    vec2 uCrtc;
    int uMenuFlat;           // 1: menu screen, all views flat on the arcade plane
    float uFogFar;           // > 0: pop-in fade towards this depth (Model 1)
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
    float uOverlayK;        // screen overlays: enlargement of the arcade frame over the view (1 = flat)
    vec3 uSky;
    float uInterp;          // smooth motion: 0 = previous arcade frame, 1 = current
    vec3 uGround;
    int uAniso;
    int uFilterMode;
    int uMipBias;
    int uAlphaCoverage;
    float uContrast;
    float uBright;
    int uTestStage;
    int uCountOverdraw;
    int uSmpBase;       // 0 = samplers with hardware anisotropy, 4 = without (debug.tcvr.m2_hwAnisoOn)
    int uGammaFolded;   // 1 = colorxlat already holds gamma(colorxlat): skip gamma8()
    int uTexImplicit;   // bit 0: texture() with implicit derivatives; bit 1: use the texture array
    int uBoardLod;      // 1: the board's mip level in the exact path (m2.boardLod)
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
    uint rgb;                 // tcvr_m2_prim::rgb: bit 24 = direct colour 0xRRGGBB (Model 1)
    uint window;              // tcvr_m2_prim::window: the board draws the last window first (25/09)
    uint motion_addr, motion_poly, motion_serial;   // tcvr_m2_prim: smooth motion identity (CPU only, keeps the layout)
};

layout(std430, set = 0, binding = 1) readonly buffer Prims { Prim prims[]; };
layout(std430, set = 0, binding = 2) readonly buffer Palram { uint palram[]; };
// Smooth motion (24/09): each vertex's position in the previous arcade frame (w = 1 when matched), so a display
// refresh between two arcade frames draws the in-between position (uInterp 0..1).
layout(std430, set = 0, binding = 16) readonly buffer PrevPos { vec4 prevPos[]; };

layout(location = 0) out vec3 vParam;
layout(location = 1) flat out uint vPrim;
layout(location = 2) flat out uint vSecondary;
layout(location = 3) out vec2 vBoard;
// The polygon's constant state, handed to the fragment stage once per VERTEX instead of
// being reloaded from the storage buffer for every pixel (120 bytes per fragment before).
layout(location = 4) flat out uvec4 vPA;   // texx, texy, texwidth | texheight << 16, flags
layout(location = 5) flat out uvec4 vPB;   // utexx | utexy << 16, lumabase | luma << 16, colorbase, texlod
layout(location = 6) flat out ivec4 vPC;   // clip l, t, r, b
layout(location = 7) flat out uint vSlot;  // region texture slots: main | microtexture << 16 (0xffff = none)
layout(location = 8) flat out uint vColor; // the polygon's palette entry (palram[colorbase + 0x1000]), 16 bits
layout(location = 9) flat out uint vLayer; // texture-array layers: main | microtexture << 16 (0xffff = none)
#ifdef APPSW_MV
// AppSW motion vectors (26/09): CurrNDC - PrevNDC of this vertex, both through THIS eye's current transform (the head's
// own motion is the headset's business). Zero where the previous position is unknown (prevPos.w = 0).
layout(location = 10) out vec3 vMv;
#endif

void main() {
#ifdef APPSW_MV
    vMv = vec3(0.0);
#endif
    gl_ClipDistance[0] = 1.0; gl_ClipDistance[1] = 1.0; gl_ClipDistance[2] = 1.0; gl_ClipDistance[3] = 1.0;
    vPrim = aPrim;
    {
        Prim q = prims[aPrim];
        uint flags = (q.texsheet & 1u) | ((q.texwrapx & 1u) << 1) | ((q.texwrapy & 1u) << 2) |
                     ((q.texmirrorx & 1u) << 3) | ((q.texmirrory & 1u) << 4) | ((q.translucent != 0u ? 1u : 0u) << 5) |
                     ((q.textured != 0u ? 1u : 0u) << 6) | ((q.checker != 0u ? 1u : 0u) << 7) | ((q.utex != 0u ? 1u : 0u) << 8) |
                     ((q.utexminlod & 15u) << 9);
        vPA = uvec4(q.texx, q.texy, (q.texwidth & 0xffffu) | (q.texheight << 16), flags);
        vPB = uvec4((q.utexx & 0xffffu) | (q.utexy << 16), (q.lumabase & 0xffffu) | (q.luma << 16), q.colorbase, uint(q.texlod));
        vPC = ivec4(q.clip_l, q.clip_t, q.clip_r, q.clip_b);
        vSlot = q.first_vertex;   // the renderer stores the region slots in this otherwise unused field
        vLayer = q.vertex_count;  // ... and the texture-array layers in this one
        uint ci = q.colorbase + 0x1000u;
        // Model 1: the final lit colour is in the primitive (no palette is published for it).
        vColor = ((q.rgb & 0x1000000u) != 0u) ? q.rgb
               : (((ci & 1u) == 0u) ? (palram[ci >> 1] & 0xffffu) : (palram[ci >> 1] >> 16));
    }
    vSecondary = 0u;
    vBoard = aPos;
    vec2 aPosI = aPos; vec3 aParamI = aParam;
    if (uImmersive != 0 && uRaw != 0 && uInterp < 0.999) {
        vec4 pp = prevPos[gl_VertexIndex];
        if (pp.w > 0.5) { aPosI = mix(pp.xy, aPos, uInterp); aParamI.x = mix(pp.z, aParam.x, uInterp); }
    }
    if (uImmersive != 0) {
        Prim p = prims[aPrim];
        float zr = max(aParamI.x, 1e-6);
        vec2 xs = (uRaw != 0) ? vec2(uCrtc.x + float(p.center_x) + aPosI.x / zr,
                                     (384.0 - float(p.center_y)) + uCrtc.y - aPosI.y / zr)
                              : aPosI;
        vBoard = xs;
        if ((p.rgb & 0x2000000u) != 0u) {
            // Screen overlay (fade, hit flash): straight to the screen like the cabinet, enlarged over the view. A
            // letterbox band (bit 26) is the edge of a screen the headset does not have: not drawn in immersive.
#ifdef APPSW_DEPTH
            // a fade or a flash is not a surface: the headset must see the scene behind it
            gl_Position = vec4(0.0, 0.0, -2.0, 1.0); vParam = vec3(0.0); return;
#endif
            if ((p.rgb & 0x4000000u) != 0u && uOverlayK > 1.0) { gl_Position = vec4(0.0, 0.0, -2.0, 1.0); vParam = vec3(0.0); return; }
            // On the arcade screen plane (the HUD's), enlarged uOverlayK times: each eye sees it through its own
            // projection. Straight screen coordinates were the same NDC in both eyes, which the Quest's asymmetric
            // eye frusta turn into a false disparity -- texts that made the player squint (Guillaume, 23/09).
            vec2 plane = vec2(xs.x / uViewport.x - 0.5, 0.5 - xs.y / uViewport.y) * uOverlayK;
            gl_Position = uHudMvp * vec4(plane, 0.0, 1.0);
            if (DEPTH_ORDER_ON) gl_Position.z = ((float(aPrim) + 0.5) / max(uPrimCount, 1.0)) * gl_Position.w;
            vParam = (uRaw != 0) ? vec3(aParamI.y / 8.0, aParamI.z / 8.0, zr) : aParamI;
            return;
        }
        if (uMenuFlat != 0 || p.center_x != uMainCenter.x || p.center_y != uMainCenter.y ||
            abs(p.clip_l - uMainClip.x) > 2 || abs(p.clip_t - uMainClip.y) > 2 ||
            abs(p.clip_r - uMainClip.z) > 2 || abs(p.clip_b - uMainClip.w) > 2) {
            vSecondary = 1u;
            // Cut by the geometry at the view's clip rectangle (26/09): a rear-view mirror's polygons extend far beyond
            // its small window and were shaded in full, then discarded pixel by pixel -- 12 ms of GPU in a race. The
            // fragment test stays (exact at the edges: kept pixels are clip_l <= floor(x) <= clip_r).
            gl_ClipDistance[0] = xs.x - float(p.clip_l);
            gl_ClipDistance[1] = float(p.clip_r) + 1.0 - xs.x;
            gl_ClipDistance[2] = xs.y - float(p.clip_t);
            gl_ClipDistance[3] = float(p.clip_b) + 1.0 - xs.y;
            vec2 plane = vec2(xs.x / uViewport.x - 0.5, 0.5 - xs.y / uViewport.y);
            gl_Position = uHudMvp * vec4(plane, 0.0, 1.0);
            gl_Position.z += uDepthBias * 5.0 * float(aPrim) * gl_Position.w;
            if (DEPTH_ORDER_ON) gl_Position.z = ((float(aPrim) + 0.5) / max(uPrimCount, 1.0)) * gl_Position.w;
            vParam = (uRaw != 0) ? vec3(1.0 / zr, aParamI.y / (8.0 * zr), aParamI.z / (8.0 * zr)) : aParamI;
            return;
        }
        float z = (uRaw != 0) ? aParamI.x
                 : (p.textured != 0u) ? (1.0 / max(aParamI.x, 1e-7)) : max(aParamI.x, 1e-7);
        float X = (uRaw != 0) ? aPosI.x / max(uFocus.x, 1e-6) : (xs.x - uCrtc.x - float(p.center_x)) * z / max(uFocus.x, 1e-6);
        float Y = (uRaw != 0) ? aPosI.y / max(uFocus.y, 1e-6) : ((384.0 - float(p.center_y)) + uCrtc.y - xs.y) * z / max(uFocus.y, 1e-6);
        gl_Position = uMvp * vec4(X, Y, z, 1.0);
#ifdef APPSW_MV
        if (uRaw != 0) {
            vec4 pp = prevPos[gl_VertexIndex];
            if (pp.w > 0.5) {
                vec4 cp = uMvp * vec4(pp.x / max(uFocus.x, 1e-6), pp.y / max(uFocus.y, 1e-6), pp.z, 1.0);
                if (cp.w > 1e-4 && gl_Position.w > 1e-4) vMv = gl_Position.xyz / gl_Position.w - cp.xyz / cp.w;
            }
        }
#endif
        gl_Position.z += uDepthBias * float(p.zsort) * gl_Position.w;
        if (DEPTH_ORDER_ON) gl_Position.z = ((float(aPrim) + 0.5) / max(uPrimCount, 1.0)) * gl_Position.w;
        vParam = (uRaw != 0) ? vec3(aParamI.y / 8.0, aParamI.z / 8.0, z)
                             : vec3(aParamI.y * z, aParamI.z * z, z);
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
