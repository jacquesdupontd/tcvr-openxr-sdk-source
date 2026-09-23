#version 450
#extension GL_EXT_nonuniform_qualifier : require
precision highp float;
precision highp int;

layout(location = 0) in vec3 vParam;
layout(location = 1) flat in uint vPrim;
layout(location = 2) flat in uint vSecondary;
layout(location = 3) in vec2 vBoard;
layout(location = 4) flat in uvec4 vPA;
layout(location = 5) flat in uvec4 vPB;
layout(location = 6) flat in ivec4 vPC;
layout(location = 7) flat in uint vSlot;
layout(location = 8) flat in uint vColor;
layout(location = 9) flat in uint vLayer;
layout(location = 0) out vec4 oColorOut;
vec4 oColor;

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
};

layout(std430, set = 0, binding = 1) readonly buffer Prims   { Prim prims[]; };
layout(std430, set = 0, binding = 2) readonly buffer Palram    { uint palram[]; };
layout(std430, set = 0, binding = 3) readonly buffer Colorxlat { uint colorxlat[]; };
layout(std430, set = 0, binding = 4) readonly buffer Lumaram   { uint lumaram[]; };
layout(std430, set = 0, binding = 5) readonly buffer Gamma     { uint gammaTab[]; };
// Debug counters (uTestStage 3): [0] fragments shaded by the discard-free opaque pass (after the
// early depth test), [1] by the others. Read back and reset by the renderer (TCVR_M2VK overdraw).
layout(std430, set = 0, binding = 6) buffer DebugCounters { uint dbg[]; };
layout(std430, set = 0, binding = 7) readonly buffer Unused7 { uint unused7[]; };

// Texture sheets, one texel per byte, R8_UNORM holding t*17 (t = the 4-bit Model 2 texel):
// texelFetch gives the exact texel back, and the SAME image can be sampled with the
// hardware bilinear filter, which is what the new path (uFilterMode 5) does.
layout(set = 0, binding = 8) uniform sampler2D uSheetTex0;
layout(set = 0, binding = 9) uniform sampler2D uSheetTex1;
// Cut-out form of the same sheets, RG8: R = t*17 premultiplied by opacity, G = opacity.
layout(set = 0, binding = 10) uniform sampler2D uCutTex0;
layout(set = 0, binding = 11) uniform sampler2D uCutTex1;
// Every texture region as its own mipmapped image (see vulkan_m2_regions.h): R = t*17,
// G = t*17*opaque, B = opaque. Samplers: [mirrorx | mirrory << 1], repeat otherwise, hw aniso.
layout(set = 0, binding = 12) uniform texture2D uRegion[512];
layout(set = 0, binding = 13) uniform sampler uRegionSmp[8];
// Every region <= 256x256 as one TILED 256x256 layer of a single array bound the ordinary way.
layout(set = 0, binding = 15) uniform sampler2DArray uRegionArr;
// binding 14 (colour table image) is still declared by the renderer but unused: a precomputed
// gamma(colorxlat) table, as a texture (~2x) or as a storage buffer (~3x), made the WHOLE shader
// slower on the Adreno 740 (measured 22/09, interleaved A/B). Kept out of the shader on purpose.

uint u16at(uint arr_index_hi_lo, uint packed) {
    return (arr_index_hi_lo == 0u) ? (packed & 0xffffu) : (packed >> 16);
}
uint palram16(uint i)    { return u16at(i & 1u, palram[i >> 1]); }
uint colorxlat16(uint i) { return u16at(i & 1u, colorxlat[i >> 1]); }
uint lumaram8(uint i)    { return (lumaram[i >> 2] >> (8u * (i & 3u))) & 0xffu; }
uint gamma8(uint i)      { return (gammaTab[i >> 2] >> (8u * (i & 3u))) & 0xffu; }

uint get_texel(uint base_x, uint base_y, int x, int y, uint sheet) {
    int x2 = int(base_x) + x;
    int y2 = int(base_y) + y;
    if (x2 >= 1024) {
        x2 -= 1024;
        y2 ^= 1024;
    }
    ivec2 at = ivec2(x2, y2);
    float f = (sheet == 0u) ? texelFetch(uSheetTex0, at, 0).r : texelFetch(uSheetTex1, at, 0).r;
    return uint(f * 15.0 + 0.5);
}

uint lerp_packed(uint x, uint y, uint a) {
    return (x + (((y - x) * a) >> 8)) & 0x00ff00ffu;
}

uint fetch_bilinear_texel(Prim p, int miplevel, int u, int v, bool translucent) {
    uint tex_width, tex_height, tex_x, tex_y, sheet;
    if (miplevel == -1) {
        tex_width = 128u; tex_height = 128u;
        tex_x = p.utexx; tex_y = p.utexy;
        sheet = 1u - p.texsheet;
        u <<= (1 << int(p.utexminlod));
        v <<= (1 << int(p.utexminlod));
    } else {
        tex_width  = p.texwidth  >> uint(miplevel);
        tex_height = p.texheight >> uint(miplevel);
        tex_x = ((p.texx - 2048u) >> uint(miplevel)) & 2047u;
        tex_y = ((p.texy - 1024u) >> uint(miplevel)) & 1023u;
        sheet = ((miplevel & 1) == 0) ? p.texsheet : (1u - p.texsheet);
        u >>= miplevel;
        v >>= miplevel;
    }
    if (p.texmirrorx != 0u && (u & int(tex_width  << 8)) != 0) u = ~u;
    if (p.texmirrory != 0u && (v & int(tex_height << 8)) != 0) v = ~v;
    u -= 0x80;
    v -= 0x80;
    uint ufrac = uint(u) & 0xffu;
    uint vfrac = uint(v) & 0xffu;
    int u0 = (u >> 8) & int(tex_width  - 1u);
    int u1 = (u0 + 1) & int(tex_width  - 1u);
    int v0 = (v >> 8) & int(tex_height - 1u);
    int v1 = (v0 + 1) & int(tex_height - 1u);
    if (p.texwrapx == 0u && u1 == 0) {
        if (ufrac >= 0x80u) { u0 = u1; u1 += 1; ufrac = 0u; }
        else                { u1 = u0; u0 -= 1; ufrac = 0x100u; }
    }
    if (p.texwrapy == 0u && v1 == 0) {
        if (vfrac >= 0x80u) { v0 = 0; v1 += 1; vfrac = 0u; }
        else                { v1 = v0; v0 -= 1; vfrac = 0x100u; }
    }
    uint tex00 = get_texel(tex_x, tex_y, u0, v0, sheet) << 4;
    uint tex01 = get_texel(tex_x, tex_y, u1, v0, sheet) << 4;
    uint tex10 = get_texel(tex_x, tex_y, u0, v1, sheet) << 4;
    uint tex11 = get_texel(tex_x, tex_y, u1, v1, sheet) << 4;
    if (!translucent) {
        return lerp_packed(lerp_packed(tex00, tex01, ufrac), lerp_packed(tex10, tex11, ufrac), vfrac);
    }
    if (tex00 != 0xf0u) tex00 |= 0x00800000u;
    if (tex01 != 0xf0u) tex01 |= 0x00800000u;
    if (tex10 != 0xf0u) tex10 |= 0x00800000u;
    if (tex11 != 0xf0u) tex11 |= 0x00800000u;
    if (tex00 == 0x000000f0u) tex00 = tex01 & 0xffu;
    if (tex01 == 0x000000f0u) tex01 = tex00 & 0xffu;
    if (tex10 == 0x000000f0u) tex10 = tex11 & 0xffu;
    if (tex11 == 0x000000f0u) tex11 = tex10 & 0xffu;
    uint tex0x = lerp_packed(tex00, tex01, ufrac);
    uint tex1x = lerp_packed(tex10, tex11, ufrac);
    if (tex0x == 0x000000f0u) tex0x = tex1x & 0xffu;
    if (tex1x == 0x000000f0u) tex1x = tex0x & 0xffu;
    return lerp_packed(tex0x, tex1x, vfrac);
}

const uint kLog2Table[128] = uint[128](
      0u,   2u,   5u,   8u,  11u,  14u,  16u,  19u,  22u,  25u,  27u,  30u,  33u,  35u,  38u,  40u,
     43u,  46u,  48u,  51u,  53u,  56u,  58u,  61u,  63u,  65u,  68u,  70u,  73u,  75u,  77u,  80u,
     82u,  84u,  87u,  89u,  91u,  93u,  96u,  98u, 100u, 102u, 104u, 106u, 109u, 111u, 113u, 115u,
    117u, 119u, 121u, 123u, 125u, 127u, 129u, 132u, 134u, 136u, 138u, 140u, 141u, 143u, 145u, 147u,
    149u, 151u, 153u, 155u, 157u, 159u, 161u, 162u, 164u, 166u, 168u, 170u, 172u, 173u, 175u, 177u,
    179u, 181u, 182u, 184u, 186u, 188u, 189u, 191u, 193u, 194u, 196u, 198u, 200u, 201u, 203u, 205u,
    206u, 208u, 209u, 211u, 213u, 214u, 216u, 218u, 219u, 221u, 222u, 224u, 225u, 227u, 229u, 230u,
    232u, 233u, 235u, 236u, 238u, 239u, 241u, 242u, 244u, 245u, 247u, 248u, 250u, 251u, 253u, 254u);

int fast_log2(float value) {
    if (value < 0.0) return 0;
    uint ival = floatBitsToUint(value) >> 16;
    int exp = int(ival >> 7) - 127;
    return (exp << 8) | int(kLog2Table[ival & 127u]);
}

vec3 shade(Prim p, uint luma, uint palmask) {
    uint color = palram16(p.colorbase + 0x1000u) & palmask;
    uint r = colorxlat16((0x0000u / 2u) + (((color >>  0) & 0x1fu) << 8) + luma) & 0xffu;
    uint g = colorxlat16((0x4000u / 2u) + (((color >>  5) & 0x1fu) << 8) + luma) & 0xffu;
    uint b = colorxlat16((0x8000u / 2u) + (((color >> 10) & 0x1fu) << 8) + luma) & 0xffu;
    if (uGammaFolded != 0) return vec3(float(r), float(g), float(b)) / 255.0;
    return vec3(float(gamma8(r)), float(gamma8(g)), float(gamma8(b))) / 255.0;
}

// Same as shade(), with the polygon's palette colour handed in flat by the vertex stage.
vec3 shade_c(uint color, uint luma) {
    if ((color & 0x1000000u) != 0u)   // Model 1 direct colour, already lit and in display space
        return vec3(float((color >> 16) & 0xffu), float((color >> 8) & 0xffu), float(color & 0xffu)) / 255.0;
    uint r = colorxlat16((0x0000u / 2u) + (((color >>  0) & 0x1fu) << 8) + luma) & 0xffu;
    uint g = colorxlat16((0x4000u / 2u) + (((color >>  5) & 0x1fu) << 8) + luma) & 0xffu;
    uint b = colorxlat16((0x8000u / 2u) + (((color >> 10) & 0x1fu) << 8) + luma) & 0xffu;
    if (uGammaFolded != 0) return vec3(float(r), float(g), float(b)) / 255.0;
    return vec3(float(gamma8(r)), float(gamma8(g)), float(gamma8(b))) / 255.0;
}

uint fetch_nearest_texel(Prim p, int miplevel, int u, int v, bool translucent) {
    uint tex_width, tex_height, tex_x, tex_y, sheet;
    if (miplevel == -1) {
        tex_width = 128u; tex_height = 128u;
        tex_x = p.utexx; tex_y = p.utexy;
        sheet = 1u - p.texsheet;
        u <<= (1 << int(p.utexminlod));
        v <<= (1 << int(p.utexminlod));
    } else {
        tex_width  = p.texwidth  >> uint(miplevel);
        tex_height = p.texheight >> uint(miplevel);
        tex_x = ((p.texx - 2048u) >> uint(miplevel)) & 2047u;
        tex_y = ((p.texy - 1024u) >> uint(miplevel)) & 1023u;
        sheet = ((miplevel & 1) == 0) ? p.texsheet : (1u - p.texsheet);
        u >>= miplevel;
        v >>= miplevel;
    }
    if (p.texmirrorx != 0u && (u & int(tex_width  << 8)) != 0) u = ~u;
    if (p.texmirrory != 0u && (v & int(tex_height << 8)) != 0) v = ~v;
    int u0 = (u >> 8) & int(tex_width  - 1u);
    int v0 = (v >> 8) & int(tex_height - 1u);
    uint tex00 = get_texel(tex_x, tex_y, u0, v0, sheet) << 4;
    if (translucent) {
        if (tex00 != 0xf0u) tex00 |= 0x00800000u;
    }
    return tex00;
}

vec4 texel_rgb(Prim p, uint tPacked, bool translucent) {
    float keep = 1.0;
    uint t = tPacked;
    if (translucent) { if (t < 0x00400000u) keep = 0.0; t &= 0xffu; }
    uint luma = (lumaram8(p.lumabase + (t >> 1)) * p.luma) / 256u;
    luma = min(luma, 0x3fu);
    return vec4(shade(p, luma, 0x7fffu), keep);
}

vec4 fetch_bilinear_rgb(Prim p, int miplevel, int u, int v, bool translucent) {
    uint tex_width, tex_height, tex_x, tex_y, sheet;
    if (miplevel == -1) {
        tex_width = 128u; tex_height = 128u;
        tex_x = p.utexx; tex_y = p.utexy;
        sheet = 1u - p.texsheet;
        u <<= (1 << int(p.utexminlod));
        v <<= (1 << int(p.utexminlod));
    } else {
        tex_width  = p.texwidth  >> uint(miplevel);
        tex_height = p.texheight >> uint(miplevel);
        tex_x = ((p.texx - 2048u) >> uint(miplevel)) & 2047u;
        tex_y = ((p.texy - 1024u) >> uint(miplevel)) & 1023u;
        sheet = ((miplevel & 1) == 0) ? p.texsheet : (1u - p.texsheet);
        u >>= miplevel;
        v >>= miplevel;
    }
    if (p.texmirrorx != 0u && (u & int(tex_width  << 8)) != 0) u = ~u;
    if (p.texmirrory != 0u && (v & int(tex_height << 8)) != 0) v = ~v;
    u -= 0x80;
    v -= 0x80;
    uint ufrac = uint(u) & 0xffu;
    uint vfrac = uint(v) & 0xffu;
    int u0 = (u >> 8) & int(tex_width  - 1u);
    int u1 = (u0 + 1) & int(tex_width  - 1u);
    int v0 = (v >> 8) & int(tex_height - 1u);
    int v1 = (v0 + 1) & int(tex_height - 1u);
    if (p.texwrapx == 0u && u1 == 0) {
        if (ufrac >= 0x80u) { u0 = u1; u1 += 1; ufrac = 0u; }
        else                { u1 = u0; u0 -= 1; ufrac = 0x100u; }
    }
    if (p.texwrapy == 0u && v1 == 0) {
        if (vfrac >= 0x80u) { v0 = 0; v1 += 1; vfrac = 0u; }
        else                { v1 = v0; v0 -= 1; vfrac = 0x100u; }
    }
    uint t00 = get_texel(tex_x, tex_y, u0, v0, sheet) << 4;
    uint t01 = get_texel(tex_x, tex_y, u1, v0, sheet) << 4;
    uint t10 = get_texel(tex_x, tex_y, u0, v1, sheet) << 4;
    uint t11 = get_texel(tex_x, tex_y, u1, v1, sheet) << 4;
    if (translucent) {
        if (t00 != 0xf0u) t00 |= 0x00800000u;
        if (t01 != 0xf0u) t01 |= 0x00800000u;
        if (t10 != 0xf0u) t10 |= 0x00800000u;
        if (t11 != 0xf0u) t11 |= 0x00800000u;
    }
    vec4 c00 = texel_rgb(p, t00, translucent);
    vec4 c01 = texel_rgb(p, t01, translucent);
    vec4 c10 = texel_rgb(p, t10, translucent);
    vec4 c11 = texel_rgb(p, t11, translucent);
    float fu = float(ufrac) / 256.0;
    float fv = float(vfrac) / 256.0;
    return mix(mix(c00, c01, fu), mix(c10, c11, fu), fv);
}

vec4 shaded_sample(Prim p, int level, int max_level, int mml, int u, int v, bool translucent) {
    if (uFilterMode == 3) return fetch_bilinear_rgb(p, level, u, v, translucent);
    if (uFilterMode == 4) {
        vec4 a = fetch_bilinear_rgb(p, level, u, v, translucent);
        if (mml > 0 && level < max_level) {
            vec4 b = fetch_bilinear_rgb(p, level + 1, u, v, translucent);
            return mix(a, b, float(mml & 127) / 128.0);
        }
        return a;
    }
    uint t;
    if (uFilterMode == 0) {
        t = fetch_nearest_texel(p, level, u, v, translucent);
    } else if (uFilterMode == 1) {
        t = fetch_bilinear_texel(p, level, u, v, translucent);
    } else {
        t = fetch_bilinear_texel(p, level, u, v, translucent);
        if (mml > 0 && level < max_level) {
            uint t2 = fetch_bilinear_texel(p, level + 1, u, v, translucent);
            t = lerp_packed(t, t2, uint((mml & 127) << 1));
        } else if (p.utex != 0u && mml < 0) {
            uint t2 = fetch_bilinear_texel(p, -1, u, v, translucent);
            t = lerp_packed(t, t2, uint(min(-mml >> int(p.utexminlod), 127)));
        }
    }
    float keep = 1.0;
    if (translucent) { if (t < 0x00400000u) keep = 0.0; t &= 0xffu; }
    uint luma = (lumaram8(p.lumabase + (t >> 1)) * p.luma) / 256u;
    luma = min(luma, 0x3fu);
    return vec4(shade(p, luma, 0x7fffu), keep);
}


// ---------------------------------------------------------------------------------------
// uFilterMode 5 -- "the board's own filter, done by the texture unit" (opus55, 22/09).
//
// On the real Model 2 the filtered quantity is t, the 4-bit texel (an intensity that indexes
// the tone curve lumaram); colour comes AFTER, once, from the polygon's palette entry.
// So: filter t with the hardware sampler (bilinear inside a mip level, trilinear between
// the board's own mip levels, anisotropic taps along the pixel footprint), then run the
// colour chain ONCE per pixel. Same arithmetic order as MAME, a fraction of the fetches.
//
// The mip level comes from the SCREEN derivatives of the texel coordinates -- the real
// density of the headset's pixels -- not from the board's depth LOD, which was calibrated
// for a 496x384 screen and picked mips that shimmered (too sharp) or smeared under a
// 2000-pixel-wide eye. That is the anti-shimmer.
// ---------------------------------------------------------------------------------------

// Storage position of texel (x, y) of the logical 2048x1024 sheet (MAME get_texel fold).
ivec2 sheet_at(int x2, int y2) {
    if (x2 >= 1024) { x2 -= 1024; y2 ^= 1024; }
    return ivec2(x2, y2);
}

// One mip level at continuous LEVEL-0 texel coordinate tc. Returns (t, 1) for an opaque
// texture, or (t premultiplied by opacity, opacity) for a cut-out: the caller divides.
vec2 level_s(Prim p, int L, vec2 tc, bool cut) {
    uint w = p.texwidth >> uint(L), h = p.texheight >> uint(L);
    int ox = int(((p.texx - 2048u) >> uint(L)) & 2047u);
    int oy = int(((p.texy - 1024u) >> uint(L)) & 1023u);
    uint sheet = ((L & 1) == 0) ? p.texsheet : (1u - p.texsheet);
    vec2 c = tc / float(1 << L);
    vec2 wh = vec2(float(w), float(h));
    if (p.texmirrorx != 0u) { float m = mod(c.x, 2.0 * wh.x); c.x = (m >= wh.x) ? (2.0 * wh.x - m) : m; }
    if (p.texmirrory != 0u) { float m = mod(c.y, 2.0 * wh.y); c.y = (m >= wh.y) ? (2.0 * wh.y - m) : m; }
    c -= 0.5;
    vec2 i0f = floor(c);
    vec2 f = c - i0f;
    int u0 = int(mod(i0f.x, wh.x)), v0 = int(mod(i0f.y, wh.y));
    // Interior of the region, both taps inside and no sheet fold crossed: one hardware tap.
    ivec2 b0 = sheet_at(ox + u0, oy + v0);
    ivec2 b1 = sheet_at(ox + u0 + 1, oy + v0 + 1);
    if (u0 + 1 < int(w) && v0 + 1 < int(h) && b1 == b0 + ivec2(1)) {
        vec2 uv = (vec2(b0) + 0.5 + f) / vec2(1024.0, 4096.0);
        if (cut) {
            vec2 rg = (sheet == 0u) ? textureLod(uCutTex0, uv, 0.0).rg : textureLod(uCutTex1, uv, 0.0).rg;
            return vec2(rg.r * 15.0, rg.g);
        }
        float r = (sheet == 0u) ? textureLod(uSheetTex0, uv, 0.0).r : textureLod(uSheetTex1, uv, 0.0).r;
        return vec2(r * 15.0, 1.0);
    }
    // Seam of a tiled / clamped region: the board's own 4 taps (wrap, or clamp when off).
    int u1 = (u0 + 1) % int(w), v1 = (v0 + 1) % int(h);
    if (p.texwrapx == 0u && u1 == 0) { if (f.x >= 0.5) { u0 = 0; u1 = 1; f.x = 0.0; } else { u1 = u0; u0 = u0 - 1; f.x = 1.0; } }
    if (p.texwrapy == 0u && v1 == 0) { if (f.y >= 0.5) { v0 = 0; v1 = 1; f.y = 0.0; } else { v1 = v0; v0 = v0 - 1; f.y = 1.0; } }
    float t00 = float(get_texel(uint(ox), uint(oy), u0, v0, sheet));
    float t01 = float(get_texel(uint(ox), uint(oy), u1, v0, sheet));
    float t10 = float(get_texel(uint(ox), uint(oy), u0, v1, sheet));
    float t11 = float(get_texel(uint(ox), uint(oy), u1, v1, sheet));
    if (!cut) return vec2(mix(mix(t00, t01, f.x), mix(t10, t11, f.x), f.y), 1.0);
    vec4 a = vec4(t00 != 15.0, t01 != 15.0, t10 != 15.0, t11 != 15.0);
    vec4 tv = vec4(t00, t01, t10, t11) * a;
    return vec2(mix(mix(tv.x, tv.y, f.x), mix(tv.z, tv.w, f.x), f.y), mix(mix(a.x, a.y, f.x), mix(a.z, a.w, f.x), f.y));
}
float level_t(Prim p, int L, vec2 tc) { return level_s(p, L, tc, false).x; }

// Microtexture (detail texture, 128x128 on the other sheet) at level-0 coordinate tc.
float micro_t(Prim p, vec2 tc) {
    vec2 c = tc * float(1 << (1 << int(p.utexminlod))) - 0.5;
    vec2 i0f = floor(c); vec2 f = c - i0f;
    int u0 = int(mod(i0f.x, 128.0)), v0 = int(mod(i0f.y, 128.0));
    int u1 = (u0 + 1) & 127, v1 = (v0 + 1) & 127;
    uint sheet = 1u - p.texsheet;
    float t00 = float(get_texel(p.utexx, p.utexy, u0, v0, sheet));
    float t01 = float(get_texel(p.utexx, p.utexy, u1, v0, sheet));
    float t10 = float(get_texel(p.utexx, p.utexy, u0, v1, sheet));
    float t11 = float(get_texel(p.utexx, p.utexy, u1, v1, sheet));
    return mix(mix(t00, t01, f.x), mix(t10, t11, f.x), f.y);
}

// Trilinear sample at coordinate tc for a fractional level lod (already clamped >= 0).
vec2 tri_s(Prim p, vec2 tc, float lod, int max_level, bool cut) {
    int L = min(int(lod), max_level);
    vec2 t = level_s(p, L, tc, cut);
    float fr = lod - float(L);
    if (L < max_level && fr > 0.004) t = mix(t, level_s(p, L + 1, tc, cut), fr);
    return t;
}

// Region path: one hardware trilinear(+anisotropic) sample of the region's own image.
// grads are the level-0 texel-space derivatives; bias in mip levels.
vec4 region_sample(uint slot, uint smp, vec2 tc, vec2 size, vec2 gx, vec2 gy, float bias) {
    // A 2x2 quad never spans two polygons, so implicit derivatives are valid in this per-polygon
    // branch; explicit gradients take a slower path on many mobile GPUs.
    if (uTestStage == 4) return texture(sampler2D(uRegion[0], uRegionSmp[0]), tc / size, bias);   // bench: uniform index
    if ((uTexImplicit & 1) != 0) return texture(sampler2D(uRegion[nonuniformEXT(slot)], uRegionSmp[smp]), tc / size, bias);
    float k = exp2(bias);
    return textureGrad(sampler2D(uRegion[nonuniformEXT(slot)], uRegionSmp[smp]), tc / size, gx * k / size, gy * k / size);
}

// The polygon's colour at a given luma through the precomputed table: 3 texel reads.
// Colour chain, once: filtered t -> lumaram tone curve -> palette -> colorxlat -> gamma.
vec3 tone(Prim p, float t) {
    uint t8 = min(uint(t * 16.0 + 0.5), 0xf0u);
    uint luma = min((lumaram8(p.lumabase + (t8 >> 1)) * p.luma) / 256u, 0x3fu);
    return shade_c(vColor & 0x7fffu, luma);
}

// The polygon, rebuilt from the flat inputs: registers, not a 120-byte load per pixel.
Prim flatPrim() {
    Prim p;
    p.first_vertex = 0u; p.vertex_count = 0u;
    p.clip_l = vPC.x; p.clip_t = vPC.y; p.clip_r = vPC.z; p.clip_b = vPC.w;
    uint fl = vPA.w;
    p.textured = (fl >> 6) & 1u; p.translucent = (fl >> 5) & 1u; p.checker = (fl >> 7) & 1u;
    p.colorbase = vPB.z; p.lumabase = vPB.y & 0xffffu; p.luma = vPB.y >> 16;
    p.texlod = int(vPB.w);
    p.texsheet = fl & 1u;
    p.texwidth = vPA.z & 0xffffu; p.texheight = vPA.z >> 16; p.texx = vPA.x; p.texy = vPA.y;
    p.texwrapx = (fl >> 1) & 1u; p.texwrapy = (fl >> 2) & 1u; p.texmirrorx = (fl >> 3) & 1u; p.texmirrory = (fl >> 4) & 1u;
    p.utex = (fl >> 8) & 1u; p.utexminlod = (fl >> 9) & 15u; p.utexx = vPB.x & 0xffffu; p.utexy = vPB.x >> 16;
    p.center_x = 0; p.center_y = 0; p.zsort = 0u;
    return p;
}

#if defined(NO_DISCARD) || defined(CUT_COLOR)
// No discard, no depth write from the shader: the depth test ALWAYS runs before shading, even
// with the debug counter's side effect. CUT_COLOR = colour pass of the cut-outs after their depth
// pre-pass (depth EQUAL): only the visible sample of each pixel is shaded, once.
layout(early_fragment_tests) in;
#endif
#ifdef CUT_COLOR
#define NO_DISCARD_KEEP_TRANSLUCENT 1
#endif

#ifdef LEAN
// LEAN opaque pass: only what the discard-free polygons of the main view need, nothing else, so the
// compiler allocates few registers and many pixel groups run in parallel (texture latency hidden).
void main_body() {
    uint fl = vPA.w;
    float acov = 1.0;
    float zb = max(vParam.z, 1e-6);
    vec2 tc = vec2(vParam.x, vParam.y);                 // main view: (u, v) in level-0 texels
    vec3 rgb;
    if (((fl >> 6) & 1u) == 0u) {
        rgb = shade_c(vColor, (vPB.y >> 16) >> 2);        // untextured: luma >> 2
    } else {
        uint layer = vLayer & 0xffffu;
        float bias = float(uMipBias) / 128.0;
        vec4 c;
        if (layer != 0xffffu && ((fl >> 3) & 3u) == 0u) {
            c = texture(uRegionArr, vec3(tc / 256.0, float(layer)), bias);
        } else {
            uint slot = vSlot & 0xffffu;
            uint smp = ((fl >> 3) & 1u) | (((fl >> 4) & 1u) << 1);
            vec2 size = vec2(float(vPA.z & 0xffffu), float(vPA.z >> 16));
            c = (slot != 0xffffu) ? texture(sampler2D(uRegion[nonuniformEXT(slot)], uRegionSmp[smp]), tc / size, bias) : vec4(0.47, 0.47, 1.0, 1.0);
            if (uTestStage == 9) { oColor = (slot == 0xffffu) ? vec4(1, 0, 1, 1) : vec4(0, 1, 1, 1); return; }   // diag: slot path
        }
        if (uTestStage == 10 && ((fl >> 8) & 1u) != 0u) { oColor = vec4(1, 1, 0, 1); return; }            // diag: microtextured
#ifdef LEAN_CUT
        // Cut-out: t premultiplied by opacity (G) over the filtered opacity (B), sharpened to a ~1 px
        // ramp around 0.5 so mips cannot thin sparse foliage away; alpha-to-coverage does the edge.
        float t = c.g * 15.0 / max(c.b, 1e-4);
        acov = clamp((c.b - 0.5) / max(fwidth(c.b), 1.0 / 255.0) + 0.5, 0.0, 1.0);
        if (acov <= 0.0) discard;
#else
        float t = c.r * 15.0;
#endif
#ifndef LEAN_CUT
        if (((fl >> 8) & 1u) != 0u && (vLayer >> 16) != 0xffffu) {
            vec2 dx = dFdx(tc), dy = dFdy(tc);
            float lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12)) + bias;
            if (lod < 0.0) {
                uint ulod = (fl >> 9) & 15u;
                float sc = float(1 << (1 << int(ulod)));
                float mt = texture(uRegionArr, vec3(tc * sc / 256.0, float(vLayer >> 16))).r * 15.0;
                t = mix(t, mt, min(-lod * 128.0 / float(1 << int(ulod)), 127.0) / 256.0);
            }
        }
#endif
        uint t8 = min(uint(t * 16.0 + 0.5), 0xf0u);
        uint luma = min((lumaram8((vPB.y & 0xffffu) + (t8 >> 1)) * (vPB.y >> 16)) / 256u, 0x3fu);
        rgb = shade_c(vColor & 0x7fffu, luma);
    }
    oColor = vec4(clamp((rgb - 0.5) * uContrast + 0.5 + uBright, 0.0, 1.0), 1.0);
#ifdef LEAN_CUT
    if (((fl >> 6) & 1u) != 0u) oColor.a = acov;
#endif
}
#else
void main_body() {
    Prim p = flatPrim();
#ifdef CUT_DEPTH
    // Depth pre-pass of the cut-outs: the cheapest possible shader -- the clip window of the
    // secondary views and the region's opacity (B channel, hardware filtered), nothing else. Colour
    // writes are masked; alpha-to-coverage turns the opacity into per-sample depth coverage.
    {
        bool mv = (uImmersive != 0 && vSecondary == 0u);
        ivec2 px = ivec2(floor(vBoard));
        if (!mv && (px.x < p.clip_l || px.x > p.clip_r || px.y < p.clip_t || px.y > p.clip_b)) discard;
        float zb = max(vParam.z, 1e-6);
        vec2 tcd = mv ? vec2(vParam.x, vParam.y) : vec2(0.0);
        vec2 ddx = dFdx(tcd), ddy = dFdy(tcd);
        oColor = vec4(0.0, 0.0, 0.0, 1.0);
        if (p.textured == 0u || p.translucent == 0u || !mv || (vSlot & 0xffffu) == 0xffffu) return;
        uint smp = ((vPA.w >> 3) & 1u) | (((vPA.w >> 4) & 1u) << 1);
        vec2 size = vec2(float(p.texwidth), float(p.texheight));
        float a = region_sample(vSlot & 0xffffu, smp + uint(uSmpBase), tcd, size, ddx, ddy, float(uMipBias) / 128.0).b;
        a = clamp((a - 0.5) / max(fwidth(a), 1.0 / 255.0) + 0.5, 0.0, 1.0);
        if (a <= 0.0) discard;
        oColor.a = a;
        return;
    }
#endif
    if (uTestStage == 3) {
#ifdef NO_DISCARD
        atomicAdd(dbg[0], 1u);
#else
        atomicAdd(dbg[1], 1u);
#endif
    }
#ifndef NO_DISCARD
    ivec2 pix = (uImmersive != 0) ? ivec2(floor(vBoard)) : ivec2(gl_FragCoord.xy) / uScale;
    if ((uImmersive == 0 || vSecondary != 0u) && (pix.x < p.clip_l || pix.x > p.clip_r || pix.y < p.clip_t || pix.y > p.clip_b)) discard;
    if (p.checker != 0u) {
        ivec2 fine = ivec2(gl_FragCoord.xy);
        if (uStipple == 0 && ((fine.x ^ fine.y) & 1) == 0) discard;
    }
#endif
    bool glass = (p.checker != 0u) && (uStipple != 0);
    float outAlpha = glass ? 0.5 : 1.0;

    // Texel coordinates and their screen derivatives, in uniform control flow.
    bool mainView = (uImmersive != 0 && vSecondary == 0u);
    float ooz = vParam.x, uoz = vParam.y, voz = vParam.z;
    if (mainView) { float zb = max(vParam.z, 1e-6); ooz = 1.0 / zb; uoz = vParam.x * ooz; voz = vParam.y * ooz; }
    ooz = max(ooz, 1e-9);
    float z = 1.0 / ooz;
    vec2 tc = vec2(uoz, voz) * z;          // level-0 texel units (u, v / 256)
    vec2 dx = dFdx(tc), dy = dFdy(tc);

    if (p.textured == 0u) {
#ifndef NO_DISCARD
        if (p.translucent != 0u) discard;
#endif
        oColor = vec4(shade_c(vColor, p.luma >> 2), outAlpha);
    } else {
#if defined(NO_DISCARD) && !defined(CUT_COLOR)
        bool translucent = false;   // routed here only when its texture has no transparent texel
#else
        bool translucent = (p.translucent != 0u);
#endif
        uint texmin = min(p.texwidth, p.texheight);
        int max_level = (texmin == 0u) ? 0 : max(findMSB(texmin) - 1, 0);
        int u = int(tc.x * 256.0), v = int(tc.y * 256.0);
        if (uFilterMode == 5 && mainView && uTestStage == 1) {
            oColor = vec4(0.5, 0.5, 0.5, outAlpha);
        } else if (uFilterMode == 5 && mainView && uCountOverdraw != 0 && (vSlot & 0xffffu) != 0xffffu) {
            // Every texture its own GPU image: repeat, mips and anisotropy done by the texture unit.
            uint smp = ((vPA.w >> 3) & 1u) | (((vPA.w >> 4) & 1u) << 1);
            vec2 size = vec2(float(p.texwidth), float(p.texheight));
            float bias = float(uMipBias) / 128.0;
            if (uTestStage == 5) { oColor = vec4(region_sample(vSlot & 0xffffu, smp, tc, size, dx, dy, bias).rrr, 1.0); return; }
            if (uTestStage == 6) { oColor = vec4(region_sample(vSlot & 0xffffu, smp, vec2(0.5), size, dx, dy, bias).rrr, 1.0); return; }
            if (uTestStage == 7) { oColor = vec4(vec3(fract(tc.x + tc.y)), 1.0); return; }
            vec4 c;
            uint layer = vLayer & 0xffffu;
            bool mirror = ((vPA.w >> 3) & 3u) != 0u;
            if (uTestStage == 11 && layer != 0xffffu) {   // diag: layer at mip 0 vs region image, side by side in stripes
                float a0 = textureLod(uRegionArr, vec3(tc / 256.0, float(layer)), 0.0).r;
                float b0 = ((vSlot & 0xffffu) != 0xffffu) ? textureLod(sampler2D(uRegion[nonuniformEXT(vSlot & 0xffffu)], uRegionSmp[0]), tc / size, 0.0).r : 0.0;
                oColor = vec4(a0, b0, float(layer) / 192.0, 1.0); return;
            }
            if ((uTexImplicit & 2) != 0 && layer != 0xffffu && !mirror) {
                // Tiled layer: layer repeat == region repeat, so tc/256 samples it directly.
                c = texture(uRegionArr, vec3(tc / 256.0, float(layer)), bias);
            } else {
                c = region_sample(vSlot & 0xffffu, smp + uint(uSmpBase), tc, size, dx, dy, bias);
            }
            vec2 ts = translucent ? vec2(c.g * 15.0, c.b) : vec2(c.r * 15.0, 1.0);
            uint ms = vSlot >> 16;
            if (!translucent && p.utex != 0u && ms != 0xffffu) {
                // Close up the board fades its 128x128 microtexture in, by how far below mip 0.
                float lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12)) + bias;
                if (lod < 0.0) {
                    float sc = float(1 << (1 << int(p.utexminlod)));
                    uint ml = vLayer >> 16;
                    float mt = ((uTexImplicit & 2) != 0 && ml != 0xffffu)
                                   ? texture(uRegionArr, vec3(tc * sc / 256.0, float(ml))).r * 15.0
                                   : region_sample(ms, 0u, tc * sc, vec2(128.0), dx * sc, dy * sc, 0.0).r * 15.0;
                    float w = min(-lod * 128.0 / float(1 << int(p.utexminlod)), 127.0) / 256.0;
                    ts.x = mix(ts.x, mt, w);
                }
            }
            float t = translucent ? ts.x / max(ts.y, 1e-4) : ts.x;
            if (uTestStage == 2) oColor = vec4(vec3(t / 15.0), outAlpha);
            else oColor = vec4(tone(p, t), outAlpha);
#ifndef NO_DISCARD
            if (translucent) {
                float a = clamp((ts.y - 0.5) / max(fwidth(ts.y), 1.0 / 255.0) + 0.5, 0.0, 1.0);
                if (a <= 0.0) discard;
                if (uAlphaCoverage == 0 && a < 0.5) discard;
                oColor.a = (uAlphaCoverage != 0) ? a : 1.0;
            }
#endif
        } else if (uFilterMode == 5 && mainView) {
            float lx = dot(dx, dx), ly = dot(dy, dy);
            vec2 maj = (lx >= ly) ? dx : dy;
            float majL = sqrt(max(lx, ly)), minL = sqrt(max(min(lx, ly), 1e-12));
            float n = clamp(ceil(majL / minL), 1.0, float(max(uAniso, 1)));
            float lod = log2(max(majL / n, minL)) + float(uMipBias) / 128.0;
            vec2 ts;
            if (lod < 0.0 && p.utex != 0u && !translucent) {
                // Close up: the board fades its microtexture in, weighted by how far below level 0.
                float tb = level_t(p, 0, tc);
                float w = min(-lod * 128.0 / float(1 << int(p.utexminlod)), 127.0) / 256.0;
                ts = vec2(mix(tb, micro_t(p, tc), w), 1.0);
            } else {
                float l = max(lod, 0.0);
                if (n <= 1.0) {
                    ts = tri_s(p, tc, l, max_level, translucent);
                } else {
                    ts = vec2(0.0);
                    for (int i = 0; i < 8; i++) {
                        if (float(i) >= n) break;
                        ts += tri_s(p, tc + maj * ((float(i) + 0.5) / n - 0.5), l, max_level, translucent);
                    }
                    ts /= n;
                }
            }
            float t = translucent ? ts.x / max(ts.y, 1e-4) : ts.x;
            // uTestStage (debug.tcvr.m2_stage), cost breakdown: 1 = flat colour, no texture;
            // 2 = texture only (t as grey, no colour chain); 0 = full.
            if (uTestStage == 1) oColor = vec4(0.5, 0.5, 0.5, outAlpha);
            else if (uTestStage == 2) oColor = vec4(vec3(t / 15.0), outAlpha);
            else oColor = vec4(tone(p, t), outAlpha);
#ifndef NO_DISCARD
            if (translucent) {
                // Filtered opacity, sharpened to a ~1 pixel ramp around 0.5 so mips cannot
                // thin a sparse cut-out away at distance; alpha-to-coverage spreads it on MSAA.
                float a = clamp((ts.y - 0.5) / max(fwidth(ts.y), 1.0 / 255.0) + 0.5, 0.0, 1.0);
                if (a <= 0.0) discard;
                if (uAlphaCoverage == 0 && a < 0.5) discard;
                oColor.a = (uAlphaCoverage != 0) ? a : 1.0;
            }
#endif
        } else {
            // Board path (cut-outs, secondary views, legacy modes): MAME's taps and alpha reject.
            int mml;
            if (mainView && uFilterMode == 5) {
                float lod = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-12)) + float(uMipBias) / 128.0;
                mml = int(lod * 128.0);
            } else {
                mml = -p.texlod + fast_log2(z);
                if (mainView) mml += uMipBias;
            }
            int level = clamp(mml >> 7, 0, max_level);
            int fm = (uFilterMode == 5) ? 2 : uFilterMode;
            vec4 base;
            if (fm == 3) base = fetch_bilinear_rgb(p, level, u, v, translucent);
            else if (fm == 4) {
                base = fetch_bilinear_rgb(p, level, u, v, translucent);
                if (mml > 0 && level < max_level) base = mix(base, fetch_bilinear_rgb(p, level + 1, u, v, translucent), float(mml & 127) / 128.0);
            } else {
                uint t;
                if (fm == 0) t = fetch_nearest_texel(p, level, u, v, translucent);
                else {
                    t = fetch_bilinear_texel(p, level, u, v, translucent);
                    if (fm == 2 && mml > 0 && level < max_level)
                        t = lerp_packed(t, fetch_bilinear_texel(p, level + 1, u, v, translucent), uint((mml & 127) << 1));
                    else if (fm == 2 && p.utex != 0u && mml < 0)
                        t = lerp_packed(t, fetch_bilinear_texel(p, -1, u, v, translucent), uint(min(-mml >> int(p.utexminlod), 127)));
                }
                float keep = 1.0;
                if (translucent) { if (t < 0x00400000u) keep = 0.0; t &= 0xffu; }
                uint luma = (lumaram8(p.lumabase + (t >> 1)) * p.luma) / 256u;
                base = vec4(shade(p, min(luma, 0x3fu), 0x7fffu), keep);
            }
            float acov = 1.0;
#ifndef NO_DISCARD
            if (translucent) {
                if (uAlphaCoverage != 0 && mainView) {
                    acov = clamp((base.a - 0.5) / max(fwidth(base.a), 1.0 / 255.0) + 0.5, 0.0, 1.0);
                    if (acov <= 0.0) discard;
                } else if (base.a < 0.5) {
                    discard;
                }
            }
#endif
            oColor = vec4(base.rgb, (uAlphaCoverage != 0 && mainView && translucent) ? acov : outAlpha);
        }
    }
    if (mainView && uEdgeFade > 0.5) {
        vec4 cl = vec4(uMainClip);
        float d = min(min(vBoard.x - cl.x, cl.z - vBoard.x), min(vBoard.y - cl.y, cl.w - vBoard.y));
        float fe = clamp(d / uEdgeFade, 0.0, 1.0);
        float hr = (uHorizonRow >= 0.0) ? uHorizonRow : 192.0;
        oColor.rgb = mix((vBoard.y > hr) ? uGround : uSky, oColor.rgb, fe);
    }
    if (uImmersive != 0) oColor.rgb = clamp((oColor.rgb - 0.5) * uContrast + 0.5 + uBright, 0.0, 1.0);
#ifdef NO_DISCARD
    oColor.a = 1.0;
#else
    if (!glass && !(uAlphaCoverage != 0 && mainView && p.translucent != 0u)) oColor.a = 1.0;
#endif
}
#endif  // LEAN

// Arcade colours are display values (CRT-referred). The eye image is an _SRGB attachment that encodes what we
// write: decode first, so the stored byte is the arcade's (true colours, as the GL build and MAME show) without
// the costly MUTABLE_FORMAT raw views (they disable framebuffer compression: -3 ms on Sega Rally, 23/09).
void main() {
    main_body();
    // Pop-in fade (24/09, like Wanszai's PC port of Virtua Racing): the scenery the game only sends within its
    // draw distance fades in from the sky's haze instead of popping up. uSky = the sky just above the horizon.
    if (uFogFar > 0.0 && uImmersive != 0 && vSecondary == 0u) {
        float f = smoothstep(0.7 * uFogFar, uFogFar, vParam.z);
        oColor.rgb = mix(oColor.rgb, uSky, f);
    }
    vec3 d = clamp(oColor.rgb, 0.0, 1.0);
    oColorOut = vec4(d * (d * (d * (d * -0.23012586 + 0.73328061) + 0.44219034) + 0.05115943), oColor.a);   // sRGB->linear, 4 FMA, <= 1.2/255 after re-encoding (fitted 23/09)
}
