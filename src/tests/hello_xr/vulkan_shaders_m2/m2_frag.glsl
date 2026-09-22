#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 vParam;
layout(location = 1) flat in uint vPrim;
layout(location = 2) flat in uint vSecondary;
layout(location = 3) in vec2 vBoard;
layout(location = 0) out vec4 oColor;

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

layout(std430, set = 0, binding = 1) readonly buffer Prims   { Prim prims[]; };
layout(std430, set = 0, binding = 2) readonly buffer Palram    { uint palram[]; };
layout(std430, set = 0, binding = 3) readonly buffer Colorxlat { uint colorxlat[]; };
layout(std430, set = 0, binding = 4) readonly buffer Lumaram   { uint lumaram[]; };
layout(std430, set = 0, binding = 5) readonly buffer Gamma     { uint gammaTab[]; };
layout(std430, set = 0, binding = 6) readonly buffer Tex0      { uint tex0[]; };
layout(std430, set = 0, binding = 7) readonly buffer Tex1      { uint tex1[]; };

layout(set = 0, binding = 8) uniform usampler2D uSheetTex0;
layout(set = 0, binding = 9) uniform usampler2D uSheetTex1;

uint u16at(uint arr_index_hi_lo, uint packed) {
    return (arr_index_hi_lo == 0u) ? (packed & 0xffffu) : (packed >> 16);
}
uint palram16(uint i)    { return u16at(i & 1u, palram[i >> 1]); }
uint colorxlat16(uint i) { return u16at(i & 1u, colorxlat[i >> 1]); }
uint lumaram8(uint i)    { return (lumaram[i >> 2] >> (8u * (i & 3u))) & 0xffu; }
uint gamma8(uint i)      { return (gammaTab[i >> 2] >> (8u * (i & 3u))) & 0xffu; }

uint sheet32(uint sheet, uint i) {
    ivec2 at = ivec2(int(i & 1023u), int(i >> 10));
    return (sheet == 0u) ? texelFetch(uSheetTex0, at, 0).r : texelFetch(uSheetTex1, at, 0).r;
}

uint get_texel(uint base_x, uint base_y, int x, int y, uint sheet) {
    int x2 = int(base_x) + x;
    int y2 = int(base_y) + y;
    if (x2 >= 1024) {
        x2 -= 1024;
        y2 ^= 1024;
    }
    uint offset = uint((y2 / 2) * 512 + (x2 / 2));
    uint texel = sheet32(sheet, offset >> 1);
    if ((offset & 1u) != 0u) texel >>= 16;
    if ((y & 1) == 0)        texel >>= 8;
    if ((x & 1) == 0)        texel >>= 4;
    return texel & 0x0fu;
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
    if (translucent) {
        if (tex00 != 0xf0u) tex00 |= 0x00800000u;
        if (tex01 != 0xf0u) tex01 |= 0x00800000u;
        if (tex10 != 0xf0u) tex10 |= 0x00800000u;
        if (tex11 != 0xf0u) tex11 |= 0x00800000u;
        if (tex00 == 0x000000f0u) tex00 = tex01 & 0xffu;
        if (tex01 == 0x000000f0u) tex01 = tex00 & 0xffu;
        if (tex10 == 0x000000f0u) tex10 = tex11 & 0xffu;
        if (tex11 == 0x000000f0u) tex11 = tex10 & 0xffu;
    }
    uint tex0x = lerp_packed(tex00, tex01, ufrac);
    uint tex1x = lerp_packed(tex10, tex11, ufrac);
    if (translucent) {
        if (tex0x == 0x000000f0u) tex0x = tex1x & 0xffu;
        if (tex1x == 0x000000f0u) tex1x = tex0x & 0xffu;
    }
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

void main() {
    Prim p = prims[vPrim];
    ivec2 pix = (uImmersive != 0) ? ivec2(floor(vBoard)) : ivec2(gl_FragCoord.xy) / uScale;
    if ((uImmersive == 0 || vSecondary != 0u) && (pix.x < p.clip_l || pix.x > p.clip_r || pix.y < p.clip_t || pix.y > p.clip_b)) discard;
    if (p.checker != 0u) {
        ivec2 fine = ivec2(gl_FragCoord.xy);
        if (uStipple == 0 && ((fine.x ^ fine.y) & 1) == 0) discard;
    }
    bool glass = (p.checker != 0u) && (uStipple != 0);
    if (uPass == 1 && glass) discard;
    if (uPass == 0) {
        if (glass || p.translucent != 0u) discard;
        bool sel = (uPrepassClass >= 3) || (p.textured == 0u) || (uPrepassClass >= 2 && p.utex != 0u);
        if (!sel) discard;
    }
    float outAlpha = glass ? 0.5 : 1.0;
    if (p.textured == 0u) {
        if (p.translucent != 0u) discard;
        if (uPass == 0) return;
        oColor = vec4(shade(p, p.luma >> 2, 0xffffu), outAlpha);
        return;
    }
    bool translucent = (p.translucent != 0u);
    float ooz = vParam.x;
    float uoz = vParam.y;
    float voz = vParam.z;
    if (uImmersive != 0 && vSecondary == 0u) { float zb = max(vParam.z, 1e-6); ooz = 1.0 / zb; uoz = vParam.x * ooz; voz = vParam.y * ooz; }
    if (ooz <= 0.0) discard;
    if (uPass == 0) return;

    float z = 1.0 / ooz;
    uint texmin = min(p.texwidth, p.texheight);
    int max_level = (texmin == 0u) ? 0 : max(findMSB(texmin) - 1, 0);
    int mml = -p.texlod + fast_log2(z);
    if (uImmersive != 0 && vSecondary == 0u) mml += uMipBias;
    int level = clamp(mml >> 7, 0, max_level);
    int u = int(uoz * z * 256.0);
    int v = int(voz * z * 256.0);

    vec4 base = shaded_sample(p, level, max_level, mml, u, v, translucent);
    bool useCov = (uAlphaCoverage != 0) && (uImmersive != 0) && (vSecondary == 0u);
    float acov = 1.0;
    if (translucent) {
        if (useCov) {
            acov = clamp((base.a - 0.5) / max(fwidth(base.a), 1.0/255.0) + 0.5, 0.0, 1.0);
            if (acov <= 0.0) discard;
        } else if (base.a < 0.5) {
            discard;
        }
    }
    vec3 rgb = base.rgb;
    if (uImmersive != 0 && vSecondary == 0u && uAniso > 1) {
        vec2 ax = vec2(dFdx(float(u)), dFdx(float(v)));
        vec2 ay = vec2(dFdy(float(u)), dFdy(float(v)));
        vec2 maj = (dot(ax, ax) >= dot(ay, ay)) ? ax : ay;
        float wsum = 1.0;
        for (int i = 0; i < 8; i++) {
            if (i >= uAniso - 1) break;
            float off = (float(i) + 1.0) / float(uAniso) - 0.5;
            int su = u + int(maj.x * off);
            int sv = v + int(maj.y * off);
            vec4 sm = shaded_sample(p, level, max_level, mml, su, sv, translucent);
            rgb += sm.rgb * sm.a;
            wsum += sm.a;
        }
        rgb /= wsum;
    }
    oColor = vec4(rgb, useCov ? acov : outAlpha);
    if (uImmersive != 0 && vSecondary == 0u && uEdgeFade > 0.5) {
        vec4 cl = vec4(uMainClip);
        float d = min(min(vBoard.x - cl.x, cl.z - vBoard.x), min(vBoard.y - cl.y, cl.w - vBoard.y));
        float f = clamp(d / uEdgeFade, 0.0, 1.0);
        float h = (uHorizonRow >= 0.0) ? uHorizonRow : 192.0;
        vec3 voidc = (vBoard.y > h) ? uGround : uSky;
        oColor.rgb = mix(voidc, oColor.rgb, f);
    }
    if (uImmersive != 0) oColor.rgb = clamp((oColor.rgb - 0.5) * uContrast + 0.5 + uBright, 0.0, 1.0);
}
