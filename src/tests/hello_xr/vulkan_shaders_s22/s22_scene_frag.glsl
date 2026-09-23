#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 2 fragment stage: renderscanline_poly_ss22 / renderscanline_sprite (verbatim port of kSceneFs).
#include "s22_common.glsl"
layout(location = 0) in vec3 vTex;
layout(location = 1) in float vOriginalDepth;
layout(location = 2) in vec2 vScreen;
layout(location = 3) flat in int vPrim;
layout(location = 4) flat in uvec4 vP0;   // packed per-polygon constants (POLY3D reads these)
layout(location = 5) flat in uvec4 vP1;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oPri;
uint texPen(vec2 uv, int bn) {
    int tx = int(floor(uv.x)) & 0xfff;
    int ty = (int(floor(uv.y)) & 0xfff) | bn;
    int to = ((ty << 4) & 0xfff00) | (tx >> 4);
    uint tile = texelFetch(TileMap, ivec2(to & 255, to >> 8), 0).r;
    uint attr = texelFetch(TileAttr, ivec2(to & 255, to >> 8), 0).r;
    int ai = (int(attr) << 8) | ((ty << 4) & 0xf0) | (tx & 0xf);
    uint pix = texelFetch(Ayx, ivec2(ai & 255, ai >> 8), 0).r;
    return texelFetch(TileAtlas, ivec2(int(tile & 255u) * 16 + int(pix & 15u), int(tile >> 8u) * 16 + int(pix >> 4u)), 0).r;
}
// One level of the decoded bank, bilinear on the COLOURS of the 4 neighbouring indices (wraps like the
// 12-bit address). Level l = the index mipmap (majority of each 2x2 block).
vec3 bankBilinear(vec2 uv, int bank, int l, uint pensOff, uint pshift, uint pmask) {
    int size = 4096 >> l, m = size - 1;
    vec2 p = uv / float(1 << l) - 0.5;
    ivec2 i0 = ivec2(floor(p));
    vec2 f = p - vec2(i0);
    uint a = texelFetch(PenBanks, ivec3(i0 & m, bank), l).r;
    uint b = texelFetch(PenBanks, ivec3((i0 + ivec2(1, 0)) & m, bank), l).r;
    uint c = texelFetch(PenBanks, ivec3((i0 + ivec2(0, 1)) & m, bank), l).r;
    uint d = texelFetch(PenBanks, ivec3((i0 + ivec2(1, 1)) & m, bank), l).r;
    vec3 ca = vec3(penRGB(pensOff + ((a >> pshift) & pmask))), cb = vec3(penRGB(pensOff + ((b >> pshift) & pmask)));
    vec3 cc = vec3(penRGB(pensOff + ((c >> pshift) & pmask))), cd = vec3(penRGB(pensOff + ((d >> pshift) & pmask)));
    return mix(mix(ca, cb, f.x), mix(cc, cd, f.x), f.y);
}
void main() {
    int Immersive = Flags.x;
    // Texture footprint in texels per pixel (derivatives taken in uniform control flow).
    vec2 tdx = dFdx(vTex.xy), tdy = dFdy(vTex.xy);
    float lod = log2(max(max(length(tdx), length(tdy)), 1e-6));
#ifdef POLY3D
    // Per-polygon constants packed by the vertex stage (s22_scene_vert): no primitive-table reads per pixel.
    uint pensOff = vP0.x & 0xffffu, pmask = (vP0.x >> 16) & 0xffu, pshift = vP0.x >> 24;
    int bn = int(vP0.y & 15u) << 12;
    bool texEn = ((vP0.y >> 4) & 1u) != 0u, shadeEn = ((vP0.y >> 5) & 1u) != 0u;
    int fogMode = int((vP0.y >> 6) & 3u);
    float prio = float((vP0.y >> 8) & 0xffu);
    bool pfadeEn = ((vP0.y >> 16) & 1u) != 0u, fadeEn = ((vP0.y >> 17) & 1u) != 0u, alphaEn = ((vP0.y >> 18) & 1u) != 0u;
    int alpha = int((vP0.y >> 20) & 0xffu);
    uvec3 fogc = uvec3(vP0.z & 0xffu, (vP0.z >> 8) & 0xffu, (vP0.z >> 16) & 0xffu);
    int fogfactor = int(vP0.z >> 24);
    int czBank = int(vP0.w & 0xffu), czDelta = bitfieldExtract(int(vP0.w), 8, 16);
    uint alphaPen = vP0.w >> 24;
    uvec3 fadec = uvec3(vP1.x & 0xffu, (vP1.x >> 8) & 0xffu, (vP1.x >> 16) & 0xffu);
    int fadef = int(vP1.x >> 24);
    uvec3 polyc = uvec3(vP1.y & 511u, (vP1.y >> 9) & 511u, (vP1.y >> 18) & 511u);
#else
    vec4 t0 = P(vPrim, 0);
    vec4 t1 = P(vPrim, 1);
    vec2 sp = (Immersive != 0) ? vScreen : gl_FragCoord.xy / ScreenOut.zw;
    if ((Immersive == 0 || t0.x > 0.5 || t0.y > 0.5) &&
        (sp.x < t1.x || sp.x >= t1.z + 1.0 || sp.y < t1.y || sp.y >= t1.w + 1.0)) discard;
    vec4 t2 = P(vPrim, 2), t3 = P(vPrim, 3), t4 = P(vPrim, 4), t5 = P(vPrim, 5);
    vec4 t6 = P(vPrim, 6), t7 = P(vPrim, 7), t8 = P(vPrim, 8), t9 = P(vPrim, 9);
    uint pensOff = uint(t2.x + 0.5), pmask = uint(t2.z + 0.5), pshift = uint(t2.w + 0.5);
    int bn = int(t2.y + 0.5);
    bool texEn = t3.x > 0.5, shadeEn = t3.y > 0.5;
    int fogMode = int(t3.w + 0.5);
    float prio = t3.z;
    bool pfadeEn = t7.x > 0.5, fadeEn = t5.w > 0.5, alphaEn = t8.x > 0.5;
    int alpha = int(t8.y + 0.5);
    uvec3 fogc = uvec3(t5.xyz + 0.5);
    int fogfactor = int(t4.x + 0.5);
    int czBank = int(t4.z + 0.5), czDelta = int(floor(t4.y + 0.5));
    uint alphaPen = uint(t4.w + 0.5);
    uvec3 fadec = uvec3(t6.xyz + 0.5);
    int fadef = int(t6.w + 0.5);
    uvec3 polyc = uvec3(t7.yzw + 0.5);
#endif
    uvec3 c; uint pen; float srcWeight = 1.0;
#ifndef POLY3D
    if (t0.x < 0.5) {
#else
    {
#endif
        pen = 0u;
        if (texEn && Sprite.w != 0 && ((uint(Sprite.z) >> uint(bn >> 12)) & 1u) != 0u) {
            // Bilinear filter of the COLOURS: one gather of the 4 neighbouring palette indices from the
            // decoded bank, palette per neighbour, then the blend (an index cannot be interpolated).
            vec2 p = vTex.xy - 0.5;
            vec2 i0 = floor(p), f = p - i0;
            uvec4 g = textureGather(PenBanks, vec3((i0 + 1.0) / 4096.0, float(bn >> 12)));
            pen = f.y < 0.5 ? (f.x < 0.5 ? g.w : g.z) : (f.x < 0.5 ? g.x : g.y);   // nearest, for the alpha pen test
            // Mode 3: anisotropic footprint. The level follows the MINOR axis of the pixel's footprint (the
            // road seen at a grazing angle keeps its detail across), and up to 4 taps are spread along the
            // MAJOR axis to cover the rest (what the hardware anisotropic filter does on Sega Rally).
            float lodMin = log2(max(min(length(tdx), length(tdy)), 1e-6));
            float lodUse = Sprite.w >= 3 ? max(lodMin, lod - 2.0) : lod;
            if (Sprite.w >= 2 && lodUse > 0.0) {
                // Minified: trilinear between index-mipmap levels (colours blended, never indices).
                float lc = min(lodUse, 8.0);
                int l0 = int(floor(lc));
                int taps = 1;
                vec2 major = length(tdx) > length(tdy) ? tdx : tdy;
                if (Sprite.w >= 3) taps = int(clamp(exp2(lod - lodUse), 1.0, 4.0) + 0.5);
                vec3 acc = vec3(0.0);
                for (int k = 0; k < taps; ++k) {
                    vec2 uvk = vTex.xy + major * ((float(k) + 0.5) / float(taps) - 0.5);
                    vec3 c0 = bankBilinear(uvk, bn >> 12, l0, pensOff, pshift, pmask);
                    vec3 c1 = l0 < 8 ? bankBilinear(uvk, bn >> 12, l0 + 1, pensOff, pshift, pmask) : c0;
                    acc += mix(c0, c1, lc - float(l0));
                }
                c = uvec3(acc / float(taps) + 0.5);
            } else {
                // Magnified. Mode 3 = "sharp bilinear" (pixel-art filtering): each texel stays a solid block
                // like the board's, the blend is confined to ONE headset pixel at each texel border - crisp
                // road lines without stairs or shimmer. Modes 1-2 = plain bilinear (soft).
                if (Sprite.w >= 3) {
                    vec2 tpp = max(abs(tdx) + abs(tdy), vec2(1e-4));   // texels per pixel, per axis
                    f = clamp((f - 0.5) / tpp + 0.5, 0.0, 1.0);
                }
                vec3 c00 = vec3(penRGB(pensOff + ((g.w >> pshift) & pmask)));
                vec3 c10 = vec3(penRGB(pensOff + ((g.z >> pshift) & pmask)));
                vec3 c01 = vec3(penRGB(pensOff + ((g.x >> pshift) & pmask)));
                vec3 c11 = vec3(penRGB(pensOff + ((g.y >> pshift) & pmask)));
                c = uvec3(mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y) + 0.5);
            }
        } else if (texEn) {
            pen = texPen(vTex.xy, bn);
#ifndef NO_TEXAA
            int ts = Flags.z;
            if (ts > 1) {
                vec2 fx = tdx, fy = tdy;
                int n = 2;
                if (ts > 4) n = clamp(int(ceil(max(length(fx), length(fy)))), 2, 4);
                vec3 acc = vec3(0.0);
                for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
                    vec2 o = (vec2(float(i), float(j)) + 0.5) / float(n) - 0.5;
                    acc += vec3(penRGB(pensOff + ((texPen(vTex.xy + fx * o.x + fy * o.y, bn) >> pshift) & pmask)));
                }
                c = uvec3(acc / float(n * n) + 0.5);
            } else
#endif
            {
                c = penRGB(pensOff + ((pen >> pshift) & pmask));
            }
        } else {
            c = penRGB(pensOff + ((pen >> pshift) & pmask));
        }
        if (shadeEn) {
            uint shade = uint(max(floor(vTex.z), 0.0));
            c = min((c * (shade << 2u)) >> 8u, uvec3(255u));
        }
        if (fogMode == 2) {
            float depth = min(Immersive != 0 ? vOriginalDepth : 1.0 / gl_FragCoord.w, 8.0e6);
            int cz = min(int(depth) >> 8, 0x1fff);
            int f = int(czAt(uint(czBank * 0x2000 + cz))) + czDelta;
            if (f > 0) { f = min(f, 255); c = blend(c, fogc, uint(255 - f)); }
        } else if (fogMode == 1) {
            if (fogfactor != 0) c = blend(c, fogc, uint(255 - fogfactor));
        }
        if (pfadeEn) c = min((c * polyc) >> 8u, uvec3(255u));
        if (fadeEn && fadef != 0) c = blend(c, fadec, uint(255 - fadef));
        if (alpha != 0 && (alphaEn || pen == alphaPen)) srcWeight = float(255 - alpha) / 256.0;
    }
#ifndef POLY3D
    else {
        int code = int(t8.z + 0.5);
        int tx = clamp(int(vTex.x - t8.w), 0, Sprite.x - 1);
        int ty = clamp(int(vTex.y - t9.x), 0, Sprite.y - 1);
        int spr = Flags.w;
        pen = texelFetch(SpriteAtlas, ivec2((code % spr) * Sprite.x + tx, (code / spr) * Sprite.y + ty), 0).r;
        if (pen == 0xffu) discard;
        c = penRGB(pensOff + pen);
        if (fogMode == 1 && fogfactor != 0) c = blend(c, fogc, uint(255 - fogfactor));
        if (fadeEn && fadef != 0) c = blend(c, fadec, uint(255 - fadef));
        if (alpha != 0xff && (alphaEn || pen == alphaPen)) srcWeight = float(alpha) / 256.0;
    }
#endif
    oColor = vec4(vec3(c) / 255.0, srcWeight);
    oPri = vec4(prio / 255.0, 0.0, 0.0, 1.0);
}
