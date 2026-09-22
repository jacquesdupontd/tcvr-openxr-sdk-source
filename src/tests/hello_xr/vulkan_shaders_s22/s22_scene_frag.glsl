#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 2 fragment stage: renderscanline_poly_ss22 / renderscanline_sprite (verbatim port of kSceneFs).
#include "s22_common.glsl"
layout(location = 0) in vec3 vTex;
layout(location = 1) in float vOriginalDepth;
layout(location = 2) in vec2 vScreen;
layout(location = 3) flat in int vPrim;
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
void main() {
    int Immersive = Flags.x;
    vec4 t0 = P(vPrim, 0);
    vec4 t1 = P(vPrim, 1);
#ifndef POLY3D
    vec2 sp = (Immersive != 0) ? vScreen : gl_FragCoord.xy / ScreenOut.zw;
    if ((Immersive == 0 || t0.x > 0.5 || t0.y > 0.5) &&
        (sp.x < t1.x || sp.x >= t1.z + 1.0 || sp.y < t1.y || sp.y >= t1.w + 1.0)) discard;
#endif
    vec4 t2 = P(vPrim, 2), t3 = P(vPrim, 3), t4 = P(vPrim, 4), t5 = P(vPrim, 5);
    vec4 t6 = P(vPrim, 6), t7 = P(vPrim, 7), t8 = P(vPrim, 8), t9 = P(vPrim, 9);
    uint pensOff = uint(t2.x + 0.5);
    uvec3 c; uint pen; float srcWeight = 1.0;
    uvec3 fogc = uvec3(t5.xyz + 0.5);
    int fogMode = int(t3.w + 0.5);
#ifndef POLY3D
    if (t0.x < 0.5) {
#else
    {
#endif
        pen = 0u;
        int bn = int(t2.y + 0.5);
        uint pshift = uint(t2.w + 0.5), pmask = uint(t2.z + 0.5);
        if (t3.x > 0.5) {
            pen = texPen(vTex.xy, bn);
#ifndef NO_TEXAA
            int ts = Flags.z;
            if (ts > 1) {
                vec2 fx = dFdx(vTex.xy), fy = dFdy(vTex.xy);
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
        if (t3.y > 0.5) {
            uint shade = uint(max(floor(vTex.z), 0.0));
            c = min((c * (shade << 2u)) >> 8u, uvec3(255u));
        }
        if (fogMode == 2) {
            float depth = min(Immersive != 0 ? vOriginalDepth : 1.0 / gl_FragCoord.w, 8.0e6);
            int cz = min(int(depth) >> 8, 0x1fff);
            int f = int(czAt(uint(int(t4.z + 0.5) * 0x2000 + cz))) + int(floor(t4.y + 0.5));
            if (f > 0) { f = min(f, 255); c = blend(c, fogc, uint(255 - f)); }
        } else if (fogMode == 1) {
            int f = int(t4.x + 0.5);
            if (f != 0) c = blend(c, fogc, uint(255 - f));
        }
        if (t7.x > 0.5) c = min((c * uvec3(t7.yzw + 0.5)) >> 8u, uvec3(255u));
        if (t5.w > 0.5) { int f = int(t6.w + 0.5); if (f != 0) c = blend(c, uvec3(t6.xyz + 0.5), uint(255 - f)); }
        int a = int(t8.y + 0.5);
        if (a != 0 && (t8.x > 0.5 || pen == uint(t4.w + 0.5))) srcWeight = float(255 - a) / 256.0;
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
        if (fogMode == 1) { int f = int(t4.x + 0.5); if (f != 0) c = blend(c, fogc, uint(255 - f)); }
        if (t5.w > 0.5) { int f = int(t6.w + 0.5); if (f != 0) c = blend(c, uvec3(t6.xyz + 0.5), uint(255 - f)); }
        int a = int(t8.y + 0.5);
        if (a != 0xff && (t8.x > 0.5 || pen == uint(t4.w + 0.5))) srcWeight = float(a) / 256.0;
    }
#endif
    oColor = vec4(vec3(c) / 255.0, srcWeight);
    oPri = vec4(t3.z / 255.0, 0.0, 0.0, 1.0);
}
