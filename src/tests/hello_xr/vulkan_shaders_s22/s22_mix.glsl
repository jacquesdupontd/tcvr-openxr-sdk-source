// Continuous text-layer coordinate of this pixel (text pixels). Always written, so derivatives of q can be
// taken in uniform control flow; false when the pixel is off the arcade screen plane.
bool textCoord(out vec2 q) {
    q = vec2(-1.0);
    if (Flags.y == 0) { q = gl_FragCoord.xy / ScreenOut.zw; return true; }
    vec2 ndc = gl_FragCoord.xy / OutText.xy * 2.0 - 1.0;
    vec4 c0 = HudMvp[0], c1 = HudMvp[1], c3 = HudMvp[3];
    float a11 = c0.x - ndc.x * c0.w, a12 = c1.x - ndc.x * c1.w, b1 = ndc.x * c3.w - c3.x;
    float a21 = c0.y - ndc.y * c0.w, a22 = c1.y - ndc.y * c1.w, b2 = ndc.y * c3.w - c3.y;
    float det = a11 * a22 - a12 * a21;
    if (abs(det) < 1e-12) return false;
    float u = (b1 * a22 - a12 * b2) / det, v = (a11 * b2 - a21 * b1) / det;
    q = vec2((u + 0.5) * OutText.z, (0.5 - v) * OutText.w);
    if (abs(u) > 0.5 || abs(v) > 0.5) return false;
    if (u * c0.w + v * c1.w + c3.w <= 0.0) return false;
    return true;
}
// namcos22s_mix_text_layer, one pixel, for a given prival (verbatim port).
bool textTexel(out ivec2 p) {
    if (Flags.y == 0) { p = ivec2(gl_FragCoord.xy / ScreenOut.zw); return true; }
    vec2 ndc = gl_FragCoord.xy / OutText.xy * 2.0 - 1.0;
    vec4 c0 = HudMvp[0], c1 = HudMvp[1], c3 = HudMvp[3];
    float a11 = c0.x - ndc.x * c0.w, a12 = c1.x - ndc.x * c1.w, b1 = ndc.x * c3.w - c3.x;
    float a21 = c0.y - ndc.y * c0.w, a22 = c1.y - ndc.y * c1.w, b2 = ndc.y * c3.w - c3.y;
    float det = a11 * a22 - a12 * a21;
    if (abs(det) < 1e-12) return false;
    float u = (b1 * a22 - a12 * b2) / det, v = (a11 * b2 - a21 * b1) / det;
    if (abs(u) > 0.5 || abs(v) > 0.5) return false;
    if (u * c0.w + v * c1.w + c3.w <= 0.0) return false;
    p = clamp(ivec2((u + 0.5) * OutText.z, (0.5 - v) * OutText.w), ivec2(0), ivec2(OutText.zw) - 1);
    return true;
}
uvec3 mixText(ivec2 p, uvec3 dest, int prival) {
    bool spot = Mix0.x != 0 && Flags.z != 79;   // 79: diagnostic, spot remap off
    uint src = textAt(p);
    uint pen; uvec3 rgb;
    if (spot) {
        // spotram holds u16 words but MAME stores the result in a u8 (namcos22s_mix_text_layer: u8 pen):
        // the high byte is dropped. Without the mask every remapped pen was >= 0x80 and no text showed.
        pen = spotAt(((src << 2u) | uint(Mix0.z)) & 0x3ffu) & 0xffu;
        if (pen < 0x80u) rgb = penRGB(pen | uint(Mix0.w));
        else if (prival != 6) return dest;
        else rgb = dest;
    } else {
        rgb = penRGB(src); pen = src;
    }
    if (spot && pen >= 0x80u) {
        uint factor = (uint(Mix0.y) * (pen & 0x7fu)) >> 7u;
        rgb = min((dest * (0x100u - factor)) >> 8u, uvec3(255u));
    } else {
        if (Mix1.x != 0u) rgb = blend(rgb, FadeColor.xyz, Mix1.y);
        if (Mix1.z != 0u && ((pen & 0xfu) == Mix1.w || (pen >= Mix2.x && pen <= Mix2.y)))
            rgb = blend(rgb, dest, 255u - Mix1.z);
    }
    return rgb;
}

// Mode 3 for the text layer ("sharp bilinear", pixel-art filtering): every text pixel stays a solid block,
// the blend is confined to ONE output pixel at its borders - no stairs on the HUD, no blur. A neighbour
// contributes its mixed colour when ((its opaque-text bit) | extraPri) == want, as the board decides.
vec3 mixTextSharp(vec2 q, vec2 tpp, uvec3 dest, int prival, uint extraPri, uint want) {
    vec2 pp = q - 0.5;
    vec2 i0 = floor(pp);
    vec2 f = clamp((pp - i0 - 0.5) / max(tpp, vec2(1e-4)) + 0.5, 0.0, 1.0);
    ivec2 mx = ivec2(OutText.zw) - 1;
    vec3 acc = vec3(0.0);
    for (int k = 0; k < 4; ++k) {
        ivec2 o = ivec2(k & 1, k >> 1);
        ivec2 pk = clamp(ivec2(i0) + o, ivec2(0), mx);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        bool on = ((priAt(pk) & 4u) | extraPri) == want;
        acc += w * vec3(on ? mixText(pk, dest, prival) : dest);
    }
    return acc;
}
