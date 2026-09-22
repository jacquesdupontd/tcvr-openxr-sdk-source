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
    uint src = textAt(p);
    uint pen; uvec3 rgb;
    if (Mix0.x != 0) {
        pen = spotAt(((src << 2u) | uint(Mix0.z)) & 0x3ffu);
        if (pen < 0x80u) rgb = penRGB(pen | uint(Mix0.w));
        else if (prival != 6) return dest;
        else rgb = dest;
    } else {
        rgb = penRGB(src); pen = src;
    }
    if (Mix0.x != 0 && pen >= 0x80u) {
        uint factor = (uint(Mix0.y) * (pen & 0x7fu)) >> 7u;
        rgb = min((dest * (0x100u - factor)) >> 8u, uvec3(255u));
    } else {
        if (Mix1.x != 0u) rgb = blend(rgb, FadeColor.xyz, Mix1.y);
        if (Mix1.z != 0u && ((pen & 0xfu) == Mix1.w || (pen >= Mix2.x && pen <= Mix2.y)))
            rgb = blend(rgb, dest, 255u - Mix1.z);
    }
    return rgb;
}
