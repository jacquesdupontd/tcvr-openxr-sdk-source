#version 450
// Studio-lit pistol: GGX specular + Fresnel for the key and a rim light, an analytic "studio" environment
// (sky gradient + softbox) reflected by the metal, diffuse only on dielectrics. Output is LINEAR light: the eye
// image is an _SRGB attachment that encodes it (unlike the arcade colours, which are display values).
layout(push_constant) uniform PC { mat4 uMvp; vec4 uModel0; vec4 uModel1; vec4 uModel2; vec4 uEye; };
layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vColor;
layout(location = 3) in vec3 vObj;
layout(location = 4) flat in vec3 vMaterial;
layout(location = 0) out vec4 oColor;

const float PI = 3.14159265;

vec3 environment(vec3 d) {
    float up = d.y * 0.5 + 0.5;
    vec3 sky = mix(vec3(0.020, 0.021, 0.024), vec3(0.42, 0.44, 0.48), smoothstep(0.35, 1.0, up));
    float box = smoothstep(0.86, 0.97, dot(d, normalize(vec3(0.25, 0.85, 0.45))));          // overhead softbox
    float strip = smoothstep(0.93, 0.99, dot(d, normalize(vec3(-0.9, 0.25, 0.2))));         // side strip light
    return sky + vec3(2.6) * box + vec3(1.1, 1.15, 1.25) * strip;
}

float ggx(float nh, float a) { float a2 = a * a; float d = nh * nh * (a2 - 1.0) + 1.0; return a2 / (PI * d * d + 1e-6); }
float smith(float nv, float nl, float a) { float k = (a + 1.0) * (a + 1.0) / 8.0; return nv / (nv * (1.0 - k) + k) * nl / (nl * (1.0 - k) + k); }

void main() {
    float rough = clamp(vMaterial.x, 0.05, 1.0), metal = vMaterial.y;
    int flags = int(vMaterial.z + 0.5);
    vec3 base = vColor;
    if (flags == 2) { oColor = vec4(base * 2.2, 1.0); return; }        // tritium sight dots
    if (flags == 3) { oColor = vec4(base, 1.0); return; }              // inside the bore
    vec3 n = normalize(vNormal);
    if (flags == 1) {
        // Grip checkering: a diamond grid (1.4 mm pitch) as a bump on the object-space position.
        vec2 g = vec2(vObj.y + vObj.z, vObj.y - vObj.z) * (1.0 / 0.0014);
        vec2 f = fract(g) - 0.5;
        vec2 slope = sign(f) * step(abs(f.yx), abs(f)) * 0.55;
        vec3 bump = vec3(0.0, slope.x + slope.y, slope.x - slope.y);
        vec3 wb = vec3(dot(uModel0.xyz, bump), dot(uModel1.xyz, bump), dot(uModel2.xyz, bump));   // model rotation (rows)
        n = normalize(n + normalize(wb + 1e-6) * length(bump) * 0.35);
    }
    vec3 v = normalize(uEye.xyz - vWorld);
    float nv = max(dot(n, v), 1e-3);
    vec3 f0 = mix(vec3(0.04), base, metal);
    vec3 fres = f0 + (1.0 - f0) * pow(1.0 - nv, 5.0);
    float a = rough * rough;
    vec3 color = vec3(0.0);
    const vec3 L[2] = vec3[2](normalize(vec3(0.30, 0.90, 0.35)), normalize(vec3(-0.55, 0.25, -0.80)));
    const vec3 C[2] = vec3[2](vec3(3.2, 3.1, 3.0), vec3(1.4, 1.5, 1.7));
    for (int i = 0; i < 2; ++i) {
        float nl = max(dot(n, L[i]), 0.0);
        if (nl <= 0.0) continue;
        vec3 h = normalize(L[i] + v);
        vec3 fl = f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
        vec3 spec = fl * ggx(max(dot(n, h), 0.0), a) * smith(nv, nl, a) / (4.0 * nv * nl + 1e-4);
        vec3 diff = (1.0 - fl) * (1.0 - metal) * base / PI;
        color += (diff + spec) * C[i] * nl;
    }
    vec3 r = reflect(-v, n);
    color += environment(r) * fres * (1.0 - 0.75 * rough);                 // reflections (sharp on the blued steel)
    color += base * (1.0 - metal) * environment(n) * 0.35;                  // diffuse ambient on the polymer
    color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);   // filmic tone curve
    oColor = vec4(color, 1.0);
}
