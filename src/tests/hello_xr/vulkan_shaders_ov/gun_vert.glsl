#version 450
// Time Crisis pistol (mesh from scripts/gen_gun_mesh.py). World position/normal for the lighting, object position
// for the grip checkering, material per vertex (roughness, metalness, flags + 8 * part).
// Animation (28/09): uEye.w packs three bytes -- slide recoil (0..255), trigger pull (0..255), muzzle flash (0..255).
// Parts: 0 frame (fixed), 1 slide (back along +z up to SLIDE_TRAVEL), 2 trigger (about its top, up to TRIGGER_ANGLE),
// 3 muzzle flash (scaled from the muzzle; 0 = collapsed, nothing drawn).
layout(push_constant) uniform PC { mat4 uMvp; vec4 uModel0; vec4 uModel1; vec4 uModel2; vec4 uEye; };
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in vec3 aMaterial;
layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vColor;
layout(location = 3) out vec3 vObj;
layout(location = 4) flat out vec3 vMaterial;
const float SLIDE_TRAVEL = 0.018;                    // m
const float TRIGGER_ANGLE = 0.30;                    // rad, full pull
const vec2 TRIGGER_PIVOT = vec2(-0.0265, -0.040);   // (y, z) of the trigger's top (gen_gun_mesh.py: FB + 0.001, -0.040)
const float MUZZLE_Z = -0.202;                       // Z_FRONT - 0.004
void main() {
    int f = int(aMaterial.z + 0.5);
    int part = f / 8;
    uint packedAnim = uint(uEye.w + 0.5);
    float slide = float(packedAnim & 255u) / 255.0;
    float pull = float((packedAnim >> 8) & 255u) / 255.0;
    float flash = float((packedAnim >> 16) & 255u) / 255.0;
    vec3 p = aPos, n = aNormal;
    if (part == 1) {
        p.z += slide * SLIDE_TRAVEL;
    } else if (part == 2) {
        float a = -pull * TRIGGER_ANGLE;   // negative: the finger end swings to the rear (+z)
        float c = cos(a), s = sin(a);
        vec2 q = vec2(p.y, p.z) - TRIGGER_PIVOT;
        vec2 r = vec2(q.x * c - q.y * s, q.x * s + q.y * c) + TRIGGER_PIVOT;
        p.y = r.x; p.z = r.y;
        n = vec3(n.x, n.y * c - n.z * s, n.y * s + n.z * c);
    } else if (part == 3) {
        p = vec3(p.xy * flash, MUZZLE_Z + (p.z - MUZZLE_Z) * flash);
    }
    mat4 model = transpose(mat4(uModel0, uModel1, uModel2, vec4(0.0, 0.0, 0.0, 1.0)));
    vWorld = (model * vec4(p, 1.0)).xyz;
    vNormal = mat3(model) * n;
    vColor = aColor;
    vObj = p;
    vMaterial = vec3(aMaterial.xy, float(f - part * 8));
    gl_Position = uMvp * vec4(p, 1.0);
}
