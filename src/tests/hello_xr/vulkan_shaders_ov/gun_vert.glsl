#version 450
// Time Crisis pistol (mesh from scripts/gen_gun_mesh.py). World position/normal for the lighting, object position
// for the grip checkering, material per vertex (roughness, metalness, flags).
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
void main() {
    mat4 model = transpose(mat4(uModel0, uModel1, uModel2, vec4(0.0, 0.0, 0.0, 1.0)));
    vWorld = (model * vec4(aPos, 1.0)).xyz;
    vNormal = mat3(model) * aNormal;
    vColor = aColor;
    vObj = aPos;
    vMaterial = aMaterial;
    gl_Position = uMvp * vec4(aPos, 1.0);
}
