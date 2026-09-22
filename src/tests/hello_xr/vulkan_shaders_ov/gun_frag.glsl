#version 450
layout(push_constant) uniform PC { mat4 uMvp; vec4 uModel0; vec4 uModel1; vec4 uModel2; vec4 uEye; };
layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec3 vColor;
layout(location = 0) out vec4 oColor;
void main() {
    vec3 n = normalize(vNormal);
    vec3 key = normalize(vec3(0.35, 0.85, 0.40));
    vec3 fill = normalize(vec3(-0.3, -0.6, -0.5));
    vec3 v = normalize(uEye.xyz - vWorld);
    float diff = max(dot(n, key), 0.0) * 0.85 + max(dot(n, fill), 0.0) * 0.20;
    vec3 h = normalize(key + v);
    float spec = pow(max(dot(n, h), 0.0), 40.0) * 0.35;
    oColor = vec4(vColor * (0.22 + diff) + vec3(spec), 1.0);
}
