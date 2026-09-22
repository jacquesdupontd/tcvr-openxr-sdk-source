// Copyright (c) 2017-2026 The Khronos Group Inc.
// SPDX-License-Identifier: Apache-2.0
#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#pragma vertex

layout (std140, push_constant) uniform buf
{
    mat4 mvp;
} ubuf;

layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;

layout (location = 0) out vec2 PSTexCoord;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    PSTexCoord = TexCoord;
    gl_Position = ubuf.mvp * vec4(Position, 1.0);
}
