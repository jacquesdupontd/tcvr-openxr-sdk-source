// Copyright (c) 2017-2026 The Khronos Group Inc.
// SPDX-License-Identifier: Apache-2.0
#version 400
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

#pragma fragment

layout (binding = 0) uniform sampler2D ScreenTexture;

layout (location = 0) in vec2 PSTexCoord;

layout (location = 0) out vec4 FragColor;

void main()
{
    FragColor = vec4(texture(ScreenTexture, PSTexCoord).rgb, 1.0);
}
