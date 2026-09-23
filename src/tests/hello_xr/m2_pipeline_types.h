#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include "scene_bridge.h"

namespace arcadexr::vulkan {

// Set at device creation (graphicsplugin_vulkan.cpp), read by the Model 2 renderer.
inline bool g_m2DescriptorIndexing = false;
inline bool g_m2SamplerAnisotropy = false;

// Layout matching std140 UBO in m2_vert.glsl and m2_frag.glsl
struct M2UniformBufferObject {
    float uMvp[16];          // 64 bytes, offset 0
    float uHudMvp[16];       // 64 bytes, offset 64
    float uViewport[2];      // 8 bytes, offset 128
    float uFocus[2];         // 8 bytes, offset 136
    float uCrtc[2];          // 8 bytes, offset 144
    int32_t pad0[2];         // 8 bytes, offset 152
    int32_t uMainClip[4];    // 16 bytes, offset 160
    int32_t uMainCenter[2];  // 8 bytes, offset 176
    float uEyeOffset;        // 4 bytes, offset 184
    float uConvergence;      // 4 bytes, offset 188
    float uDepthBias;        // 4 bytes, offset 192
    float uPrimCount;        // 4 bytes, offset 196
    int32_t uImmersive;      // 4 bytes, offset 200
    int32_t uDepthOrder;     // 4 bytes, offset 204
    int32_t uRaw;            // 4 bytes, offset 208
    int32_t uScale;          // 4 bytes, offset 212
    int32_t uStipple;        // 4 bytes, offset 216
    int32_t uPass;           // 4 bytes, offset 220
    int32_t uPrepassClass;   // 4 bytes, offset 224
    float uEdgeFade;         // 4 bytes, offset 228
    float uHorizonRow;       // 4 bytes, offset 232
    float uOverlayK;         // 4 bytes, offset 236: screen overlays drawn this many times the view (1 = flat)
    float uSky[3];           // 12 bytes, offset 240
    int32_t pad2;            // 4 bytes, offset 252
    float uGround[3];        // 12 bytes, offset 256
    int32_t uAniso;          // 4 bytes, offset 268
    int32_t uFilterMode;     // 4 bytes, offset 272
    int32_t uMipBias;        // 4 bytes, offset 276
    int32_t uAlphaCoverage;  // 4 bytes, offset 280
    float uContrast;         // 4 bytes, offset 284
    float uBright;           // 4 bytes, offset 288
    int32_t uTestStage;      // 4 bytes, offset 292
    int32_t uCountOverdraw;  // 4 bytes, offset 296
    int32_t padEnd[3];       // 12 bytes, total 312 -> 320
};

struct VoidPushConstants {
    float uInvMvp[16];
    float uSky[4];
    float uGround[4];
    float uOutSize[2];
    float pad[2];
};

struct PlanePushConstants {
    float uHudMvp[16];
    float uOutSize[2];
    int32_t uKeyZero;
    float uUvScaleX;   // content width / texture width: the 2D image is 496 wide in a 512-wide texture
    float uClipRow;    // > 0: the back layer is not drawn below this board row (ground fill, gun games)
    float uNoTile;     // 1: the back layer is a menu page -- only inside the arcade frame, not repeated
};

}  // namespace arcadexr::vulkan
