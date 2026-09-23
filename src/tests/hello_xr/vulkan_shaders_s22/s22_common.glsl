// Shared declarations of the System 22 Vulkan module (port of hardware/namco_system22/stereo_renderer.cpp).
// Included by every s22 shader. Per-frame tables are storage buffers (two frames in flight), static
// ROM-derived assets are integer textures uploaded once.
layout(set = 0, binding = 0, std140) uniform S22 {
    mat4 ImmersiveMvp;
    mat4 HudMvp;          // arcade plane (u,v in -0.5..0.5) -> clip, this eye
    vec4 ScreenOut;       // ScreenSize.xy (board px), OutScale.xy
    vec4 OutText;         // OutSize.xy (target px), TextSize.xy
    vec4 DepthInfo;       // DepthMapScale, DepthMapSize.xy, SpriteMinDepth
    ivec4 Flags;          // Immersive, HudPlane, TexSamples, SpritesPerRow
    ivec4 Sprite;         // SpriteSize.xy, composite: current scene layer, dark-flicker gate on
    uvec4 Bg;             // background rgb
    ivec4 Mix0;           // SpotEnabled, SpotFactor, SpotPalbase, TextPalbase
    uvec4 Mix1;           // FadeEnabled, FadeFactor, AlphaFactor, AlphaMask
    uvec4 Mix2;           // AlphaCheck12, AlphaCheck13, HUD mode 3 (sharp text/sprites),
    uvec4 FadeColor;
    vec4 Bias;            // relative painter-order depth offset per primitive, EyeOffset, Convergence, near plane (m)
};
layout(std430, set = 0, binding = 1) readonly buffer PrimTable { vec4 prim[]; };   // 16 vec4 per primitive
layout(std430, set = 0, binding = 2) readonly buffer Pens { uint pens[]; };
layout(std430, set = 0, binding = 3) readonly buffer CzRam { uint czram[]; };       // bytes, 4 banks x 0x2000
layout(std430, set = 0, binding = 4) readonly buffer SpotRam { uint spot[]; };      // u16 x 0x400
layout(std430, set = 0, binding = 5) readonly buffer GammaT { uint gammaT[]; };     // bytes, 3 x 256
layout(std430, set = 0, binding = 6) readonly buffer TextL { uint textL[]; };       // u16, width x height
layout(std430, set = 0, binding = 7) readonly buffer PriL { uint priL[]; };         // bytes, width x height
layout(set = 0, binding = 8) uniform highp usampler2D TileAtlas;
layout(set = 0, binding = 9) uniform highp usampler2D TileMap;
layout(set = 0, binding = 10) uniform highp usampler2D TileAttr;
layout(set = 0, binding = 11) uniform highp usampler2D Ayx;
layout(set = 0, binding = 12) uniform highp usampler2D SpriteAtlas;
layout(set = 0, binding = 13) uniform highp sampler2D DepthMap;
layout(set = 0, binding = 14) uniform highp sampler2DArray SceneColor;   // composite: resolved scene, layers = current / previous arcade frame
layout(set = 0, binding = 15) uniform highp sampler2D ScenePri;     // composite: resolved priority
layout(set = 0, binding = 16) uniform highp usampler2DArray PenBanks;   // decoded texture address space, layer = bank

vec4 P(int p, int col) { return prim[p * 16 + col]; }
uint u8at(uint i, uint w) { return (w >> (8u * (i & 3u))) & 0xffu; }
uint czAt(uint i) { return u8at(i, czram[i >> 2]); }
uint gammaAt(uint i) { return u8at(i, gammaT[i >> 2]); }
uint spotAt(uint i) { uint w = spot[i >> 1]; return ((i & 1u) == 0u) ? (w & 0xffffu) : (w >> 16); }
uint textAt(ivec2 p) { uint i = uint(p.y) * uint(OutText.z) + uint(p.x); uint w = textL[i >> 1]; return ((i & 1u) == 0u) ? (w & 0xffffu) : (w >> 16); }
uint priAt(ivec2 p) { uint i = uint(p.y) * uint(OutText.z) + uint(p.x); return u8at(i, priL[i >> 2]); }

uvec3 penRGB(uint idx) { uint p = pens[idx]; return uvec3((p >> 16u) & 255u, (p >> 8u) & 255u, p & 255u); }
// rgbaint_t::blend(other, factor): (this*factor + other*(256-factor)) >> 8
uvec3 blend(uvec3 c, uvec3 o, uint f) { return (c * f + o * (256u - f)) >> 8u; }
