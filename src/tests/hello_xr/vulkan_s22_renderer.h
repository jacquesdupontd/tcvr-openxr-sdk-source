#pragma once

// Namco System 22 (Time Crisis, Dirt Dash) immersive renderer, Vulkan (opus55, 23/09/2026).
//
// Port of hardware/namco_system22/stereo_renderer.cpp (GLES), same per-pixel arithmetic (shaders in
// vulkan_shaders_s22/, transcribed line by line), same three passes, following the "one board = one
// module" method of the Sega Model 2 GOLD:
//   D. depth map (once per frame): 3D polygons in the board's own projection, camera depth per texel,
//      half resolution; read back one frame late for the lightgun ray cast and used by world sprites;
//   S. scene pass (per eye, MSAA optional): background + text at priority 4, then polygons (depth
//      tested) and screen-plane sprites (painter order), colour + priority attachments, resolved;
//   C. composite (per eye) straight into the eye image: text over polygons at priority 6, gamma.
// Per-frame tables live in storage buffers duplicated per frame slot (two frames in flight); the
// ROM-derived tile store and sprites are integer textures uploaded once.

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vulkan_utils.h"
#include "logger.h"
#include "common.h"
#include "../../../../../third_party/mame/src/mame/tcvr_scene.h"
#include "s22_quad_vert_spv.h"
#include "s22_init_frag_spv.h"
#include "s22_scene_vert_spv.h"
#include "s22_scene_frag_spv.h"
#include "s22_scene_frag_p3_spv.h"
#include "s22_scene_frag_p3n_spv.h"
#include "s22_depth_frag_spv.h"
#include "s22_comp_frag_spv.h"

namespace arcadexr::vulkan {

class VulkanSystem22Renderer {
public:
    static constexpr int kFrames = 2;

    struct Ubo {   // std140, matches s22_common.glsl
        float ImmersiveMvp[16];
        float HudMvp[16];
        float ScreenOut[4];
        float OutText[4];
        float DepthInfo[4];
        int32_t Flags[4];
        int32_t Sprite[4];
        uint32_t Bg[4];
        int32_t Mix0[4];
        uint32_t Mix1[4];
        uint32_t Mix2[4];
        uint32_t FadeColor[4];
        float Bias[4];
    };

    struct Settings {
        bool depthTest = true;
        int lean = 1;
        int skip = 0;   // bench: 1 init, 2 polygons, 4 composite
        // Reversed infinite depth: z_ndc = near * (1 + depthBias * primitive) / distance. Float precision is then
        // relative everywhere, so the painter-order offset is relative too and the same for every game
        // (2.5e-7 = ~4 ulp between consecutive primitives). The old per-game values were for the standard
        // mapping, whose precision collapsed with distance (Dirt Dash's far shadows flashed on the road).
        float depthBias = 2.5e-7f;
        float nearClip = 0.005f;
        int voidMode = 0;              // 0 game bg, 1 fog colour (slate if black), 2 fixed
        unsigned voidRGB[3] = {28, 30, 38};
        int texSamples = 4;
        float spriteMinDepth = 50.0f;
        bool darkFlicker = false;   // profile temporal.darkFlicker (Dirt Dash): CRT-like integration of dark 30 Hz flashes
    };

    bool Ready() const { return m_ready; }
    bool AssetsReady() const { return m_assetsReady; }
    unsigned LastPrims() const { return m_lastPrims; }
    float LastZoom() const { return m_lastZoom; }

    void Init(VkDevice dev, const MemoryAllocator* alloc, VkFormat eyeFormat, int msaa) {
        if (m_ready) return;
        m_dev = dev; m_alloc = alloc; m_eyeFormat = eyeFormat;
        m_samples = msaa >= 4 ? VK_SAMPLE_COUNT_4_BIT : (msaa >= 2 ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT);
        CreateLayouts();
        CreateRenderPasses();
        CreatePipelines();
        CreateSamplerAndDummies();
        for (int s = 0; s < kFrames; ++s) CreateSlot(s);
        m_ready = true;
        Log::Write(Log::Level::Info, Fmt("TCVR_S22VK ready: MSAA x%d", int(m_samples)));
    }

    ~VulkanSystem22Renderer() { StopBankWorker(); }
    void Destroy() {
        if (!m_dev) return;
        vkDeviceWaitIdle(m_dev);
        ResetAssets();
        for (auto& f : m_compFbs) { vkDestroyFramebuffer(m_dev, f.fb, nullptr); vkDestroyImageView(m_dev, f.view, nullptr); }
        m_compFbs.clear();
        for (int e = 0; e < 3; ++e) DestroyEyeTarget(m_eye[e]);
        DestroyFlatOut();
        DestroyDepthMap();
        for (int s = 0; s < kFrames; ++s) DestroySlot(s);
        auto dp = [&](VkPipeline& p) { if (p) vkDestroyPipeline(m_dev, p, nullptr); p = VK_NULL_HANDLE; };
        dp(m_pInit); dp(m_pScene3D); dp(m_pP3); dp(m_pP3NoAa); dp(m_pSceneHud); dp(m_pDepth); dp(m_pComp);
        auto drp = [&](VkRenderPass& r) { if (r) vkDestroyRenderPass(m_dev, r, nullptr); r = VK_NULL_HANDLE; };
        drp(m_rpDepth); drp(m_rpScene); drp(m_rpComp);
        if (m_layout) vkDestroyPipelineLayout(m_dev, m_layout, nullptr);
        if (m_setLayout) vkDestroyDescriptorSetLayout(m_dev, m_setLayout, nullptr);
        if (m_pool) vkDestroyDescriptorPool(m_dev, m_pool, nullptr);
        if (m_nearest) vkDestroySampler(m_dev, m_nearest, nullptr);
        if (m_linear) vkDestroySampler(m_dev, m_linear, nullptr);
        if (m_repeatNearest) vkDestroySampler(m_dev, m_repeatNearest, nullptr);
        DestroyImage(m_dummyU); DestroyImage(m_dummyF); DestroyImage(m_dummyArr); DestroyImage(m_dummyU16);
        m_dev = VK_NULL_HANDLE; m_ready = false;
    }

    // ---- static assets (tile store, sprites): once per game --------------------------------------
    void ResetAssets() {
        if (!m_dev) return;
        StopBankWorker();
        if (m_assetsReady) vkDeviceWaitIdle(m_dev);
        DestroyImage(m_penBanks);
        m_bankStaging.clear();
        m_bankReadyMask = 0;
        DestroyImage(m_tileAtlas); DestroyImage(m_tileMap); DestroyImage(m_tileAttr); DestroyImage(m_ayx); DestroyImage(m_spriteAtlas);
        m_assetsReady = false;
        m_setsDirty = true;
    }

    bool UploadAssets(VkCommandBuffer cmd, const tcvr_scene_assets& a) {
        if (!a.tiledata || !a.tilemap || !a.tileattr || !a.ayx || a.tiledata_bytes < 256) return false;
        std::vector<uint8_t> atlas(size_t(4096) * 4096, 0);
        const uint32_t tiles = std::min<uint32_t>(a.tiledata_bytes / 256, 65536);
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint8_t* src = a.tiledata + size_t(t) * 256;
            const size_t bx = size_t(t & 255) * 16, by = size_t(t >> 8) * 16;
            for (int row = 0; row < 16; ++row) std::memcpy(&atlas[(by + row) * 4096 + bx], src + row * 16, 16);
        }
        m_tileAtlas = MakeImage(cmd, VK_FORMAT_R8_UINT, 4096, 4096, atlas.data(), 1);
        m_tileMap = MakeImage(cmd, VK_FORMAT_R16_UINT, 256, 4096, a.tilemap, 2);
        m_tileAttr = MakeImage(cmd, VK_FORMAT_R8_UINT, 256, 4096, a.tileattr, 1);
        m_ayx = MakeImage(cmd, VK_FORMAT_R8_UINT, 256, 16, a.ayx, 1);
        if (a.sprites && a.sprite_count > 0 && a.sprite_width > 0 && a.sprite_height > 0) {
            m_spriteW = int(a.sprite_width); m_spriteH = int(a.sprite_height);
            m_spritesPerRow = std::max(1, 2048 / m_spriteW);
            const int rows = int((a.sprite_count + m_spritesPerRow - 1) / m_spritesPerRow);
            const int w = m_spritesPerRow * m_spriteW, h = rows * m_spriteH;
            std::vector<uint8_t> sp(size_t(w) * h, 0xff);
            for (uint32_t e = 0; e < a.sprite_count; ++e) {
                const size_t bx = size_t(e % m_spritesPerRow) * m_spriteW, by = size_t(e / m_spritesPerRow) * m_spriteH;
                for (int y = 0; y < m_spriteH; ++y)
                    std::memcpy(&sp[(by + y) * w + bx], a.sprites + (size_t(e) * m_spriteH + y) * m_spriteW, m_spriteW);
            }
            m_spriteAtlas = MakeImage(cmd, VK_FORMAT_R8_UINT, uint32_t(w), uint32_t(h), sp.data(), 1);
        } else {
            m_spriteW = m_spriteH = 1; m_spritesPerRow = 1;
            const uint8_t none = 0xff;
            m_spriteAtlas = MakeImage(cmd, VK_FORMAT_R8_UINT, 1, 1, &none, 1);
        }
        // Decoded pen banks for filtered sampling (see StartBankWorker). Shader-readable from the start;
        // a bank is used only once its bit is in the ready mask, the ROM table chain covers the rest.
        m_penBanks = NewImage(VK_FORMAT_R8_UINT, 4096, 4096, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_SAMPLE_COUNT_1_BIT, false, VK_IMAGE_ASPECT_COLOR_BIT, 16, kBankMips);
        Barrier(cmd, m_penBanks.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        StartBankWorker(a, tiles);
        m_assetsReady = true;
        m_setsDirty = true;
        Log::Write(Log::Level::Info, Fmt("TCVR_S22VK assets: tiles %u bytes, sprites %u (%ux%u)", a.tiledata_bytes, a.sprite_count,
                                         a.sprite_width, a.sprite_height));
        return true;
    }

    // ---- decoded pen banks -----------------------------------------------------------------------
    // The texture address is 12-bit u, 12-bit v and a 4-bit bank (texturebank * 0x1000 OR-ed into v),
    // resolved by the board through three ROM tables (tile map, tile attribute, ayx swizzle). Decoding a
    // bank gives a plain 4096x4096 image of palette indices: one textureGather then returns the 4
    // neighbours a bilinear filter needs (the chain would cost 4 dependent fetches per neighbour).
    // 16 banks = 256 MB, decoded on a worker at game load and uploaded one per frame.
    void StartBankWorker(const tcvr_scene_assets& a, uint32_t tiles) {
        StopBankWorker();
        m_bankCancel = false;
        m_bankReadyMask = 0;
        const tcvr_scene_assets A = a;
        m_bankWorker = std::thread([this, A, tiles] {
            for (uint32_t bank = 0; bank < 16 && !m_bankCancel; ++bank) {
                std::vector<uint8_t> img(size_t(4096) * 4096, 0);
                for (uint32_t y = 0; y < 4096; ++y) {
                    const uint32_t ty = (bank << 12) | y;
                    uint8_t* row = &img[size_t(y) * 4096];
                    for (uint32_t tx = 0; tx < 4096; ++tx) {
                        const uint32_t to = ((ty << 4) & 0xfff00u) | (tx >> 4);
                        if (to >= A.tilemap_entries || to >= A.tileattr_entries) continue;
                        const uint32_t tile = A.tilemap[to];
                        const uint32_t ai = (uint32_t(A.tileattr[to]) << 8) | ((ty << 4) & 0xf0u) | (tx & 0xfu);
                        if (ai >= A.ayx_entries || tile >= tiles) continue;
                        const uint32_t pix = A.ayx[ai];
                        row[tx] = A.tiledata[size_t(tile) * 256 + (pix >> 4) * 16 + (pix & 15u)];
                    }
                }
                // Index mipmaps: a palette index cannot be averaged (each polygon picks its own palette), so
                // each level keeps, per 2x2 block, the index that occurs most (ties: first in scan order). A
                // stable choice: the distance stops shimmering; the shader then blends COLOURS between levels.
                size_t off = img.size();
                img.resize(off + BankMipBytes() - size_t(4096) * 4096);
                size_t prev = 0;
                for (uint32_t l = 1; l < kBankMips; ++l) {
                    const uint32_t ps = 4096u >> (l - 1), ns = ps >> 1;
                    const uint8_t* src = img.data() + prev;
                    uint8_t* dst = img.data() + off;
                    for (uint32_t y = 0; y < ns; ++y)
                        for (uint32_t x = 0; x < ns; ++x) {
                            const uint8_t a = src[(2 * y) * ps + 2 * x], b = src[(2 * y) * ps + 2 * x + 1];
                            const uint8_t c = src[(2 * y + 1) * ps + 2 * x], d = src[(2 * y + 1) * ps + 2 * x + 1];
                            uint8_t m = a;
                            if (b == c || b == d) m = b;
                            if (a == b || a == c || a == d) m = a;
                            else if (c == d) m = c;
                            dst[y * ns + x] = m;
                        }
                    prev = off; off += size_t(ns) * ns;
                }
                std::lock_guard<std::mutex> lock(m_bankMutex);
                m_bankQueue.emplace_back(int(bank), std::move(img));
            }
        });
    }
    void StopBankWorker() {
        m_bankCancel = true;
        if (m_bankWorker.joinable()) m_bankWorker.join();
        std::lock_guard<std::mutex> lock(m_bankMutex);
        m_bankQueue.clear();
    }
public:
    // Once per frame at view 0: upload at most one decoded bank, free stagings no frame can still read.
    void PumpBanks(VkCommandBuffer cmd) {
        ++m_frameCounter;
        while (!m_bankStaging.empty() && m_bankStaging.front().second + 4 <= m_frameCounter) {
            m_bankStaging.front().first->Reset(m_dev);
            m_bankStaging.erase(m_bankStaging.begin());
        }
        if (!m_assetsReady || !m_penBanks.image) return;
        std::pair<int, std::vector<uint8_t>> job;
        {
            std::lock_guard<std::mutex> lock(m_bankMutex);
            if (m_bankQueue.empty()) return;
            job = std::move(m_bankQueue.front());
            m_bankQueue.erase(m_bankQueue.begin());
        }
        auto sb = std::make_unique<BufferAndMemory>();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = job.second.size(); bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sb->Create(m_dev, *m_alloc, bi);
        void* p = nullptr;
        XRC_CHECK_THROW_VKCMD(vkMapMemory(m_dev, sb->mem, 0, bi.size, 0, &p));
        std::memcpy(p, job.second.data(), job.second.size());
        vkUnmapMemory(m_dev, sb->mem);
        Barrier(cmd, m_penBanks.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy regions[kBankMips]{};
        VkDeviceSize off = 0;
        for (uint32_t l = 0; l < kBankMips; ++l) {
            const uint32_t sz = 4096u >> l;
            regions[l].bufferOffset = off;
            regions[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, uint32_t(job.first), 1};
            regions[l].imageExtent = {sz, sz, 1};
            off += VkDeviceSize(sz) * sz;
        }
        vkCmdCopyBufferToImage(cmd, sb->buf, m_penBanks.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kBankMips, regions);
        Barrier(cmd, m_penBanks.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        m_bankStaging.emplace_back(std::move(sb), m_frameCounter);
        m_bankReadyMask |= 1u << job.first;
        if (m_bankReadyMask == 0xffffu) Log::Write(Log::Level::Info, "TCVR_S22VK pen banks: 16/16 decoded and uploaded");
    }
    void SetFilter(int mode) { m_filter = mode; }
    static constexpr uint32_t kBankMips = 9;   // 4096 .. 16
    static size_t BankMipBytes() { size_t n = 0; for (uint32_t l = 0; l < kBankMips; ++l) n += size_t(4096u >> l) * (4096u >> l); return n; }
    void SetDiagAlternating(bool on) { m_diagAlt = on; }
    void SetAltFix(bool on) { m_altFix = on; }
    // A 3D primitive group (same render state) drawn one arcade frame out of two - Dirt Dash's car
    // shadows, measured 23/09 with s22.diagAlt: 8 quads present, then absent, every other frame. A CRT's
    // phosphor showed that as a steady half-strength shadow; an XR display shows a 30 Hz flash. Drawing
    // the group every frame at 50 % is exactly the time average of the on and off frames, in the scene
    // (no image-space history, so no ghosting when the head moves): present frames draw it at 50 %,
    // absent frames redraw the previous frame's copies at 50 %. Detection is generic (presence history
    // per group), nothing is listed per game.
    static bool AltKeyEligible(const tcvr_scene_prim& p) { return p.kind == 0 && p.direct == 0; }
    static uint64_t AltKey(const tcvr_scene_prim& p) {
        const uint32_t v[] = {p.pens_offset, p.bn, p.penmask, p.penshift, p.texture_enabled, p.shade_enabled, p.fog_mode,
                              p.pfade_enabled, uint32_t(int(p.poly_r)), uint32_t(int(p.poly_g)), uint32_t(int(p.poly_b)),
                              p.alpha_enabled, uint32_t(p.alpha), p.alpha_pen, p.fade_enabled, p.prioverchar};
        uint64_t h = 1469598103934665603ull;
        for (uint32_t x : v) { h ^= x; h *= 1099511628211ull; }
        return h;
    }
    // Per-INSTANCE tracking (a group-level test missed cars whose shadows flash in opposite phases: the
    // group looked present every frame). Only groups with few instances are tracked (<= 16: not the road's
    // hundreds of quads); each polygon is matched to the previous frame's by screen position and depth.
    struct AltTrack {
        uint64_t key = 0; float sx = 0, sy = 0, z = 0; uint8_t hist = 0; bool matched = false;
        tcvr_scene_prim prim{}; std::vector<tcvr_scene_vertex> verts;
    };
    static bool Centroid(const tcvr_scene_prim& p, const tcvr_scene_vertex* v, float& sx, float& sy, float& z) {
        sx = sy = z = 0.0f;
        for (uint32_t i = 0; i < p.vertex_count; ++i) {
            if (v[i].z <= 1e-3f) return false;
            sx += v[i].x / v[i].z; sy += v[i].y / v[i].z; z += v[i].z;
        }
        const float n = float(p.vertex_count);
        sx = float(p.cx) + sx / n; sy = float(p.cy) - sy / n; z /= n;
        return true;
    }
    void UpdateAlternation(const tcvr_scene_frame& f) {
        m_altSeq = f.sequence;
        m_altHalfPrim.assign(f.prim_count, 0);
        m_altCarry.clear(); m_altCarryVerts.clear(); m_altCarryFirst.clear();
        std::unordered_map<uint64_t, int> count;
        for (uint32_t i = 0; i < f.prim_count; ++i) if (AltKeyEligible(f.prims[i])) ++count[AltKey(f.prims[i])];
        for (AltTrack& t : m_tracks) { t.matched = false; t.hist = uint8_t(t.hist << 1); }
        for (uint32_t i = 0; i < f.prim_count; ++i) {
            const tcvr_scene_prim& p = f.prims[i];
            if (!AltKeyEligible(p) || p.vertex_count < 3 || p.first_vertex + p.vertex_count > f.vertex_count) continue;
            const uint64_t key = AltKey(p);
            if (count[key] > 16) continue;
            const tcvr_scene_vertex* v = f.vertices + p.first_vertex;
            float sx, sy, z;
            if (!Centroid(p, v, sx, sy, z)) continue;
            AltTrack* best = nullptr; float bestD = 1e30f;
            for (AltTrack& t : m_tracks) {
                if (t.matched || t.key != key || std::fabs(t.z - z) > 0.25f * z) continue;
                const float d = std::fabs(t.sx - sx) + std::fabs(t.sy - sy);
                if (d < 40.0f && d < bestD) { bestD = d; best = &t; }
            }
            if (!best) { m_tracks.emplace_back(); best = &m_tracks.back(); best->key = key; }
            best->matched = true; best->hist |= 1u; best->sx = sx; best->sy = sy; best->z = z;
            best->prim = p; best->verts.assign(v, v + p.vertex_count);
            if ((best->hist & 0x0f) == 0x5) m_altHalfPrim[i] = 1;   // present now, absent last frame, alternating
        }
        for (AltTrack& t : m_tracks) {
            if (t.matched || (t.hist & 0x0f) != 0xa) continue;          // absent now, present last frame, alternating
            m_altCarryFirst.push_back(m_altCarryVerts.size());
            m_altCarryVerts.insert(m_altCarryVerts.end(), t.verts.begin(), t.verts.end());
            m_altCarry.push_back(t.prim);
        }
        m_tracks.erase(std::remove_if(m_tracks.begin(), m_tracks.end(), [](const AltTrack& t) { return (t.hist & 0x0f) == 0; }), m_tracks.end());
        if (m_tracks.size() > 4096) m_tracks.clear();   // safety: never grow without bound
        if (m_diagAlt) {
            static auto last = std::chrono::steady_clock::now();
            static int frames = 0, halves = 0, carries = 0;
            ++frames; carries += int(m_altCarry.size());
            for (uint8_t h : m_altHalfPrim) halves += h;
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
                Log::Write(Log::Level::Info, Fmt("TCVR_S22ALTFIX frames=%d tracks=%zu half/frame=%.2f carried/frame=%.2f",
                                                 frames, m_tracks.size(), halves / float(frames), carries / float(frames)));
                last = std::chrono::steady_clock::now(); frames = halves = carries = 0;
            }
        }
    }
    bool m_altFix = true;
    uint64_t m_altSeq = ~0ull;
    std::vector<AltTrack> m_tracks;
    std::vector<uint8_t> m_altHalfPrim;
    std::vector<tcvr_scene_prim> m_altCarry;
    std::vector<tcvr_scene_vertex> m_altCarryVerts;
    std::vector<size_t> m_altCarryFirst;
    // Diagnostic (s22.diagAlt=1): which primitives appear in one arcade frame and not the next? Groups the
    // primitives by their full render state and logs, every 2 s, the groups whose count differs between
    // the last two frames. Raw matter for "what is the flashing shadow" - no judgement, all fields printed.
    void DiagAlternating(const tcvr_scene_frame& f) {
        if (f.sequence == m_diagSeq) return;
        m_diagSeq = f.sequence;
        std::unordered_map<std::string, int> cur;
        for (uint32_t i = 0; i < f.prim_count; ++i) {
            const tcvr_scene_prim& p = f.prims[i];
            char k[256];
            std::snprintf(k, sizeof(k), "kind%u dir%u nv%u tex%u shade%u pens%u bn%u mask%u fog%u pfade%u rgb%d,%d,%d aen%u a%d apen%u fade%u prio%u",
                          p.kind, p.direct, p.vertex_count, p.texture_enabled, p.shade_enabled, p.pens_offset, p.bn >> 12, p.penmask,
                          p.fog_mode, p.pfade_enabled, int(p.poly_r), int(p.poly_g), int(p.poly_b), p.alpha_enabled, p.alpha,
                          p.alpha_pen, p.fade_enabled, p.prioverchar);
            ++cur[k];
        }
        const auto now = std::chrono::steady_clock::now();
        if (!m_diagPrev.empty() && now - m_diagLast > std::chrono::seconds(2)) {
            m_diagLast = now;
            std::vector<std::pair<int, std::string>> diffs;
            for (const auto& kv : cur) { auto it = m_diagPrev.find(kv.first); const int pv = it == m_diagPrev.end() ? 0 : it->second;
                if (pv != kv.second) diffs.push_back({kv.second - pv, kv.first + Fmt(" cur=%d prev=%d", kv.second, pv)}); }
            for (const auto& kv : m_diagPrev) if (!cur.count(kv.first)) diffs.push_back({-kv.second, kv.first + Fmt(" cur=0 prev=%d", kv.second)});
            std::sort(diffs.begin(), diffs.end(), [](const auto& a, const auto& b) { return std::abs(a.first) > std::abs(b.first); });
            Log::Write(Log::Level::Info, Fmt("TCVR_S22ALT seq=%llu prims=%u groups=%zu differing=%zu", (unsigned long long)f.sequence, f.prim_count, cur.size(), diffs.size()));
            for (size_t i = 0; i < diffs.size() && i < 8; ++i) Log::Write(Log::Level::Info, "TCVR_S22ALT   " + diffs[i].second);
        }
        m_diagPrev.swap(cur);
    }
    bool m_diagAlt = false;
    uint64_t m_diagSeq = ~0ull;
    std::unordered_map<std::string, int> m_diagPrev;
    std::chrono::steady_clock::time_point m_diagLast{};
private:
    std::thread m_bankWorker;
    std::atomic<bool> m_bankCancel{false};
    std::mutex m_bankMutex;
    std::vector<std::pair<int, std::vector<uint8_t>>> m_bankQueue;
    std::vector<std::pair<std::unique_ptr<BufferAndMemory>, uint64_t>> m_bankStaging;
    uint32_t m_bankReadyMask = 0;
    uint64_t m_frameCounter = 0;
    int m_filter = 1;
    VkSampler m_repeatNearest = VK_NULL_HANDLE;
public:

    // ---- per frame (view 0, after the slot's fence) --------------------------------------------
    // CPU port of SceneRenderer::PrepareFrame, written into this slot's buffers.
    bool PrepareFrame(int slot, const tcvr_scene_frame& f) {
        if (!m_ready || !m_assetsReady) return false;
        m_slot = slot & 1;
        Slot& S = m_slots[m_slot];
        const auto tBuild = std::chrono::steady_clock::now();
        m_vertexData.clear(); m_indexData.clear(); m_primData.clear(); m_runs.clear(); m_ranges.clear();
        std::uint32_t lastPolyIndex = 0; (void)lastPolyIndex;
        bool anyPoly = false;
        m_fogBgValid = false;
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.kind == 0 && pr.direct == 0) anyPoly = true;
            if (!m_fogBgValid && pr.fog_mode == 2) { m_fogBg[0] = pr.fog_r; m_fogBg[1] = pr.fog_g; m_fogBg[2] = pr.fog_b; m_fogBgValid = true; }
        }
        if (m_diagAlt) DiagAlternating(f);
        m_groupCentre.clear();
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.kind != 1 || pr.vertex_count < 3) continue;
            auto& g = m_groupCentre[uint32_t(pr.cz_bank)];
            for (uint32_t i = 0; i < pr.vertex_count; ++i) { g.x += f.vertices[pr.first_vertex + i].x; g.y += f.vertices[pr.first_vertex + i].y; }
            g.n += float(pr.vertex_count);
        }
        if (f.sequence != m_altSeq) UpdateAlternation(f);
        float neighbourZoom = 1.0f;
        // One primitive into the draw arrays. `verts` = its own vertices; `half` draws it at 50 % (see
        // UpdateAlternation). The vertex carries the index of its row in OUR table (not MAME's index).
        auto emit = [&](const tcvr_scene_prim& pr0, const tcvr_scene_vertex* verts, bool half) {
            tcvr_scene_prim pr = pr0;
            if (half) { pr.alpha_enabled = 1; pr.alpha = 128; }
            const float primIdx = float(m_primData.size() / 64);
            float spriteDepth = 0.0f;
            if (pr.kind == 0 && pr.direct == 0) { neighbourZoom = pr.zoom; if (pr.zoom > 0.0f) m_lastZoom = pr.zoom; }
            float centreX = 0.0f, centreY = 0.0f;
            if (pr.kind == 1) {
                const auto it = m_groupCentre.find(uint32_t(pr.cz_bank));
                if (it != m_groupCentre.end() && it->second.n > 0.0f) { centreX = it->second.x / it->second.n; centreY = it->second.y / it->second.n; }
                if (anyPoly) spriteDepth = 1.0f;
            }
            const uint32_t base = uint32_t(m_vertexData.size() / 8);
            for (uint32_t i = 0; i < pr.vertex_count; ++i) {
                const tcvr_scene_vertex& v = verts[i];
                const float row[8] = {v.x, v.y, v.z, v.u, v.v, v.bri, pr.zoom, primIdx};
                m_vertexData.insert(m_vertexData.end(), row, row + 8);
            }
            const bool hud = pr.kind != 0 || pr.direct != 0;
            if (m_runs.empty() || m_runs.back().hud != hud) {
                if (!m_runs.empty()) m_runs.back().count = uint32_t(m_indexData.size()) - m_runs.back().first;
                m_runs.push_back({uint32_t(m_indexData.size()), 0, hud});
            }
            const uint32_t idx0 = uint32_t(m_indexData.size());
            for (uint32_t i = 1; i + 1 < pr.vertex_count; ++i) {
                m_indexData.push_back(base); m_indexData.push_back(base + i); m_indexData.push_back(base + i + 1);
            }
            m_ranges.push_back({idx0, uint32_t(m_indexData.size()) - idx0, pr.alpha == 0});
            const float row[64] = {
                float(pr.kind), float(pr.direct), float(pr.cx), float(pr.cy),
                float(pr.clip_l), float(pr.clip_t), float(pr.clip_r), float(pr.clip_b),
                float(pr.pens_offset), float(pr.bn), float(pr.penmask), float(pr.penshift),
                float(pr.texture_enabled), float(pr.shade_enabled), float(pr.prioverchar), float(pr.fog_mode),
                float(pr.fogfactor), float(pr.cz_sdelta), float(pr.cz_bank), float(pr.alpha_pen),
                pr.fog_r, pr.fog_g, pr.fog_b, float(pr.fade_enabled),
                pr.fade_r, pr.fade_g, pr.fade_b, float(pr.fadefactor),
                float(pr.pfade_enabled), pr.poly_r, pr.poly_g, pr.poly_b,
                float(pr.alpha_enabled), float(pr.alpha), float(pr.sprite_code), float(pr.flipx),
                float(pr.flipy), spriteDepth, neighbourZoom, 0,
                centreX, centreY, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            m_primData.insert(m_primData.end(), row, row + 64);
        };
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.vertex_count < 3 || pr.first_vertex + pr.vertex_count > f.vertex_count) continue;
            emit(pr, f.vertices + pr.first_vertex, m_altFix && p < m_altHalfPrim.size() && m_altHalfPrim[p] != 0);
        }
        if (m_altFix)
            for (size_t i = 0; i < m_altCarry.size(); ++i) emit(m_altCarry[i], m_altCarryVerts.data() + m_altCarryFirst[i], true);
        if (!m_runs.empty()) m_runs.back().count = uint32_t(m_indexData.size()) - m_runs.back().first;
        if (m_reorder) ReorderOpaqueFrontToBack();
        m_lastPrims = uint32_t(m_primData.size() / 64);
        m_lastIndices = uint32_t(m_indexData.size());
        const auto tCopy = std::chrono::steady_clock::now();
        // Copies into this slot (bounded by the buffer capacities).
        auto put = [](void* dst, size_t cap, const void* src, size_t n) { if (src && n) std::memcpy(dst, src, std::min(cap, n)); };
        put(S.vboMap, kVboBytes, m_vertexData.data(), m_vertexData.size() * sizeof(float));
        put(S.iboMap, kIboBytes, m_indexData.data(), m_indexData.size() * sizeof(uint32_t));
        put(S.primMap, kPrimBytes, m_primData.data(), m_primData.size() * sizeof(float));
        if (m_indexData.size() * 4 > kIboBytes || m_vertexData.size() * 4 > kVboBytes) m_lastIndices = 0;   // never draw truncated data
        put(S.pensMap, kPensBytes, f.pens, size_t(f.pen_count) * 4);
        if (f.czram && f.cz_entries == 0x2000) put(S.czMap, kCzBytes, f.czram, size_t(std::min<uint32_t>(f.cz_banks, 4)) * 0x2000);
        if (f.spotram) put(S.spotMap, kSpotBytes, f.spotram, 0x400 * 2);
        if (f.gamma_r && f.gamma_g && f.gamma_b) {
            uint8_t* g = static_cast<uint8_t*>(S.gammaMap);
            std::memcpy(g, f.gamma_r, 256); std::memcpy(g + 256, f.gamma_g, 256); std::memcpy(g + 512, f.gamma_b, 256);
        }
        m_textW = f.width; m_textH = f.height;
        const size_t tw = size_t(std::max(0, f.width)), th = size_t(std::max(0, f.height));
        if (f.text && tw * th * 2 <= kTextBytes) {
            uint16_t* d = static_cast<uint16_t*>(S.textMap);
            for (size_t y = 0; y < th; ++y) std::memcpy(d + y * tw, f.text + y * f.text_stride, tw * 2);
        }
        if (tw * th <= kPriBytes) {
            uint8_t* d = static_cast<uint8_t*>(S.priMap);
            if (f.pri) for (size_t y = 0; y < th; ++y) std::memcpy(d + y * tw, f.pri + y * f.pri_stride, tw);
            else std::memset(d, 0, tw * th);
        }
        {
            const auto tEnd = std::chrono::steady_clock::now();
            static double accB = 0, accC = 0; static int n = 0; static auto last = tEnd;
            accB += std::chrono::duration<double, std::milli>(tCopy - tBuild).count();
            accC += std::chrono::duration<double, std::milli>(tEnd - tCopy).count(); ++n;
            if (tEnd - last > std::chrono::seconds(2)) {
                Log::Write(Log::Level::Info, Fmt("TCVR_S22PREP ms/frame build=%.2f copy=%.2f prims=%u verts=%zu", accB / n, accC / n,
                                                 f.prim_count, m_vertexData.size() / 8));
                accB = accC = 0; n = 0; last = tEnd;
            }
        }
        m_frame = f;   // scalars for the uniforms (pointers not used later)
        m_havePrepared = true;
        return true;
    }

    // Depth map pass + its asynchronous readback (this slot's buffer is read the next time the slot is
    // fenced). Record at view 0, outside any render pass, after PrepareFrame.
    void RenderDepthMap(VkCommandBuffer cmd) {
        if (!m_havePrepared || m_lastIndices == 0) return;
        Slot& S = m_slots[m_slot];
        // Collect this slot's previous readback (its fence has passed).
        if (S.depthReadValid && m_dmW > 0) {
            m_depthCpu.resize(size_t(m_dmW) * m_dmH);
            const float* p = static_cast<const float*>(S.depthReadMap);
            std::memcpy(m_depthCpu.data(), p, m_depthCpu.size() * sizeof(float));
            m_depthCpuW = m_dmW; m_depthCpuH = m_dmH;
        }
        const uint32_t dw = uint32_t(std::max(1, m_textW / 2)), dh = uint32_t(std::max(1, m_textH / 2));
        if (dw != m_dmW || dh != m_dmH) { vkDeviceWaitIdle(m_dev); DestroyDepthMap(); CreateDepthMap(dw, dh); m_setsDirty = true; }
        UpdateSetsIfDirty();
        Ubo u{};
        FillCommonUbo(u, nullptr, nullptr, float(m_textW), float(m_textH));
        u.Flags[0] = 0;   // board projection
        std::memcpy(S.uboMap[2], &u, sizeof(u));
        VkClearValue cv[2]{}; cv[1].depthStencil = {0.0f, 0};   // reversed depth
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_rpDepth; bi.framebuffer = m_dmFb; bi.renderArea = {{0, 0}, {dw, dh}};
        bi.clearValueCount = 2; bi.pClearValues = cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        SetVp(cmd, dw, dh);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pDepth);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[2], 0, nullptr);
        BindGeometry(cmd, S);
        for (const Run& r : m_runs) if (!r.hud && r.count) vkCmdDrawIndexed(cmd, r.count, 1, r.first, 0, 0);
        vkCmdEndRenderPass(cmd);
        // Readback into this slot's host buffer.
        VkBufferImageCopy c{};
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {dw, dh, 1};
        vkCmdCopyImageToBuffer(cmd, m_dmImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, S.depthRead.buf, 1, &c);
        Barrier(cmd, m_dmImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        S.depthReadValid = true;
    }

    // One eye: scene pass (offscreen, scaled) then composite straight into `target` (the eye image).
    bool RenderEye(VkCommandBuffer cmd, uint32_t eye, const float mvp[16], const float hudMvp[16], VkImage target,
                   VkExtent2D outExt, const VkRect2D& area, float renderScale, const Settings& st) {
        if (!m_havePrepared || !m_assetsReady) return false;
        eye = eye & 1;
        Slot& S = m_slots[m_slot];
        const uint32_t rw = std::max(64u, uint32_t(float(outExt.width) * renderScale));
        const uint32_t rh = std::max(64u, uint32_t(float(outExt.height) * renderScale));
        EyeTarget& T = m_eye[eye];
        if (T.w != rw || T.h != rh) { vkDeviceWaitIdle(m_dev); DestroyEyeTarget(T); CreateEyeTarget(T, rw, rh); m_setsDirty = true; }
        UpdateSetsIfDirty();
        m_settings = st;
        Ubo u{};
        FillCommonUbo(u, mvp, hudMvp, float(rw), float(rh));
        std::memcpy(S.uboMap[eye], &u, sizeof(u));
        // ---- S: scene pass
        VkClearValue cv[5]{};
        cv[2].depthStencil = {0.0f, 0};   // reversed depth
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        AdvanceTarget(T);
        bi.renderPass = m_rpScene; bi.framebuffer = T.fbs[T.cur]; bi.renderArea = {{0, 0}, {rw, rh}};
        bi.clearValueCount = IsMsaa() ? 5 : 3; bi.pClearValues = cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        SetVp(cmd, rw, rh);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[eye], 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pInit);
        if ((st.skip & 1) == 0) vkCmdDraw(cmd, 3, 1, 0, 0);
        if (m_lastIndices > 0 && (st.skip & 2) == 0) {
            BindGeometry(cmd, S);
            for (const Run& r : m_runs) {
                if (!r.count) continue;
                VkPipeline pl = (r.hud || !st.depthTest) ? m_pSceneHud
                               : (st.lean == 0 ? m_pScene3D : (st.texSamples > 1 ? m_pP3 : m_pP3NoAa));
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl);
                vkCmdDrawIndexed(cmd, r.count, 1, r.first, 0, 0);
            }
        }
        vkCmdEndRenderPass(cmd);
        // ---- C: composite into the eye image
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_compFbs) if (f.image == target) fb = f.fb;
        if (!fb) {
            CompFb f{}; f.image = target;
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = target; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = m_eyeFormat;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &vi, nullptr, &f.view));
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_rpComp; fi.attachmentCount = 1; fi.pAttachments = &f.view;
            fi.width = outExt.width; fi.height = outExt.height; fi.layers = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &f.fb));
            m_compFbs.push_back(f); fb = f.fb;
        }
        // Composite uniforms: output size = the eye image (text lookup per output pixel).
        Ubo uc = u;
        uc.OutText[0] = float(area.extent.width); uc.OutText[1] = float(area.extent.height);
        uc.ScreenOut[2] = float(area.extent.width) / float(std::max(1, m_textW));
        uc.ScreenOut[3] = float(area.extent.height) / float(std::max(1, m_textH));
        uc.Sprite[2] = T.cur; uc.Sprite[3] = (st.darkFlicker && T.havePrev) ? 1 : 0;
        std::memcpy(S.uboMap[3 + eye], &uc, sizeof(uc));
        VkRenderPassBeginInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        ci.renderPass = m_rpComp; ci.framebuffer = fb; ci.renderArea = area; ci.clearValueCount = 0;
        vkCmdBeginRenderPass(cmd, &ci, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{float(area.offset.x), float(area.offset.y), float(area.extent.width), float(area.extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &area);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pComp);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[3 + eye], 0, nullptr);
        if ((st.skip & 4) == 0) vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        return true;
    }

    // SCREEN presentation: the board's own projection, painter order exactly like the hardware (no depth
    // test), drawn at `scale` x the board size, text + gamma composited into a sampled image that the
    // virtual screen shows. MAME no longer rasterises (scene mode 2): that is what keeps it at 60 Hz.
    bool RenderFlat(VkCommandBuffer cmd, float scale, const Settings& st) {
        if (!m_havePrepared || !m_assetsReady || m_frame.width <= 0 || m_frame.height <= 0) return false;
        Slot& S = m_slots[m_slot];
        const uint32_t w = uint32_t(float(m_frame.width) * scale), h = uint32_t(float(m_frame.height) * scale);
        EyeTarget& T = m_eye[2];
        if (T.w != w || T.h != h) {
            vkDeviceWaitIdle(m_dev); DestroyEyeTarget(T); CreateEyeTarget(T, w, h); DestroyFlatOut(); CreateFlatOut(w, h);
            m_setsDirty = true;
        }
        UpdateSetsIfDirty();
        m_settings = st;
        Ubo u{};
        FillCommonUbo(u, nullptr, nullptr, float(w), float(h));
        std::memcpy(S.uboMap[5], &u, sizeof(u));
        VkClearValue cv[5]{};
        cv[2].depthStencil = {0.0f, 0};   // reversed depth
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        AdvanceTarget(T);
        bi.renderPass = m_rpScene; bi.framebuffer = T.fbs[T.cur]; bi.renderArea = {{0, 0}, {w, h}};
        bi.clearValueCount = IsMsaa() ? 5 : 3; bi.pClearValues = cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        SetVp(cmd, w, h);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[5], 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pInit);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        if (m_lastIndices > 0) {
            BindGeometry(cmd, S);
            // 3D runs with the depth test: the recorded order is NOT the painter order (and the opaque
            // polygons are re-sorted front to back anyway); the board projection gives a monotonic depth.
            for (const Run& r : m_runs) {
                if (!r.count) continue;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, r.hud ? m_pSceneHud : m_pScene3D);
                vkCmdDrawIndexed(cmd, r.count, 1, r.first, 0, 0);
            }
        }
        vkCmdEndRenderPass(cmd);
        Ubo uc = u;
        uc.Sprite[2] = T.cur; uc.Sprite[3] = (st.darkFlicker && T.havePrev) ? 1 : 0;
        std::memcpy(S.uboMap[6], &uc, sizeof(uc));
        VkRenderPassBeginInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        ci.renderPass = m_rpComp; ci.framebuffer = m_flatOutFb; ci.renderArea = {{0, 0}, {w, h}};
        vkCmdBeginRenderPass(cmd, &ci, VK_SUBPASS_CONTENTS_INLINE);
        SetVp(cmd, w, h);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pComp);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[6], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        Barrier(cmd, m_flatOut.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        return true;
    }
    VkImageView FlatView() const { return m_flatOut.view; }
    VkImage FlatImage() const { return m_flatOut.image; }
    VkFormat FlatFormat() const { return m_eyeFormat; }
    uint32_t FlatWidth() const { return m_eye[2].w; }
    uint32_t FlatHeight() const { return m_eye[2].h; }

    // Immersive aim (port of SceneRenderer::RayCast): march the ray in board camera space until it meets
    // the depth map. False when it leaves the screen or meets nothing.
    bool RayCast(const float o[3], const float d[3], int screenW, int screenH, float& sx, float& sy, float hit[3]) const {
        if (m_depthCpu.empty() || m_depthCpuW == 0 || d[2] <= 1e-6f) return false;
        const float zoom = m_lastZoom > 0.0f ? m_lastZoom : 1.0f;
        const float kx = float(m_depthCpuW) / float(screenW), ky = float(m_depthCpuH) / float(screenH);
        float t = (100.0f - o[2]) / d[2];
        if (t < 0.0f) t = 0.0f;
        for (int i = 0; i < 600; ++i) {
            const float px = o[0] + t * d[0], py = o[1] + t * d[1], pz = o[2] + t * d[2];
            if (pz > 1.0f) {
                const float x = screenW * 0.5f + zoom * px / pz, y = screenH * 0.5f - zoom * py / pz;
                if (x < 0.0f || y < 0.0f || x >= screenW || y >= screenH) return false;
                const float depth = m_depthCpu[size_t(int(y * ky)) * m_depthCpuW + size_t(int(x * kx))];
                if (depth > 0.0f && pz >= depth) { sx = x; sy = y; hit[0] = px; hit[1] = py; hit[2] = pz; return true; }
                if (pz > 3.0e6f) return false;
            }
            t = std::max(t * 1.02f, t + 20.0f);
        }
        return false;
    }

private:
    static constexpr size_t kVboBytes = 64 * 1024 * 8 * 4;     // 64k vertices x 8 floats
    static constexpr size_t kIboBytes = 256 * 1024 * 4;
    static constexpr size_t kPrimBytes = 16384 * 64 * 4;       // 16k primitives x 16 vec4
    static constexpr size_t kPensBytes = 65536 * 4;
    static constexpr size_t kCzBytes = 4 * 0x2000;
    static constexpr size_t kSpotBytes = 0x400 * 2;
    static constexpr size_t kGammaBytes = 768;
    static constexpr size_t kTextBytes = 1024 * 1024 * 2;
    static constexpr size_t kPriBytes = 1024 * 1024;

    struct Img { VkImage image = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
    struct Run { uint32_t first, count; bool hud; };
    struct Range { uint32_t first, count; bool opaque; };
    std::vector<Range> m_ranges;
    std::vector<uint32_t> m_reorderTmp;
    bool m_reorder = true;
public:
    void SetReorder(bool on) { m_reorder = on; }
private:
    // The board paints back to front. Depth already encodes that order (per-primitive bias in the vertex
    // stage), so the opaque result is order independent: draw the opaque polygons of each 3D run front to
    // back (early-Z rejects what is hidden, instead of shading every layer), then its translucent ones in
    // the original order on top. HUD runs, and the order between runs, are untouched.
    void ReorderOpaqueFrontToBack() {
        size_t ri = 0;
        for (const Run& run : m_runs) {
            const uint32_t end = run.first + run.count;
            const size_t r0 = ri;
            while (ri < m_ranges.size() && m_ranges[ri].first < end) ++ri;
            if (run.hud || run.count == 0) continue;
            m_reorderTmp.clear();
            for (size_t k = ri; k-- > r0;)
                if (m_ranges[k].opaque) m_reorderTmp.insert(m_reorderTmp.end(), m_indexData.begin() + m_ranges[k].first,
                                                             m_indexData.begin() + m_ranges[k].first + m_ranges[k].count);
            for (size_t k = r0; k < ri; ++k)
                if (!m_ranges[k].opaque) m_reorderTmp.insert(m_reorderTmp.end(), m_indexData.begin() + m_ranges[k].first,
                                                             m_indexData.begin() + m_ranges[k].first + m_ranges[k].count);
            if (m_reorderTmp.size() == run.count) std::copy(m_reorderTmp.begin(), m_reorderTmp.end(), m_indexData.begin() + run.first);
        }
    }
    struct GroupAcc { float x = 0, y = 0, n = 0; };
    struct Slot {
        BufferAndMemory vbo, ibo, prim, pens, cz, spot, gamma, text, pri, ubo[7], depthRead;
        void *vboMap = nullptr, *iboMap = nullptr, *primMap = nullptr, *pensMap = nullptr, *czMap = nullptr, *spotMap = nullptr,
             *gammaMap = nullptr, *textMap = nullptr, *priMap = nullptr, *uboMap[7] = {}, *depthReadMap = nullptr;
        VkDescriptorSet set[7] = {};   // eye0, eye1, depth map, composite eye0, composite eye1, flat scene, flat composite
        bool depthReadValid = false;
    };
    struct EyeTarget {
        uint32_t w = 0, h = 0;
        Img msColor, msPri, depth, color, pri;   // color: 2 layers (current / previous arcade frame), view = array
        VkImageView layerView[2] = {};
        VkFramebuffer fbs[2] = {};
        int cur = 0;
        bool havePrev = false;
        uint64_t seq = ~0ull;
    };
    // The resolved scene alternates layers only when MAME delivers a new frame, so the other layer is the
    // PREVIOUS ARCADE FRAME (the dark-flicker gate compares against it, like the GL path did).
    void AdvanceTarget(EyeTarget& T) {
        if (m_frame.sequence == T.seq) return;
        if (T.seq != ~0ull) { T.cur ^= 1; T.havePrev = true; }
        T.seq = m_frame.sequence;
    }
    struct CompFb { VkImage image; VkImageView view; VkFramebuffer fb; };

    bool IsMsaa() const { return m_samples != VK_SAMPLE_COUNT_1_BIT; }

    void FillCommonUbo(Ubo& u, const float* mvp, const float* hud, float outW, float outH) {
        const tcvr_scene_frame& f = m_frame;
        if (mvp) std::memcpy(u.ImmersiveMvp, mvp, 64);
        if (hud) std::memcpy(u.HudMvp, hud, 64);
        u.ScreenOut[0] = float(f.width); u.ScreenOut[1] = float(f.height);
        u.ScreenOut[2] = outW / float(std::max(1, f.width)); u.ScreenOut[3] = outH / float(std::max(1, f.height));
        u.OutText[0] = outW; u.OutText[1] = outH; u.OutText[2] = float(f.width); u.OutText[3] = float(f.height);
        u.DepthInfo[0] = 0.5f; u.DepthInfo[1] = float(m_dmW); u.DepthInfo[2] = float(m_dmH); u.DepthInfo[3] = m_settings.spriteMinDepth;
        u.Flags[0] = mvp ? 1 : 0; u.Flags[1] = hud ? 1 : 0; u.Flags[2] = mvp ? m_settings.texSamples : 1; u.Flags[3] = m_spritesPerRow;
        u.Sprite[0] = m_spriteW; u.Sprite[1] = m_spriteH;
        // Scene passes: Sprite.z = decoded pen banks ready (bit per bank), Sprite.w = texture filter
        // (0 texel-exact like the board, 1 bilinear). The composite overwrites both with its own meaning.
        u.Sprite[2] = int(m_bankReadyMask); u.Sprite[3] = m_filter;
        if (mvp && m_settings.voidMode == 2) {
            u.Bg[0] = m_settings.voidRGB[0]; u.Bg[1] = m_settings.voidRGB[1]; u.Bg[2] = m_settings.voidRGB[2];
        } else if (mvp && m_settings.voidMode == 1) {
            const bool fogUsable = m_fogBgValid && (m_fogBg[0] + m_fogBg[1] + m_fogBg[2]) > 24.0f;
            if (fogUsable) { u.Bg[0] = unsigned(m_fogBg[0] + 0.5f); u.Bg[1] = unsigned(m_fogBg[1] + 0.5f); u.Bg[2] = unsigned(m_fogBg[2] + 0.5f); }
            else { u.Bg[0] = 28; u.Bg[1] = 30; u.Bg[2] = 38; }
        } else {
            u.Bg[0] = (f.bg_color >> 16) & 0xff; u.Bg[1] = (f.bg_color >> 8) & 0xff; u.Bg[2] = f.bg_color & 0xff;
        }
        u.Mix0[0] = int(f.mix_spot_enabled && f.spotram); u.Mix0[1] = int(f.mix_spot_factor);
        u.Mix0[2] = int(f.mix_spot_palbase); u.Mix0[3] = int(f.text_palbase);
        u.Mix1[0] = f.mix_fade_enabled; u.Mix1[1] = f.mix_fade_factor & 0xff; u.Mix1[2] = f.mix_alpha_factor & 0xff; u.Mix1[3] = f.mix_alpha_mask & 0xf;
        u.Mix2[0] = f.mix_alpha_check12 & 0xff; u.Mix2[1] = f.mix_alpha_check13 & 0xff;
        u.FadeColor[0] = unsigned(f.mix_fade_r); u.FadeColor[1] = unsigned(f.mix_fade_g); u.FadeColor[2] = unsigned(f.mix_fade_b);
        u.Bias[0] = (mvp && m_settings.depthTest) ? m_settings.depthBias : 0.0f;
        u.Bias[3] = m_settings.nearClip;
    }

    void BindGeometry(VkCommandBuffer cmd, Slot& S) {
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &S.vbo.buf, &off);
        vkCmdBindIndexBuffer(cmd, S.ibo.buf, 0, VK_INDEX_TYPE_UINT32);
    }
    static void SetVp(VkCommandBuffer cmd, uint32_t w, uint32_t h) {
        VkViewport vp{0.0f, 0.0f, float(w), float(h), 0.0f, 1.0f};
        VkRect2D sc{{0, 0}, {w, h}};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
    }
    static void Barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da,
                        VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = sa; b.dstAccessMask = da; b.oldLayout = from; b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    VkShaderModule Mod(const uint8_t* code, size_t n) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = n; ci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule m = VK_NULL_HANDLE;
        XRC_CHECK_THROW_VKCMD(vkCreateShaderModule(m_dev, &ci, nullptr, &m));
        return m;
    }

    void CreateLayouts() {
        std::vector<VkDescriptorSetLayoutBinding> b;
        auto add = [&](uint32_t i, VkDescriptorType t) {
            VkDescriptorSetLayoutBinding x{}; x.binding = i; x.descriptorType = t; x.descriptorCount = 1;
            x.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; b.push_back(x);
        };
        add(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        for (uint32_t i = 1; i <= 7; ++i) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        for (uint32_t i = 8; i <= 16; ++i) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = uint32_t(b.size()); li.pBindings = b.data();
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_dev, &li, nullptr, &m_setLayout));
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_setLayout;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_dev, &pli, nullptr, &m_layout));
        VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 160}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 16; pi.poolSizeCount = 3; pi.pPoolSizes = ps;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorPool(m_dev, &pi, nullptr, &m_pool));
    }

    VkAttachmentDescription Att(VkFormat fmt, VkSampleCountFlagBits s, VkAttachmentLoadOp lo, VkAttachmentStoreOp so,
                                VkImageLayout init, VkImageLayout fin) {
        VkAttachmentDescription a{};
        a.format = fmt; a.samples = s; a.loadOp = lo; a.storeOp = so;
        a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout = init; a.finalLayout = fin;
        return a;
    }

    void CreateRenderPasses() {
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkSubpassDependency depOut{};
        depOut.srcSubpass = 0; depOut.dstSubpass = VK_SUBPASS_EXTERNAL;
        depOut.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        depOut.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        depOut.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        depOut.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        VkSubpassDependency deps[2] = {dep, depOut};
        // D: depth map. R32F colour -> TRANSFER_SRC (read back, then made shader-readable), depth transient.
        {
            VkAttachmentDescription at[2] = {
                Att(VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                Att(VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)};
            VkAttachmentReference c{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, d{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sp{}; sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sp.colorAttachmentCount = 1; sp.pColorAttachments = &c; sp.pDepthStencilAttachment = &d;
            VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            ri.attachmentCount = 2; ri.pAttachments = at; ri.subpassCount = 1; ri.pSubpasses = &sp; ri.dependencyCount = 2; ri.pDependencies = deps;
            XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_dev, &ri, nullptr, &m_rpDepth));
        }
        // S: scene. colour RGBA8 + priority R8 (+ MSAA resolves), depth transient; outputs shader-readable.
        {
            const bool ms = IsMsaa();
            std::vector<VkAttachmentDescription> at;
            const VkImageLayout outL = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            at.push_back(Att(VK_FORMAT_R8G8B8A8_UNORM, m_samples, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                             ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                             ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : outL));
            at.push_back(Att(VK_FORMAT_R8_UNORM, m_samples, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                             ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                             ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : outL));
            at.push_back(Att(VK_FORMAT_D32_SFLOAT, m_samples, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
            if (ms) {
                at.push_back(Att(VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                 VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, outL));
                at.push_back(Att(VK_FORMAT_R8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                 VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, outL));
            }
            VkAttachmentReference c[2] = {{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
            VkAttachmentReference r[2] = {{3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, {4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
            VkAttachmentReference d{2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sp{}; sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sp.colorAttachmentCount = 2; sp.pColorAttachments = c; sp.pDepthStencilAttachment = &d;
            sp.pResolveAttachments = ms ? r : nullptr;
            VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            ri.attachmentCount = uint32_t(at.size()); ri.pAttachments = at.data(); ri.subpassCount = 1; ri.pSubpasses = &sp;
            ri.dependencyCount = 2; ri.pDependencies = deps;
            XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_dev, &ri, nullptr, &m_rpScene));
        }
        // C: composite into the eye image (fully overwritten).
        {
            VkAttachmentDescription at = Att(m_eyeFormat, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE,
                                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VkAttachmentReference c{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
            VkSubpassDescription sp{}; sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sp.colorAttachmentCount = 1; sp.pColorAttachments = &c;
            VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            ri.attachmentCount = 1; ri.pAttachments = &at; ri.subpassCount = 1; ri.pSubpasses = &sp; ri.dependencyCount = 1; ri.pDependencies = &dep;
            XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_dev, &ri, nullptr, &m_rpComp));
        }
    }

    void CreatePipelines() {
        VkShaderModule qv = Mod(c_s22QuadVertSpv, sizeof(c_s22QuadVertSpv)), ifr = Mod(c_s22InitFragSpv, sizeof(c_s22InitFragSpv));
        VkShaderModule sv = Mod(c_s22SceneVertSpv, sizeof(c_s22SceneVertSpv)), sf = Mod(c_s22SceneFragSpv, sizeof(c_s22SceneFragSpv));
        VkShaderModule df = Mod(c_s22DepthFragSpv, sizeof(c_s22DepthFragSpv)), cf = Mod(c_s22CompFragSpv, sizeof(c_s22CompFragSpv));
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1; vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineMultisampleStateCreateInfo ms1{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms1.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineMultisampleStateCreateInfo msN = ms1; msN.rasterizationSamples = m_samples;
        VkPipelineVertexInputStateCreateInfo viNone{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkVertexInputBindingDescription bd{0, 8 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription ad[3] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
                                                   {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 3 * sizeof(float)},
                                                   {2, 0, VK_FORMAT_R32G32_SFLOAT, 6 * sizeof(float)}};
        VkPipelineVertexInputStateCreateInfo viScene{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        viScene.vertexBindingDescriptionCount = 1; viScene.pVertexBindingDescriptions = &bd;
        viScene.vertexAttributeDescriptionCount = 3; viScene.pVertexAttributeDescriptions = ad;
        VkPipelineDepthStencilStateCreateInfo dsOff{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineDepthStencilStateCreateInfo dsTest = dsOff;
        dsTest.depthTestEnable = VK_TRUE; dsTest.depthWriteEnable = VK_TRUE; dsTest.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;   // reversed depth (see s22_scene_vert)
        VkPipelineDepthStencilStateCreateInfo dsLess = dsTest; dsLess.depthCompareOp = VK_COMPARE_OP_GREATER;
        VkPipelineColorBlendAttachmentState noBlend{}; noBlend.colorWriteMask = 0xf;
        // Scene colour: src*a + dst*(1-a), alpha kept (glBlendFuncSeparate(SRC_ALPHA, 1-SRC_ALPHA, ZERO, ONE)).
        VkPipelineColorBlendAttachmentState sceneBlend{};
        sceneBlend.blendEnable = VK_TRUE;
        sceneBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; sceneBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        sceneBlend.colorBlendOp = VK_BLEND_OP_ADD;
        sceneBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; sceneBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        sceneBlend.alphaBlendOp = VK_BLEND_OP_ADD; sceneBlend.colorWriteMask = 0xf;
        VkPipelineColorBlendAttachmentState initAtt[2] = {noBlend, noBlend};
        VkPipelineColorBlendAttachmentState sceneAtt[2] = {sceneBlend, noBlend};
        VkPipelineColorBlendStateCreateInfo cbInit{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cbInit.attachmentCount = 2; cbInit.pAttachments = initAtt;
        VkPipelineColorBlendStateCreateInfo cbScene = cbInit; cbScene.pAttachments = sceneAtt;
        VkPipelineColorBlendStateCreateInfo cbOne = cbInit; cbOne.attachmentCount = 1; cbOne.pAttachments = &noBlend;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[0].pName = st[1].pName = "main";
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 2; pi.pStages = st; pi.pInputAssemblyState = &ia; pi.pViewportState = &vps;
        pi.pRasterizationState = &rs; pi.pDynamicState = &ds; pi.layout = m_layout;
        auto make = [&](VkShaderModule v, VkShaderModule f, VkPipelineVertexInputStateCreateInfo* vi, VkPipelineMultisampleStateCreateInfo* ms,
                        VkPipelineDepthStencilStateCreateInfo* dss, VkPipelineColorBlendStateCreateInfo* cb, VkRenderPass rp, VkPipeline* out) {
            st[0].module = v; st[1].module = f;
            pi.pVertexInputState = vi; pi.pMultisampleState = ms; pi.pDepthStencilState = dss; pi.pColorBlendState = cb; pi.renderPass = rp;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pi, nullptr, out));
        };
        make(qv, ifr, &viNone, &msN, &dsOff, &cbInit, m_rpScene, &m_pInit);
        make(sv, sf, &viScene, &msN, &dsTest, &cbScene, m_rpScene, &m_pScene3D);
        make(sv, sf, &viScene, &msN, &dsOff, &cbScene, m_rpScene, &m_pSceneHud);
        // Lean variants for the 3D polygon runs (no discard, no sprite path): the uber-shader keeps the
        // Adreno from overlapping texture latency (same lesson as the Model 2 GOLD).
        VkShaderModule p3 = Mod(c_s22SceneFragP3Spv, sizeof(c_s22SceneFragP3Spv));
        VkShaderModule p3n = Mod(c_s22SceneFragP3NoAaSpv, sizeof(c_s22SceneFragP3NoAaSpv));
        make(sv, p3, &viScene, &msN, &dsTest, &cbScene, m_rpScene, &m_pP3);
        make(sv, p3n, &viScene, &msN, &dsTest, &cbScene, m_rpScene, &m_pP3NoAa);
        vkDestroyShaderModule(m_dev, p3, nullptr); vkDestroyShaderModule(m_dev, p3n, nullptr);
        make(sv, df, &viScene, &ms1, &dsLess, &cbOne, m_rpDepth, &m_pDepth);
        make(qv, cf, &viNone, &ms1, &dsOff, &cbOne, m_rpComp, &m_pComp);
        for (VkShaderModule m : {qv, ifr, sv, sf, df, cf}) vkDestroyShaderModule(m_dev, m, nullptr);
    }

    void CreateSamplerAndDummies() {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_dev, &si, nullptr, &m_nearest));
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_dev, &si, nullptr, &m_linear));
        si.magFilter = si.minFilter = VK_FILTER_NEAREST;   // integer image: gather only, wraps like the 12-bit address
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_dev, &si, nullptr, &m_repeatNearest));
        m_dummyU = NewImage(VK_FORMAT_R8_UINT, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_dummyF = NewImage(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_dummyArr = NewImage(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_SAMPLE_COUNT_1_BIT, false, VK_IMAGE_ASPECT_COLOR_BIT, 2);
        m_dummyU16 = NewImage(VK_FORMAT_R8_UINT, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_SAMPLE_COUNT_1_BIT, false, VK_IMAGE_ASPECT_COLOR_BIT, 2);
        m_dummyPending = true;
    }

    Img NewImage(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkSampleCountFlagBits s = VK_SAMPLE_COUNT_1_BIT,
                 bool transient = false, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT, uint32_t layers = 1, uint32_t mips = 1) {
        Img im;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = mips; ii.arrayLayers = layers;
        ii.samples = s; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage | (transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0);
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        XRC_CHECK_THROW_VKCMD(vkCreateImage(m_dev, &ii, nullptr, &im.image));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(m_dev, im.image, &req);
        bool done = false;
        if (transient) { try { m_alloc->Allocate(req, &im.mem, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT); done = true; } catch (...) {} }
        if (!done) m_alloc->Allocate(req, &im.mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_dev, im.image, im.mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.image; vi.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {aspect, 0, mips, 0, layers};
        XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &vi, nullptr, &im.view));
        return im;
    }
    void DestroyImage(Img& im) {
        if (im.view) vkDestroyImageView(m_dev, im.view, nullptr);
        if (im.image) vkDestroyImage(m_dev, im.image, nullptr);
        if (im.mem) vkFreeMemory(m_dev, im.mem, nullptr);
        im = Img{};
    }

    // Static texture upload through a temporary staging buffer (kept until the next Destroy / reset).
    Img MakeImage(VkCommandBuffer cmd, VkFormat fmt, uint32_t w, uint32_t h, const void* data, uint32_t bpp) {
        Img im = NewImage(fmt, w, h, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        const size_t bytes = size_t(w) * h * bpp;
        m_stagings.emplace_back();
        BufferAndMemory& sb = m_stagings.back();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        sb.Create(m_dev, *m_alloc, bi);
        void* p = nullptr;
        XRC_CHECK_THROW_VKCMD(vkMapMemory(m_dev, sb.mem, 0, bytes, 0, &p));
        std::memcpy(p, data, bytes);
        vkUnmapMemory(m_dev, sb.mem);
        Barrier(cmd, im.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, sb.buf, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        Barrier(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        if (m_dummyPending) {   // the dummies ride on the first asset upload
            for (Img* d : {&m_dummyU, &m_dummyF, &m_dummyArr, &m_dummyU16}) {
                Barrier(cmd, d->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            }
            m_dummyPending = false;
        }
        return im;
    }

    void CreateSlot(int s) {
        Slot& S = m_slots[s];
        auto mk = [&](BufferAndMemory& b, void** map, size_t n, VkBufferUsageFlags u) {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = n; bi.usage = u;
            b.Create(m_dev, *m_alloc, bi);
            XRC_CHECK_THROW_VKCMD(vkMapMemory(m_dev, b.mem, 0, n, 0, map));
            std::memset(*map, 0, n);
        };
        mk(S.vbo, &S.vboMap, kVboBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        mk(S.ibo, &S.iboMap, kIboBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        mk(S.prim, &S.primMap, kPrimBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.pens, &S.pensMap, kPensBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.cz, &S.czMap, kCzBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.spot, &S.spotMap, kSpotBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.gamma, &S.gammaMap, kGammaBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.text, &S.textMap, kTextBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        mk(S.pri, &S.priMap, kPriBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        for (int i = 0; i < 7; ++i) mk(S.ubo[i], &S.uboMap[i], sizeof(Ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        mk(S.depthRead, &S.depthReadMap, 1024 * 1024 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkDescriptorSetLayout ls[7] = {m_setLayout, m_setLayout, m_setLayout, m_setLayout, m_setLayout, m_setLayout, m_setLayout};
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = m_pool; ai.descriptorSetCount = 7; ai.pSetLayouts = ls;
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_dev, &ai, S.set));
        m_setsDirty = true;
    }
    void DestroySlot(int s) {
        Slot& S = m_slots[s];
        for (BufferAndMemory* b : {&S.vbo, &S.ibo, &S.prim, &S.pens, &S.cz, &S.spot, &S.gamma, &S.text, &S.pri, &S.depthRead}) b->Reset(m_dev);
        for (auto& u : S.ubo) u.Reset(m_dev);
        for (auto& st : m_stagings) st.Reset(m_dev);
        m_stagings.clear();
    }

    void CreateDepthMap(uint32_t w, uint32_t h) {
        m_dmW = w; m_dmH = h;
        Img c = NewImage(VK_FORMAT_R32_SFLOAT, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        m_dmImage = c.image; m_dmMem = c.mem; m_dmView = c.view;
        m_dmDepth = NewImage(VK_FORMAT_D32_SFLOAT, w, h, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT, true, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkImageView att[2] = {m_dmView, m_dmDepth.view};
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = m_rpDepth; fi.attachmentCount = 2; fi.pAttachments = att; fi.width = w; fi.height = h; fi.layers = 1;
        XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &m_dmFb));
        for (auto& S : m_slots) S.depthReadValid = false;
        m_depthCpu.clear();
    }
    void DestroyDepthMap() {
        if (m_dmFb) vkDestroyFramebuffer(m_dev, m_dmFb, nullptr);
        if (m_dmView) vkDestroyImageView(m_dev, m_dmView, nullptr);
        if (m_dmImage) vkDestroyImage(m_dev, m_dmImage, nullptr);
        if (m_dmMem) vkFreeMemory(m_dev, m_dmMem, nullptr);
        DestroyImage(m_dmDepth);
        m_dmFb = VK_NULL_HANDLE; m_dmView = VK_NULL_HANDLE; m_dmImage = VK_NULL_HANDLE; m_dmMem = VK_NULL_HANDLE;
        m_dmW = m_dmH = 0;
    }

    void CreateEyeTarget(EyeTarget& T, uint32_t w, uint32_t h) {
        T.w = w; T.h = h;
        const VkImageUsageFlags att = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        T.color = NewImage(VK_FORMAT_R8G8B8A8_UNORM, w, h, att | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SAMPLE_COUNT_1_BIT, false,
                           VK_IMAGE_ASPECT_COLOR_BIT, 2);
        for (uint32_t l = 0; l < 2; ++l) {
            VkImageViewCreateInfo lv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            lv.image = T.color.image; lv.viewType = VK_IMAGE_VIEW_TYPE_2D; lv.format = VK_FORMAT_R8G8B8A8_UNORM;
            lv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, l, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &lv, nullptr, &T.layerView[l]));
        }
        T.pri = NewImage(VK_FORMAT_R8_UNORM, w, h, att | VK_IMAGE_USAGE_SAMPLED_BIT);
        T.depth = NewImage(VK_FORMAT_D32_SFLOAT, w, h, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m_samples, true, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (IsMsaa()) {
            T.msColor = NewImage(VK_FORMAT_R8G8B8A8_UNORM, w, h, att, m_samples, true);
            T.msPri = NewImage(VK_FORMAT_R8_UNORM, w, h, att, m_samples, true);
        }
        for (int l = 0; l < 2; ++l) {
            std::vector<VkImageView> v;
            if (IsMsaa()) v = {T.msColor.view, T.msPri.view, T.depth.view, T.layerView[l], T.pri.view};
            else v = {T.layerView[l], T.pri.view, T.depth.view};
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_rpScene; fi.attachmentCount = uint32_t(v.size()); fi.pAttachments = v.data();
            fi.width = w; fi.height = h; fi.layers = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &T.fbs[l]));
        }
        Log::Write(Log::Level::Info, Fmt("TCVR_S22VK eye target %ux%u MSAA x%d", w, h, int(m_samples)));
    }
    void CreateFlatOut(uint32_t w, uint32_t h) {
        m_flatOut = NewImage(m_eyeFormat, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = m_rpComp; fi.attachmentCount = 1; fi.pAttachments = &m_flatOut.view;
        fi.width = w; fi.height = h; fi.layers = 1;
        XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &m_flatOutFb));
    }
    void DestroyFlatOut() {
        if (m_flatOutFb) vkDestroyFramebuffer(m_dev, m_flatOutFb, nullptr);
        m_flatOutFb = VK_NULL_HANDLE;
        DestroyImage(m_flatOut);
    }
    void DestroyEyeTarget(EyeTarget& T) {
        for (int l = 0; l < 2; ++l) {
            if (T.fbs[l]) vkDestroyFramebuffer(m_dev, T.fbs[l], nullptr);
            if (T.layerView[l]) vkDestroyImageView(m_dev, T.layerView[l], nullptr);
        }
        DestroyImage(T.msColor); DestroyImage(T.msPri); DestroyImage(T.depth); DestroyImage(T.color); DestroyImage(T.pri);
        T = EyeTarget{};
    }

    void UpdateSetsIfDirty() {
        if (!m_setsDirty || !m_assetsReady) return;
        vkDeviceWaitIdle(m_dev);   // rare: new assets or new targets
        for (int s = 0; s < kFrames; ++s) {
            Slot& S = m_slots[s];
            for (int k = 0; k < 7; ++k) {
                std::vector<VkWriteDescriptorSet> w;
                VkDescriptorBufferInfo bufs[8] = {
                    {S.ubo[k].buf, 0, sizeof(Ubo)}, {S.prim.buf, 0, kPrimBytes}, {S.pens.buf, 0, kPensBytes}, {S.cz.buf, 0, kCzBytes},
                    {S.spot.buf, 0, kSpotBytes}, {S.gamma.buf, 0, kGammaBytes}, {S.text.buf, 0, kTextBytes}, {S.pri.buf, 0, kPriBytes}};
                for (uint32_t b = 0; b < 8; ++b) {
                    VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    x.dstSet = S.set[k]; x.dstBinding = b; x.descriptorCount = 1;
                    x.descriptorType = b == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    x.pBufferInfo = &bufs[b];
                    w.push_back(x);
                }
                const int eye = (k == 4) ? 1 : (k == 6 ? 2 : 0);   // composite sets read their eye's scene
                VkImageView dm = m_dmView ? m_dmView : m_dummyF.view;
                VkImageView sc = m_eye[eye].color.view ? m_eye[eye].color.view : m_dummyArr.view;
                VkImageView spv = m_eye[eye].pri.view ? m_eye[eye].pri.view : m_dummyF.view;
                VkDescriptorImageInfo imgs[9] = {
                    {m_nearest, m_tileAtlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_tileMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_tileAttr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_ayx.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_spriteAtlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, dm, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_linear, sc, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, spv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_repeatNearest, m_penBanks.view ? m_penBanks.view : m_dummyU16.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
                for (uint32_t b = 0; b < 9; ++b) {
                    VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    x.dstSet = S.set[k]; x.dstBinding = 8 + b; x.descriptorCount = 1;
                    x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; x.pImageInfo = &imgs[b];
                    w.push_back(x);
                }
                vkUpdateDescriptorSets(m_dev, uint32_t(w.size()), w.data(), 0, nullptr);
            }
        }
        m_setsDirty = false;
    }

    VkDevice m_dev = VK_NULL_HANDLE;
    const MemoryAllocator* m_alloc = nullptr;
    VkFormat m_eyeFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_samples = VK_SAMPLE_COUNT_1_BIT;
    bool m_ready = false, m_assetsReady = false, m_setsDirty = true, m_havePrepared = false, m_dummyPending = false;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkRenderPass m_rpDepth = VK_NULL_HANDLE, m_rpScene = VK_NULL_HANDLE, m_rpComp = VK_NULL_HANDLE;
    VkPipeline m_pInit = VK_NULL_HANDLE, m_pScene3D = VK_NULL_HANDLE, m_pP3 = VK_NULL_HANDLE, m_pP3NoAa = VK_NULL_HANDLE, m_pSceneHud = VK_NULL_HANDLE, m_pDepth = VK_NULL_HANDLE, m_pComp = VK_NULL_HANDLE;
    VkSampler m_nearest = VK_NULL_HANDLE, m_linear = VK_NULL_HANDLE;
    Img m_tileAtlas, m_tileMap, m_tileAttr, m_ayx, m_spriteAtlas, m_dummyU, m_dummyF, m_dummyArr, m_dummyU16, m_dmDepth, m_penBanks;
    std::vector<BufferAndMemory> m_stagings;
    int m_spriteW = 1, m_spriteH = 1, m_spritesPerRow = 1;
    Slot m_slots[kFrames];
    int m_slot = 0;
    EyeTarget m_eye[3];   // left eye, right eye, flat (board projection)
    Img m_flatOut; VkFramebuffer m_flatOutFb = VK_NULL_HANDLE;
    std::vector<CompFb> m_compFbs;
    VkImage m_dmImage = VK_NULL_HANDLE;
    VkDeviceMemory m_dmMem = VK_NULL_HANDLE;
    VkImageView m_dmView = VK_NULL_HANDLE;
    VkFramebuffer m_dmFb = VK_NULL_HANDLE;
    uint32_t m_dmW = 0, m_dmH = 0;
    std::vector<float> m_depthCpu;
    uint32_t m_depthCpuW = 0, m_depthCpuH = 0;
    std::vector<float> m_vertexData, m_primData;
    std::vector<uint32_t> m_indexData;
    std::vector<Run> m_runs;
    std::unordered_map<uint32_t, GroupAcc> m_groupCentre;
    bool m_fogBgValid = false;
    float m_fogBg[3] = {0, 0, 0};
    float m_lastZoom = 0.0f;
    unsigned m_lastPrims = 0, m_lastIndices = 0;
    int m_textW = 640, m_textH = 480;
    tcvr_scene_frame m_frame{};
    Settings m_settings;
};

}  // namespace arcadexr::vulkan
