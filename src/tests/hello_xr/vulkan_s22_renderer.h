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
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "vulkan_utils.h"
#include "logger.h"
#include "common.h"
#include "../../../../../third_party/mame/src/mame/tcvr_scene.h"
#include "s22_quad_vert_spv.h"
#include "s22_init_frag_spv.h"
#include "s22_scene_vert_spv.h"
#include "s22_scene_frag_spv.h"
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
        float depthBias = 4e-8f;
        int voidMode = 0;              // 0 game bg, 1 fog colour (slate if black), 2 fixed
        unsigned voidRGB[3] = {28, 30, 38};
        int texSamples = 4;
        float spriteMinDepth = 50.0f;
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

    void Destroy() {
        if (!m_dev) return;
        vkDeviceWaitIdle(m_dev);
        ResetAssets();
        for (auto& f : m_compFbs) { vkDestroyFramebuffer(m_dev, f.fb, nullptr); vkDestroyImageView(m_dev, f.view, nullptr); }
        m_compFbs.clear();
        for (int e = 0; e < 2; ++e) DestroyEyeTarget(m_eye[e]);
        DestroyDepthMap();
        for (int s = 0; s < kFrames; ++s) DestroySlot(s);
        auto dp = [&](VkPipeline& p) { if (p) vkDestroyPipeline(m_dev, p, nullptr); p = VK_NULL_HANDLE; };
        dp(m_pInit); dp(m_pScene3D); dp(m_pSceneHud); dp(m_pDepth); dp(m_pComp);
        auto drp = [&](VkRenderPass& r) { if (r) vkDestroyRenderPass(m_dev, r, nullptr); r = VK_NULL_HANDLE; };
        drp(m_rpDepth); drp(m_rpScene); drp(m_rpComp);
        if (m_layout) vkDestroyPipelineLayout(m_dev, m_layout, nullptr);
        if (m_setLayout) vkDestroyDescriptorSetLayout(m_dev, m_setLayout, nullptr);
        if (m_pool) vkDestroyDescriptorPool(m_dev, m_pool, nullptr);
        if (m_nearest) vkDestroySampler(m_dev, m_nearest, nullptr);
        if (m_linear) vkDestroySampler(m_dev, m_linear, nullptr);
        DestroyImage(m_dummyU); DestroyImage(m_dummyF);
        m_dev = VK_NULL_HANDLE; m_ready = false;
    }

    // ---- static assets (tile store, sprites): once per game --------------------------------------
    void ResetAssets() {
        if (!m_dev) return;
        if (m_assetsReady) vkDeviceWaitIdle(m_dev);
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
        m_assetsReady = true;
        m_setsDirty = true;
        Log::Write(Log::Level::Info, Fmt("TCVR_S22VK assets: tiles %u bytes, sprites %u (%ux%u)", a.tiledata_bytes, a.sprite_count,
                                         a.sprite_width, a.sprite_height));
        return true;
    }

    // ---- per frame (view 0, after the slot's fence) --------------------------------------------
    // CPU port of SceneRenderer::PrepareFrame, written into this slot's buffers.
    bool PrepareFrame(int slot, const tcvr_scene_frame& f) {
        if (!m_ready || !m_assetsReady) return false;
        m_slot = slot & 1;
        Slot& S = m_slots[m_slot];
        m_vertexData.clear(); m_indexData.clear(); m_primData.clear(); m_runs.clear();
        std::uint32_t lastPolyIndex = 0; (void)lastPolyIndex;
        bool anyPoly = false;
        m_fogBgValid = false;
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.kind == 0 && pr.direct == 0) anyPoly = true;
            if (!m_fogBgValid && pr.fog_mode == 2) { m_fogBg[0] = pr.fog_r; m_fogBg[1] = pr.fog_g; m_fogBg[2] = pr.fog_b; m_fogBgValid = true; }
        }
        m_groupCentre.clear();
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.kind != 1 || pr.vertex_count < 3) continue;
            auto& g = m_groupCentre[uint32_t(pr.cz_bank)];
            for (uint32_t i = 0; i < pr.vertex_count; ++i) { g.x += f.vertices[pr.first_vertex + i].x; g.y += f.vertices[pr.first_vertex + i].y; }
            g.n += float(pr.vertex_count);
        }
        float neighbourZoom = 1.0f;
        for (uint32_t p = 0; p < f.prim_count; ++p) {
            const tcvr_scene_prim& pr = f.prims[p];
            if (pr.vertex_count < 3 || pr.first_vertex + pr.vertex_count > f.vertex_count) continue;
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
                const tcvr_scene_vertex& v = f.vertices[pr.first_vertex + i];
                const float row[8] = {v.x, v.y, v.z, v.u, v.v, v.bri, pr.zoom, float(p)};
                m_vertexData.insert(m_vertexData.end(), row, row + 8);
            }
            const bool hud = pr.kind != 0 || pr.direct != 0;
            if (m_runs.empty() || m_runs.back().hud != hud) {
                if (!m_runs.empty()) m_runs.back().count = uint32_t(m_indexData.size()) - m_runs.back().first;
                m_runs.push_back({uint32_t(m_indexData.size()), 0, hud});
            }
            for (uint32_t i = 1; i + 1 < pr.vertex_count; ++i) {
                m_indexData.push_back(base); m_indexData.push_back(base + i); m_indexData.push_back(base + i + 1);
            }
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
        }
        if (!m_runs.empty()) m_runs.back().count = uint32_t(m_indexData.size()) - m_runs.back().first;
        m_lastPrims = uint32_t(m_primData.size() / 64);
        m_lastIndices = uint32_t(m_indexData.size());
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
        VkClearValue cv[2]{}; cv[1].depthStencil = {1.0f, 0};
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
        cv[2].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_rpScene; bi.framebuffer = T.fb; bi.renderArea = {{0, 0}, {rw, rh}};
        bi.clearValueCount = IsMsaa() ? 5 : 3; bi.pClearValues = cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        SetVp(cmd, rw, rh);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[eye], 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pInit);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        if (m_lastIndices > 0) {
            BindGeometry(cmd, S);
            for (const Run& r : m_runs) {
                if (!r.count) continue;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (r.hud || !st.depthTest) ? m_pSceneHud : m_pScene3D);
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
        std::memcpy(S.uboMap[3 + eye], &uc, sizeof(uc));
        VkRenderPassBeginInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        ci.renderPass = m_rpComp; ci.framebuffer = fb; ci.renderArea = area; ci.clearValueCount = 0;
        vkCmdBeginRenderPass(cmd, &ci, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{float(area.offset.x), float(area.offset.y), float(area.extent.width), float(area.extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &area);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pComp);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &S.set[3 + eye], 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        return true;
    }

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
    struct GroupAcc { float x = 0, y = 0, n = 0; };
    struct Slot {
        BufferAndMemory vbo, ibo, prim, pens, cz, spot, gamma, text, pri, ubo[5], depthRead;
        void *vboMap = nullptr, *iboMap = nullptr, *primMap = nullptr, *pensMap = nullptr, *czMap = nullptr, *spotMap = nullptr,
             *gammaMap = nullptr, *textMap = nullptr, *priMap = nullptr, *uboMap[5] = {}, *depthReadMap = nullptr;
        VkDescriptorSet set[5] = {};   // eye0, eye1, depth map, composite eye0, composite eye1
        bool depthReadValid = false;
    };
    struct EyeTarget {
        uint32_t w = 0, h = 0;
        Img msColor, msPri, depth, color, pri;
        VkFramebuffer fb = VK_NULL_HANDLE;
    };
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
        b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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
        for (uint32_t i = 8; i <= 15; ++i) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = uint32_t(b.size()); li.pBindings = b.data();
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_dev, &li, nullptr, &m_setLayout));
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_setLayout;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_dev, &pli, nullptr, &m_layout));
        VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 96},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 96}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 12; pi.poolSizeCount = 3; pi.pPoolSizes = ps;
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
        dsTest.depthTestEnable = VK_TRUE; dsTest.depthWriteEnable = VK_TRUE; dsTest.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineDepthStencilStateCreateInfo dsLess = dsTest; dsLess.depthCompareOp = VK_COMPARE_OP_LESS;
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
        m_dummyU = NewImage(VK_FORMAT_R8_UINT, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_dummyF = NewImage(VK_FORMAT_R8G8B8A8_UNORM, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        m_dummyPending = true;
    }

    Img NewImage(VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkSampleCountFlagBits s = VK_SAMPLE_COUNT_1_BIT,
                 bool transient = false, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
        Img im;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = s; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage | (transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0);
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        XRC_CHECK_THROW_VKCMD(vkCreateImage(m_dev, &ii, nullptr, &im.image));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(m_dev, im.image, &req);
        bool done = false;
        if (transient) { try { m_alloc->Allocate(req, &im.mem, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT); done = true; } catch (...) {} }
        if (!done) m_alloc->Allocate(req, &im.mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_dev, im.image, im.mem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt; vi.subresourceRange = {aspect, 0, 1, 0, 1};
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
            for (Img* d : {&m_dummyU, &m_dummyF}) {
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
        for (int i = 0; i < 5; ++i) mk(S.ubo[i], &S.uboMap[i], sizeof(Ubo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        mk(S.depthRead, &S.depthReadMap, 1024 * 1024 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkDescriptorSetLayout ls[5] = {m_setLayout, m_setLayout, m_setLayout, m_setLayout, m_setLayout};
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = m_pool; ai.descriptorSetCount = 5; ai.pSetLayouts = ls;
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
        T.color = NewImage(VK_FORMAT_R8G8B8A8_UNORM, w, h, att | VK_IMAGE_USAGE_SAMPLED_BIT);
        T.pri = NewImage(VK_FORMAT_R8_UNORM, w, h, att | VK_IMAGE_USAGE_SAMPLED_BIT);
        T.depth = NewImage(VK_FORMAT_D32_SFLOAT, w, h, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, m_samples, true, VK_IMAGE_ASPECT_DEPTH_BIT);
        std::vector<VkImageView> v;
        if (IsMsaa()) {
            T.msColor = NewImage(VK_FORMAT_R8G8B8A8_UNORM, w, h, att, m_samples, true);
            T.msPri = NewImage(VK_FORMAT_R8_UNORM, w, h, att, m_samples, true);
            v = {T.msColor.view, T.msPri.view, T.depth.view, T.color.view, T.pri.view};
        } else {
            v = {T.color.view, T.pri.view, T.depth.view};
        }
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = m_rpScene; fi.attachmentCount = uint32_t(v.size()); fi.pAttachments = v.data();
        fi.width = w; fi.height = h; fi.layers = 1;
        XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &T.fb));
        Log::Write(Log::Level::Info, Fmt("TCVR_S22VK eye target %ux%u MSAA x%d", w, h, int(m_samples)));
    }
    void DestroyEyeTarget(EyeTarget& T) {
        if (T.fb) vkDestroyFramebuffer(m_dev, T.fb, nullptr);
        DestroyImage(T.msColor); DestroyImage(T.msPri); DestroyImage(T.depth); DestroyImage(T.color); DestroyImage(T.pri);
        T = EyeTarget{};
    }

    void UpdateSetsIfDirty() {
        if (!m_setsDirty || !m_assetsReady) return;
        vkDeviceWaitIdle(m_dev);   // rare: new assets or new targets
        for (int s = 0; s < kFrames; ++s) {
            Slot& S = m_slots[s];
            for (int k = 0; k < 5; ++k) {
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
                const int eye = (k == 3) ? 0 : (k == 4 ? 1 : 0);
                VkImageView dm = m_dmView ? m_dmView : m_dummyF.view;
                VkImageView sc = m_eye[eye].color.view ? m_eye[eye].color.view : m_dummyF.view;
                VkImageView spv = m_eye[eye].pri.view ? m_eye[eye].pri.view : m_dummyF.view;
                VkDescriptorImageInfo imgs[8] = {
                    {m_nearest, m_tileAtlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_tileMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_tileAttr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_ayx.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, m_spriteAtlas.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, dm, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_linear, sc, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                    {m_nearest, spv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
                for (uint32_t b = 0; b < 8; ++b) {
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
    VkPipeline m_pInit = VK_NULL_HANDLE, m_pScene3D = VK_NULL_HANDLE, m_pSceneHud = VK_NULL_HANDLE, m_pDepth = VK_NULL_HANDLE, m_pComp = VK_NULL_HANDLE;
    VkSampler m_nearest = VK_NULL_HANDLE, m_linear = VK_NULL_HANDLE;
    Img m_tileAtlas, m_tileMap, m_tileAttr, m_ayx, m_spriteAtlas, m_dummyU, m_dummyF, m_dmDepth;
    std::vector<BufferAndMemory> m_stagings;
    int m_spriteW = 1, m_spriteH = 1, m_spritesPerRow = 1;
    Slot m_slots[kFrames];
    int m_slot = 0;
    EyeTarget m_eye[2];
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
