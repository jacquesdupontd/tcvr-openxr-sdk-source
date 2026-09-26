#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <array>
#include <android/log.h>

#include "vulkan_utils.h"
#include "gpu_renderer.h"
#include "settings.h"
#include "game_profile.h"
#include "virtual_screen.h"
#include "m2_pipeline_types.h"
#include "vulkan_m2_regions.h"

#include "m2_vert_spv.h"
#include "m2_vert_appsw_spv.h"
#include "m2_vert_appswmv_spv.h"
#include "appsw_mv_frag_spv.h"
#include "m2_frag_spv.h"
#include "m2_frag_nd_spv.h"
#include "m2_frag_cutd_spv.h"
#include "m2_frag_lean_spv.h"
#include "m2_frag_leancut_spv.h"
#include "m2_frag_cutc_spv.h"
#include "quad_vert_spv.h"
#include "quad_far_vert_spv.h"
#include "void_frag_spv.h"
#include "plane_frag_spv.h"

namespace arcadexr::vulkan {

static const char* kM2LogTag = "TCVR_M2VK";

static inline bool InvertMatrix4x4(const float* m, float* out) {
    float inv[16];
    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
    const float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (std::abs(det) < 1e-12f) return false;
    const float invDet = 1.0f / det;
    for (int i = 0; i < 16; i++) out[i] = inv[i] * invDet;
    return true;
}

class VulkanModel2Renderer {
public:
    VulkanModel2Renderer() = default;
    ~VulkanModel2Renderer() { Cleanup(); }

    // The immersive renderer owns its render pass: MSAA colour + depth that live only in the
    // tile memory (transient, lazily allocated, never stored), resolved on-chip into the
    // swapchain image. On the Adreno this is what makes 4x MSAA nearly free.
    bool Initialize(VkDevice device, const MemoryAllocator* allocator, VkFormat colorFormat, uint32_t samples, bool useFdm = false) {
        if (m_initialized) return true;
        m_vkDevice = device;
        m_memAllocator = allocator;
        m_colorFormat = colorFormat;
        m_useFdm = useFdm;
        m_samples = (samples >= 4) ? VK_SAMPLE_COUNT_4_BIT : (samples >= 2 ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT);
        CreateRenderPass();
        VkRenderPass renderPass = m_pass;

        Log::Write(Log::Level::Info, "TCVR_M2VK: Initializing Model 2 native Vulkan immersive renderer");

        // 1. Create Descriptor Pool
        std::array<VkDescriptorPoolSize, 5> poolSizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 96},   // + binding 16 (smooth motion) per set
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 80},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8 * M2RegionTextures::kMaxSlots + 16},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 80}
        }};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 24;
        poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &m_descriptorPool));

        // 2. Create Descriptor Set Layouts
        // Model 2 layout: binding 0: UBO, binding 1..7: SSBOs, binding 8..9: samplers
        std::vector<VkDescriptorSetLayoutBinding> m2Bindings;
        // Binding 0: UBO
        VkDescriptorSetLayoutBinding uboBind{};
        uboBind.binding = 0;
        uboBind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBind.descriptorCount = 1;
        uboBind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        m2Bindings.push_back(uboBind);

        // Bindings 1..7: SSBOs
        for (uint32_t b = 1; b <= 7; ++b) {
            VkDescriptorSetLayoutBinding ssboBind{};
            ssboBind.binding = b;
            ssboBind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            ssboBind.descriptorCount = 1;
            ssboBind.stageFlags = (b == 1 || b == 2) ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                                          : VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(ssboBind);
        }
        {   // Binding 16: previous positions of the vertices (smooth motion), vertex stage
            VkDescriptorSetLayoutBinding pb{};
            pb.binding = 16; pb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            pb.descriptorCount = 1; pb.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            m2Bindings.push_back(pb);
        }
        // Bindings 8, 9: Sheet texture samplers
        for (uint32_t b = 8; b <= 11; ++b) {
            VkDescriptorSetLayoutBinding texBind{};
            texBind.binding = b;
            texBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            texBind.descriptorCount = 1;
            texBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(texBind);
        }

        // Binding 12: every texture region as its own image (texture2D[kMaxSlots]); 13: 4 samplers
        // (repeat / mirrored per axis, hardware anisotropy). Indexed per polygon (nonuniformEXT).
        {
            VkDescriptorSetLayoutBinding rb{};
            rb.binding = 12; rb.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            rb.descriptorCount = M2RegionTextures::kMaxSlots; rb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(rb);
            VkDescriptorSetLayoutBinding sb{};
            sb.binding = 13; sb.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            sb.descriptorCount = 8; sb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(sb);
            VkDescriptorSetLayoutBinding ab{};
            ab.binding = 15; ab.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ab.descriptorCount = 1; ab.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(ab);
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = 14; lb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            lb.descriptorCount = 1; lb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(lb);
        }
        VkDescriptorSetLayoutCreateInfo m2LayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        m2LayoutInfo.bindingCount = (uint32_t)m2Bindings.size();
        m2LayoutInfo.pBindings = m2Bindings.data();
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_vkDevice, &m2LayoutInfo, nullptr, &m_m2DescLayout));

        // Plane layer descriptor set layout: binding 0: combined sampler2D
        VkDescriptorSetLayoutBinding planeBind{};
        planeBind.binding = 0;
        planeBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        planeBind.descriptorCount = 1;
        planeBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo planeLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        planeLayoutInfo.bindingCount = 1;
        planeLayoutInfo.pBindings = &planeBind;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_vkDevice, &planeLayoutInfo, nullptr, &m_planeDescLayout));

        // 3. Create Pipeline Layouts
        // Model 2 pipeline layout
        VkPipelineLayoutCreateInfo m2PipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        m2PipeLayoutInfo.setLayoutCount = 1;
        m2PipeLayoutInfo.pSetLayouts = &m_m2DescLayout;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_vkDevice, &m2PipeLayoutInfo, nullptr, &m_m2PipelineLayout));

        // Void pipeline layout (push constants only, 112 bytes)
        VkPushConstantRange voidPcr{};
        voidPcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        voidPcr.offset = 0;
        voidPcr.size = sizeof(VoidPushConstants);
        VkPipelineLayoutCreateInfo voidPipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        voidPipeLayoutInfo.pushConstantRangeCount = 1;
        voidPipeLayoutInfo.pPushConstantRanges = &voidPcr;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_vkDevice, &voidPipeLayoutInfo, nullptr, &m_voidPipelineLayout));

        // Plane pipeline layout (1 descriptor set, push constants 80 bytes)
        VkPushConstantRange planePcr{};
        planePcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        planePcr.offset = 0;
        planePcr.size = sizeof(PlanePushConstants);
        VkPipelineLayoutCreateInfo planePipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        planePipeLayoutInfo.setLayoutCount = 1;
        planePipeLayoutInfo.pSetLayouts = &m_planeDescLayout;
        planePipeLayoutInfo.pushConstantRangeCount = 1;
        planePipeLayoutInfo.pPushConstantRanges = &planePcr;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_vkDevice, &planePipeLayoutInfo, nullptr, &m_planePipelineLayout));

        // 4. Create Shader Modules
        m_quadVertModule = CreateShaderModule(c_quad_vertSpv, sizeof(c_quad_vertSpv));
        m_quadFarVertModule = CreateShaderModule(c_quad_far_vertSpv, sizeof(c_quad_far_vertSpv));
        m_voidFragModule = CreateShaderModule(c_void_fragSpv, sizeof(c_void_fragSpv));
        m_planeFragModule = CreateShaderModule(c_plane_fragSpv, sizeof(c_plane_fragSpv));
        m_m2VertModule = CreateShaderModule(m2_vert_spv, sizeof(m2_vert_spv));
        m_m2FragModule = CreateShaderModule(c_m2_fragSpv, sizeof(c_m2_fragSpv));
        m_m2FragNdModule = CreateShaderModule(c_m2_fragNoDiscardSpv, sizeof(c_m2_fragNoDiscardSpv));
        m_m2FragCutDModule = CreateShaderModule(c_m2_fragCutDepthSpv, sizeof(c_m2_fragCutDepthSpv));
        m_m2FragLeanModule = CreateShaderModule(c_m2_fragLeanSpv, sizeof(c_m2_fragLeanSpv));
        m_m2FragLeanCutModule = CreateShaderModule(c_m2_fragLeanCutSpv, sizeof(c_m2_fragLeanCutSpv));
        m_m2FragCutCModule = CreateShaderModule(c_m2_fragCutColorSpv, sizeof(c_m2_fragCutColorSpv));

        // 5. Create Pipelines
        CreatePipelines(renderPass);

        // 6. Allocate Samplers
        // Sheet sampler: LINEAR (the hardware bilinear of filter mode 5; texelFetch ignores it), Clamp
        VkSamplerCreateInfo sheetSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sheetSamplerInfo.magFilter = VK_FILTER_LINEAR;
        sheetSamplerInfo.minFilter = VK_FILTER_LINEAR;
        sheetSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sheetSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sheetSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sheetSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_vkDevice, &sheetSamplerInfo, nullptr, &m_sheetSampler));

        // Plane samplers: Linear, Clamp (front) / Repeat U (back)
        VkSamplerCreateInfo planeSamplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        planeSamplerInfo.magFilter = VK_FILTER_LINEAR;
        planeSamplerInfo.minFilter = VK_FILTER_LINEAR;
        planeSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        planeSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        planeSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        planeSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_vkDevice, &planeSamplerInfo, nullptr, &m_layerSampler[0]));

        planeSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_vkDevice, &planeSamplerInfo, nullptr, &m_layerSampler[1]));

        // 7. Allocate Host-Visible Buffers and Texture Storage
        AllocateBuffers();
        {
            const float an = g_m2SamplerAnisotropy ? float(std::max(1, std::min(16, arcadexr::config::GetInt("m2.hwAniso", 4)))) : 1.0f;
            m_regions.Init(m_vkDevice, m_memAllocator, an);
            m_useRegions = g_m2DescriptorIndexing;
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK region textures: %s, hw anisotropy x%.0f",
                                             m_useRegions ? "ON (descriptor indexing)" : "OFF (no indexing)", an));
        }
        AllocateTextures();

        m_initialized = true;
        Log::Write(Log::Level::Info, "TCVR_M2VK: Native Vulkan immersive renderer ready!");
        return true;
    }

    // Legacy single call: build then commit.
    void PrepareFrame(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        BuildFrame(frame);
        CommitFrame(frame, cmd);
    }

    // CPU half of the frame preparation: reads the emulator's scene and builds the CPU-side
    // arrays (sort, triangulation, main view, texture region choice). Touches NO buffer the GPU
    // may still be reading, so the plugin runs it BEFORE waiting for the previous frame's fence:
    // it overlaps the GPU instead of leaving it idle (measured 22/09: wait 9 ms + prepare 3.7 ms
    // back to back -> a 14 ms period, 71 fps, with the GPU only 73% busy).
    void BuildFrame(const tcvr_m2_frame& frame) {
        m_built = false;
        if (!m_initialized || frame.geometry_unchanged != 0u) return;
        const bool haveClipped = frame.prim_count != 0 && frame.vertex_count != 0;
        const bool haveRaw = frame.raw_prim_count != 0 && frame.raw_vertex_count != 0;
        if (!haveClipped && !haveRaw) return;
        m_built = true;
        m_overlayOn = arcadexr::profiles::GetInt("immersive.screenOverlay", arcadexr::profiles::IsDriving() ? 0 : 1) != 0;

        // Fan-triangulate and sort primitives
        const tcvr_m2_prim* kp = frame.raw_prim_count ? frame.raw_prims : frame.prims;
        const std::uint32_t kn = frame.raw_prim_count ? frame.raw_prim_count : frame.prim_count;
        std::int64_t bestArea = -1; std::uint32_t bestPrims = 0;
        std::int32_t mainL = 0, mainT = 0, mainR = 0, mainB = 0, mainCx = 0, mainCy = 0;
        {
            struct Key { std::int32_t l, t, r, b, cx, cy; std::uint32_t prims; };
            Key keys[8]; unsigned n = 0;
            for (std::uint32_t i = 0; i < kn; i++) {
                const tcvr_m2_prim& p = kp[i];
                unsigned k = 0;
                for (; k < n; ++k)
                    if (keys[k].l == p.clip_l && keys[k].t == p.clip_t && keys[k].r == p.clip_r &&
                        keys[k].b == p.clip_b && keys[k].cx == p.center_x && keys[k].cy == p.center_y)
                        break;
                if (k == n && n < 8) keys[n++] = {p.clip_l, p.clip_t, p.clip_r, p.clip_b, p.center_x, p.center_y, 0};
                if (k < n) keys[k].prims++;
            }
            if (arcadexr::config::GetInt("m2.viewDiag", 0) != 0 && (++m_viewDiagTick % 120u) == 0u)
                for (unsigned k = 0; k < n; ++k)
                    Log::Write(Log::Level::Info, Fmt("TCVR_VIEWS %u/%u clip=%d,%d..%d,%d centre screen=%d,%d prims=%u", k, n,
                        keys[k].l, keys[k].t, keys[k].r, keys[k].b, frame.crtc_xoffset + keys[k].cx, (384 - keys[k].cy) + frame.crtc_yoffset, keys[k].prims));
            m_mainLike.clear();
            const std::int64_t halfScreen = std::int64_t(496) * std::int64_t(384) / 2;
            for (unsigned k = 0; k < n; ++k) {
                const std::int64_t area = std::int64_t(keys[k].r - keys[k].l + 1) * std::int64_t(keys[k].b - keys[k].t + 1);
                const int sx = frame.crtc_xoffset + keys[k].cx;
                const int sy = (384 - keys[k].cy) + frame.crtc_yoffset;
                const bool centred = std::abs(sx - 496 / 2) <= 496 / 8 && std::abs(sy - 384 / 2) <= 384 / 8;
                if (area < halfScreen || !centred) continue;
                if (m_mainLike.size() < 8) m_mainLike.push_back({keys[k].l, keys[k].t, keys[k].r, keys[k].b, keys[k].cx, keys[k].cy});
                if (area > bestArea || (area == bestArea && keys[k].prims > bestPrims)) {
                    bestArea = area; bestPrims = keys[k].prims;
                    mainL = keys[k].l; mainT = keys[k].t; mainR = keys[k].r; mainB = keys[k].b;
                    mainCx = keys[k].cx; mainCy = keys[k].cy;
                }
            }
        }
        m_haveMainView = bestArea > 0;
        if (!m_haveMainView) { mainL = mainT = mainR = mainB = -1; mainCx = mainCy = -100000; }

        m_horizonGeo = -1.0f;
        m_haveHorizon = false;
        m_crtc[0] = float(frame.crtc_xoffset); m_crtc[1] = float(frame.crtc_yoffset);
        // The geometry engine's real focal length (the GLES renderer reads it; the first Vulkan port kept 512).
        if (frame.focus_x > 1.0f && frame.focus_y > 1.0f) {
            if (std::fabs(frame.focus_x - m_m2FocusX) > 0.5f || std::fabs(frame.focus_y - m_m2FocusY) > 0.5f)
                Log::Write(Log::Level::Info, Fmt("TCVR_M2VK focus %.1f x %.1f (was %.1f x %.1f)", frame.focus_x, frame.focus_y, m_m2FocusX, m_m2FocusY));
            m_m2FocusX = frame.focus_x; m_m2FocusY = frame.focus_y;
        }
        m_directColour = kn > 0 && (kp[0].rgb & 0x1000000u) != 0u;   // Model 1 scene
        if (m_haveMainView) {
            // "Far" was z >= 1000, Model 2 units. Model 1 units differ: its horizon appeared and vanished from frame
            // to frame and the pitch derived from it shook the whole world (Guillaume, 24/09, Virtua Racing). On a
            // Model 1 scene "far" is relative to the view's depth, and a view without depth (a menu with 3D icons)
            // has no horizon at all. Model 2 keeps its validated threshold.
            float farZ = 1000.0f;
            if (m_directColour && frame.raw_prim_count) {
                std::vector<float> zs;
                for (std::uint32_t i = 0; i < kn; i++) {
                    const tcvr_m2_prim& p = kp[i];
                    if (p.clip_l != mainL || p.clip_t != mainT || p.clip_r != mainR || p.clip_b != mainB ||
                        p.center_x != mainCx || p.center_y != mainCy) continue;
                    for (std::uint32_t v = 0; v < p.vertex_count; v++) {
                        const std::uint32_t index = p.first_vertex + v;
                        if (index < frame.raw_vertex_count && frame.raw_vertices[index].z > 0.0f) zs.push_back(frame.raw_vertices[index].z);
                    }
                }
                if (zs.size() >= 32) {
                    std::nth_element(zs.begin(), zs.begin() + zs.size() / 2, zs.end());
                    const float med = zs[zs.size() / 2];
                    const float zmax = *std::max_element(zs.begin(), zs.end());
                    farZ = (zmax > 3.0f * med) ? 0.4f * zmax : 1e30f;
                    // Pop-in fade depth: where the scenery ends (99th percentile). The 90th with a fade from 70 % washed the
                    // far forest into the sky and read as a SHORT draw distance (24/09); now only the last metres fade.
                    std::nth_element(zs.begin(), zs.begin() + (zs.size() * 99) / 100, zs.end());
                    const float z90 = zs[(zs.size() * 99) / 100];
                    m_zMaxSmooth = (m_zMaxSmooth > 0.0f) ? m_zMaxSmooth + (z90 - m_zMaxSmooth) * 0.05f : z90;
                } else {
                    farZ = 1e30f;
                }
            }
            std::vector<float> farRows;
            m_mainZMax = 0.0f;
            for (std::uint32_t i = 0; i < kn; i++) {
                const tcvr_m2_prim& p = kp[i];
                if (p.clip_l != mainL || p.clip_t != mainT || p.clip_r != mainR || p.clip_b != mainB ||
                    p.center_x != mainCx || p.center_y != mainCy) continue;
                for (std::uint32_t v = 0; v < p.vertex_count; v++) {
                    const std::uint32_t index = p.first_vertex + v;
                    if (frame.raw_prim_count) {
                        if (index >= frame.raw_vertex_count) break;
                        const tcvr_m2_raw_vertex& rv = frame.raw_vertices[index];
                        if (rv.z > m_mainZMax) m_mainZMax = rv.z;
                        if (rv.z >= farZ) farRows.push_back((384.0f - float(p.center_y)) + float(frame.crtc_yoffset) - rv.y / rv.z);
                    }
                }
            }
            if (farRows.size() >= (m_directColour ? 8u : 32u)) {
                std::nth_element(farRows.begin(), farRows.begin() + farRows.size() / 2, farRows.end());
                m_horizonGeo = farRows[farRows.size() / 2];
                m_haveHorizon = true;
            }
        }

        {   // Model 1 menu screen: no horizon AND few polygons (a course select has a few 3D icons, a race thousands);
            // held for half a second either way -- a race that loses its horizon for a moment is not a menu.
            // Model 2 too (26/09, Sega Rally's car select: a full-screen centred view made it a "scene": its 2D page
            // tiled to infinity as a sky and the boxes floated in front). A horizon row can be negative in a race
            // (-23.7 measured): "no horizon" is the flag, never the sign.
            const bool looksMenu = !m_haveHorizon && (m_directColour ? kn < 400u : true);
            m_menuFrames = looksMenu ? std::min(m_menuFrames + 1, 1000) : std::max(m_menuFrames - 1, -1000);
            if (m_menuFrames >= 30) m_isMenuM1 = true;
            if (m_menuFrames <= -30) m_isMenuM1 = false;
            // Model 2: a horizon is proof of a scene (every menu measured has none: depth 28 against 200000 in a race),
            // so leave at once -- held 30 arcade frames, the start of a race stayed flat for 2 s while the course
            // loaded (few frames published, 26/09).
            if (!m_directColour && m_haveHorizon) { m_isMenuM1 = false; m_menuFrames = 0; }
            if (!looksMenu && m_menuFrames > 0) m_menuFrames = 0;
            if (looksMenu && m_menuFrames < 0) m_menuFrames = 0;
        }
        m_mainClip[0] = mainL; m_mainClip[1] = mainT; m_mainClip[2] = mainR; m_mainClip[3] = mainB;
        m_mainCenter[0] = mainCx; m_mainCenter[1] = mainCy;

        // Build raw pre-clip geometry
        m_primDiagOn = false;
        if (arcadexr::config::GetInt("m2.primDiag", 0) != 0 && (++m_primDiagTick % 60u) == 0u) {
            m_primDiagOn = true;
            m_primDiagX = float(arcadexr::config::GetInt("m2.primDiagX", 248));
            m_primDiagY = float(arcadexr::config::GetInt("m2.primDiagY", 330));
            __android_log_print(ANDROID_LOG_INFO, "TCVR_PRIM", "---- pixel (%.0f, %.0f) seq=%llu", m_primDiagX, m_primDiagY, (unsigned long long)frame.sequence);
            m_clN = 0; m_clNotMain = 0; m_clBox[0] = m_clBox[2] = 1e9f; m_clBox[1] = m_clBox[3] = -1e9f;
            m_clBk[0] = 0xffffffffu; m_clBk[1] = 0u; for (auto& b : m_clBand) b = 0; for (auto& b : m_clPass) b = 0;
        }
        if (frame.raw_prim_count > 0 && frame.raw_vertex_count > 0) {
            const std::uint32_t n = std::min<std::uint32_t>(frame.raw_prim_count, 0xffffu);
            m_rawKeys.resize(n);
            for (std::uint32_t i = 0; i < n; i++) {
                // The board's own order (model2_3d_frame_end): windows from the last to the first, then depth buckets
                // near to far, then within a bucket the last polygon pushed first; the first written on a pixel stays.
                // The window was ignored until 25/09: Top Skater's ramp covered the skater the cabinet draws in front.
                const std::uint64_t win = std::min<std::uint32_t>(frame.raw_prims[i].window, 0xffu);
                m_rawKeys[i] = ((0xffull - win) << 32) | (std::uint64_t(frame.raw_prims[i].zsort & 0xffffu) << 16) | (0xffffu - i);
            }
            std::sort(m_rawKeys.begin(), m_rawKeys.end());

            m_rawPrims.resize(n);
            m_primSlot.assign(n, 0xffffffffu);
            m_primLayer.assign(n, 0xffffffffu);
            std::size_t vcount = 0, icount = 0;
            for (std::uint32_t k = 0; k < n; k++) {
                const tcvr_m2_prim& p = frame.raw_prims[0xffffu - uint32_t(m_rawKeys[k] & 0xffffu)];
                const std::uint32_t vc = std::min<std::uint32_t>(p.vertex_count, frame.raw_vertex_count - std::min(p.first_vertex, frame.raw_vertex_count));
                vcount += vc; if (vc >= 3) icount += (vc - 2) * 3;
            }

            m_rawVerts.resize(vcount * 5);
            m_rawPrimOfVertex.resize(vcount);
            m_rawIdx.resize(icount);
            std::vector<std::uint32_t> rawGlassIdx, rawCutIdx;
            rawGlassIdx.reserve(icount / 8);
            rawCutIdx.reserve(icount / 4);
            std::vector<std::uint32_t> cutSegStart;   // per cut polygon, to emit them FAR to NEAR
            std::vector<std::uint32_t> rawSecIdx;     // secondary views (other cameras, on the screen plane)
            std::vector<std::uint32_t> leanFallbackIdx;   // opaque, textured, no region image nor layer

            std::size_t vo = 0, io = 0;
            std::uint32_t last_zsort = 0xffffffffu;
            std::uint32_t intra_bucket_rank = 0;

            for (std::uint32_t k = 0; k < n; k++) {
                const std::uint32_t cur_zsort = std::uint32_t((m_rawKeys[k] >> 16) & 0xffffu);
                if (cur_zsort == last_zsort) {
                    intra_bucket_rank++;
                } else {
                    last_zsort = cur_zsort;
                    intra_bucket_rank = 0;
                }
                const tcvr_m2_prim& p = frame.raw_prims[0xffffu - uint32_t(m_rawKeys[k] & 0xffffu)];
                tcvr_m2_prim q = p;
                q.first_vertex = std::uint32_t(vo);
                const std::uint32_t vc = std::min<std::uint32_t>(p.vertex_count, frame.raw_vertex_count - std::min(p.first_vertex, frame.raw_vertex_count));
                q.vertex_count = vc;
                q.zsort = intra_bucket_rank;
                // Screen overlay (23/09, Virtua Cop): a polygon facing the camera at one depth that covers the whole
                // arcade frame is a camera-attached effect (fade, hit flash, transition), a full-screen veil on the
                // cabinet. In immersive it floated as a panel with visible edges. Flag it (bit 25 of rgb): the vertex
                // stage draws it straight in screen space, enlarged to cover the whole view.
                if (vc >= 3 && m_overlayOn) {
                    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, zmin = 1e9f, zmax = -1e9f; bool ok = true;
                    for (std::uint32_t v = 0; v < vc && ok; v++) {
                        const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                        if (rv.z <= 1e-3f) { ok = false; break; }
                        const float sx = float(frame.crtc_xoffset + p.center_x) + rv.x / rv.z;
                        const float sy = float((384 - p.center_y) + frame.crtc_yoffset) - rv.y / rv.z;
                        x0 = std::min(x0, sx); x1 = std::max(x1, sx); y0 = std::min(y0, sy); y1 = std::max(y1, sy);
                        zmin = std::min(zmin, rv.z); zmax = std::max(zmax, rv.z);
                    }
                    const float fw = float(p.clip_r - p.clip_l + 1), fh = float(p.clip_b - p.clip_t + 1);
                    const bool facing = ok && zmax - zmin <= 0.01f * zmin;
                    const bool fullW = x0 <= float(p.clip_l) + 0.05f * fw && x1 >= float(p.clip_r) - 0.05f * fw;
                    const bool fullH = y0 <= float(p.clip_t) + 0.05f * fh && y1 >= float(p.clip_b) - 0.05f * fh;
                    // Letterbox strips too (a black band along the top of Virtua Cop's attract floated as a bar):
                    // full width against the top or bottom edge, or full height against a side.
                    const bool band = (fullW && y1 - y0 <= 0.25f * fh && (y0 <= float(p.clip_t) + 0.05f * fh || y1 >= float(p.clip_b) - 0.05f * fh)) ||
                                      (fullH && x1 - x0 <= 0.25f * fw && (x0 <= float(p.clip_l) + 0.05f * fw || x1 >= float(p.clip_r) - 0.05f * fw));
                    // A band is a letterbox only if it is drawn OVER the scene (near the top of the draw order): a black
                    // background plate against the bottom edge caught by the same test blacked out a third of the view.
                    const bool front = k < std::max<std::uint32_t>(8u, n / 20u);
                    if (facing && ((fullW && fullH) || (band && front))) {
                        q.rgb |= 0x2000000u;
                        if (!(fullW && fullH)) q.rgb |= 0x4000000u;   // letterbox band: dropped in immersive
                        if (arcadexr::config::GetInt("m2.overlayDiag", 0) != 0)
                            Log::Write(Log::Level::Info, Fmt("TCVR_OVERLAYFLAG rank=%u/%u %s box=%.0f,%.0f..%.0f,%.0f clip=%d,%d..%d,%d z=%.2f tex=%u trans=%u cb=%u",
                                k, n, (fullW && fullH) ? "plein" : "bande", x0, y0, x1, y1, p.clip_l, p.clip_t, p.clip_r, p.clip_b, zmin, p.textured, p.translucent, p.colorbase));
                    }
                }
                m_rawPrims[k] = q;
                if (m_rawPrimSrc.size() < m_rawPrims.size()) m_rawPrimSrc.resize(m_rawPrims.size());
                m_rawPrimSrc[k] = 0xffffu - uint32_t(m_rawKeys[k] & 0xffffu);   // capture index: its motion matrix
                {
                    uint32_t slot = M2RegionTextures::kNone, micro = M2RegionTextures::kNone;
                    if (m_useRegions && q.textured != 0u) {
                        slot = m_regions.Slot(m_sheetCpu, q.texsheet & 1u, (q.texx - 2048u) & 2047u, (q.texy - 1024u) & 1023u,
                                              q.texwidth, q.texheight);
                        if (q.utex != 0u) micro = m_regions.Slot(m_sheetCpu, (1u - q.texsheet) & 1u, q.utexx, q.utexy, 128, 128);
                    }
                    m_primSlot[k] = slot | (micro << 16);
                    uint32_t layer = M2RegionTextures::kNone, mlayer = M2RegionTextures::kNone;
                    if (q.textured != 0u) {
                        layer = m_regions.Layer(m_sheetCpu, q.texsheet & 1u, (q.texx - 2048u) & 2047u, (q.texy - 1024u) & 1023u,
                                                q.texwidth, q.texheight);
                        if (q.utex != 0u) mlayer = m_regions.Layer(m_sheetCpu, (1u - q.texsheet) & 1u, q.utexx, q.utexy, 128, 128);
                    }
                    m_primLayer[k] = layer | (mlayer << 16);
                }

                for (std::uint32_t v = 0; v < vc; v++) {
                    const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                    float* d = &m_rawVerts[(vo + v) * 5];
                    d[0] = rv.x; d[1] = rv.y; d[2] = rv.z; d[3] = rv.u; d[4] = rv.v;
                    m_rawPrimOfVertex[vo + v] = k;
                }

                // Every full-screen, centred view is the main camera in immersive (25/09, Top Skater: its backdrop
                // uses the same window with the projection centre 10 px lower; drawn as a flat secondary view, its
                // few full-screen polygons cost 27 ms of GPU). The vertices are eye space: the centre does not place
                // them in 3D, so they take the main view's centre and window.
                if (!m_flatMode && m_haveMainView && !(q.center_x == mainCx && q.center_y == mainCy))
                    for (const auto& ml : m_mainLike)
                        if (q.center_x == ml.cx && q.center_y == ml.cy && q.clip_l == ml.l && q.clip_t == ml.t &&
                            q.clip_r == ml.r && q.clip_b == ml.b) {
                            q.center_x = mainCx; q.center_y = mainCy;
                            q.clip_l = mainL; q.clip_t = mainT; q.clip_r = mainR; q.clip_b = mainB;
                            m_rawPrims[k] = q;   // the GPU reads the copy
                            break;
                        }
                const bool isGlass = (q.checker != 0u);
                // A menu drawn flat by the vertex shader (uMenuFlat: every view on the screen plane, secondary
                // parameters) must be routed as secondary here too -- SAME decision, same function (MenuFlat). Routed as
                // main, the cut-outs went to the lean shader, which reads main-view texture parameters: every image with
                // transparent texels vanished (Sega Rally, 26/09: "CAR SELECT", AT/MT labels, checker, mode images).
                const bool isMain = !MenuFlat() && q.center_x == mainCx && q.center_y == mainCy &&
                                    std::abs(q.clip_l - mainL) <= 2 && std::abs(q.clip_t - mainT) <= 2 &&
                                    std::abs(q.clip_r - mainR) <= 2 && std::abs(q.clip_b - mainB) <= 2;
                // Untextured + translucent draws NOTHING on the board (draw_scanline_solid returns).
                const bool invisible = q.textured == 0u && q.translucent != 0u;
                // Discard-free: opaque, no stipple, main camera (secondary views need the clip test).
                const bool fast = !isGlass && isMain && (q.translucent == 0u || !RegionHasHoles(q));
                if (m_primDiagOn && vc >= 3 && arcadexr::config::GetInt("m2.primDiag", 0) == 2) {
                    // debug.tcvr.m2_primDiag=2 (25/09, Top Skater): a summary of every polygon whose first vertex lies
                    // between m2.clusterZ0 and m2.clusterZ1 (the skater, a car...): count, screen box, buckets, passes,
                    // and how many cover each 16-pixel band of screen rows. Logged once after the loop.
                    const tcvr_m2_raw_vertex& r0 = frame.raw_vertices[p.first_vertex];
                    const float z0c = float(arcadexr::config::GetInt("m2.clusterZ0", 10)), z1c = float(arcadexr::config::GetInt("m2.clusterZ1", 40));
                    if (r0.z >= z0c && r0.z <= z1c) {
                        ++m_clN;
                        for (std::uint32_t v = 0; v < vc && v < 8; v++) {
                            const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                            if (rv.z <= 0.0f) continue;
                            const float sxv = float(frame.crtc_xoffset + q.center_x) + rv.x / rv.z;
                            const float syv = float((384 - q.center_y) + frame.crtc_yoffset) - rv.y / rv.z;
                            m_clBox[0] = std::min(m_clBox[0], sxv); m_clBox[1] = std::max(m_clBox[1], sxv);
                            m_clBox[2] = std::min(m_clBox[2], syv); m_clBox[3] = std::max(m_clBox[3], syv);
                            const int band = std::max(0, std::min(23, int(syv) / 16));
                            ++m_clBand[band];
                        }
                        const uint32_t bk = uint32_t((m_rawKeys[k] >> 16) & 0xffffu);
                        m_clBk[0] = std::min(m_clBk[0], bk); m_clBk[1] = std::max(m_clBk[1], bk);
                        if (invisible) ++m_clPass[0]; else if (!isGlass && !isMain) ++m_clPass[1]; else if (!isGlass && !fast) ++m_clPass[2]; else if (isGlass) ++m_clPass[3]; else ++m_clPass[4];
                        if (!isMain) ++m_clNotMain;
                    }
                }
                if (m_primDiagOn && vc >= 3 && arcadexr::config::GetInt("m2.primDiag", 0) != 2) {
                    // debug.tcvr.m2_primDiag=1: every polygon covering arcade pixel (m2.primDiagX, m2.primDiagY), with
                    // its draw rank, bucket, attributes and the pass that draws it (24/09, House of the Dead floor).
                    float sx[8], sy[8]; bool ok = true;
                    for (std::uint32_t v = 0; v < vc && v < 8; v++) {
                        const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                        if (rv.z <= 0.0f) { ok = false; break; }
                        sx[v] = float(frame.crtc_xoffset + q.center_x) + rv.x / rv.z;
                        sy[v] = float((384 - q.center_y) + frame.crtc_yoffset) - rv.y / rv.z;
                    }
                    bool inside = false;
                    if (ok)
                        for (std::uint32_t i2 = 0, j2 = std::min<std::uint32_t>(vc, 8) - 1; i2 < std::min<std::uint32_t>(vc, 8); j2 = i2++)
                            if (((sy[i2] > m_primDiagY) != (sy[j2] > m_primDiagY)) &&
                                (m_primDiagX < (sx[j2] - sx[i2]) * (m_primDiagY - sy[i2]) / (sy[j2] - sy[i2]) + sx[i2]))
                                inside = !inside;
                    if (inside)
                        __android_log_print(ANDROID_LOG_INFO, "TCVR_PRIM",
                            "rank=%u src=%u bucket=%u pass=%s tex=%u transl=%u checker=%u main=%d sheet=%u tx=%u ty=%u w=%u h=%u utex=%u luma=%u lumabase=%u colorbase=%u texlod=%d z0=%.1f",
                            k, 0xffffu - uint32_t(m_rawKeys[k] & 0xffffu), uint32_t((m_rawKeys[k] >> 16) & 0xffffu),
                            invisible ? "none" : (!isGlass && !isMain) ? "sec" : (!isGlass && !fast) ? "cut" : isGlass ? "glass" : "fast",
                            q.textured, q.translucent, q.checker, int(isMain), q.texsheet, q.texx, q.texy, q.texwidth, q.texheight, q.utex,
                            q.luma, q.lumabase, q.colorbase, q.texlod, frame.raw_vertices[p.first_vertex].z);
                    if (inside) {
                        float u0 = 1e9f, u1 = -1e9f, v0 = 1e9f, v1 = -1e9f;
                        for (std::uint32_t v = 0; v < vc; v++) {
                            const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                            u0 = std::min(u0, rv.u); u1 = std::max(u1, rv.u); v0 = std::min(v0, rv.v); v1 = std::max(v1, rv.v);
                        }
                        __android_log_print(ANDROID_LOG_INFO, "TCVR_PRIM", "   wrapx=%u wrapy=%u mirx=%u miry=%u u=%.1f..%.1f v=%.1f..%.1f vc=%u",
                            q.texwrapx, q.texwrapy, q.texmirrorx, q.texmirrory, u0, u1, v0, v1, vc);
                    }
                }
                if (invisible) {
                } else if (!isGlass && !isMain) {
                    for (std::uint32_t t = 1; t + 1 < vc; t++) {
                        rawSecIdx.push_back(q.first_vertex);
                        rawSecIdx.push_back(q.first_vertex + t);
                        rawSecIdx.push_back(q.first_vertex + t + 1);
                    }
                } else if (!isGlass && !fast) {
                    for (std::uint32_t t = 1; t + 1 < vc; t++) {
                        if (t == 1) cutSegStart.push_back(std::uint32_t(rawCutIdx.size()));
                        rawCutIdx.push_back(q.first_vertex);
                        rawCutIdx.push_back(q.first_vertex + t);
                        rawCutIdx.push_back(q.first_vertex + t + 1);
                    }
                } else if (isGlass) {
                    for (std::uint32_t t = 1; t + 1 < vc; t++) {
                        rawGlassIdx.push_back(q.first_vertex);
                        rawGlassIdx.push_back(q.first_vertex + t);
                        rawGlassIdx.push_back(q.first_vertex + t + 1);
                    }
                } else {
                    // The LEAN shader only knows the region images and layers: a textured polygon that has neither
                    // (caches full) got a flat placeholder -- The House of the Dead's tiled floor as one dark green
                    // (25/09). Those go to the full shader, which falls back to the sheets like the board.
                    const bool mirrored = (q.texmirrorx | q.texmirrory) != 0u;
                    const bool noSlot = (m_primSlot[k] & 0xffffu) == (M2RegionTextures::kNone & 0xffffu);
                    const bool noLayer = (m_primLayer[k] & 0xffffu) == (M2RegionTextures::kNone & 0xffffu) || mirrored;
                    if (q.textured != 0u && noSlot && noLayer) {
                        for (std::uint32_t t = 1; t + 1 < vc; t++) {
                            leanFallbackIdx.push_back(q.first_vertex);
                            leanFallbackIdx.push_back(q.first_vertex + t);
                            leanFallbackIdx.push_back(q.first_vertex + t + 1);
                        }
                    } else {
                        for (std::uint32_t t = 1; t + 1 < vc; t++) {
                            m_rawIdx[io++] = q.first_vertex;
                            m_rawIdx[io++] = q.first_vertex + t;
                            m_rawIdx[io++] = q.first_vertex + t + 1;
                        }
                    }
                }
                vo += vc;
            }

            m_leanIndexCount = unsigned(io);
            for (std::uint32_t idx : leanFallbackIdx) m_rawIdx[io++] = idx;
            m_fastIndexCount = unsigned(io);
            // Cut-outs are drawn WITHOUT depth writes (so the Adreno keeps its early depth test and
            // skips every tree fragment hidden by the opaque scene): among themselves they must then
            // be painted far to near. Polygons are in near-to-far rank order: emit them reversed.
            // Mode 1 (no depth write) paints them far to near; modes 0/2 keep near to far (depth decides).
            m_cutMode = std::max(0, std::min(2, arcadexr::config::GetInt("m2.cutMode", 2)));
            if (m_cutMode == 1) {
                for (size_t sgi = cutSegStart.size(); sgi-- > 0;) {
                    const size_t b = cutSegStart[sgi], e = (sgi + 1 < cutSegStart.size()) ? cutSegStart[sgi + 1] : rawCutIdx.size();
                    for (size_t x = b; x < e; ++x) m_rawIdx[io++] = rawCutIdx[x];
                }
            } else {
                for (std::uint32_t idx : rawCutIdx) m_rawIdx[io++] = idx;
            }
            m_secIndexStart = unsigned(io);
            for (std::uint32_t idx : rawSecIdx) m_rawIdx[io++] = idx;
            m_opaqueIndexCount = unsigned(io);
            m_glassIndexCount = unsigned(rawGlassIdx.size());
            for (std::uint32_t idx : rawGlassIdx) {
                m_rawIdx[io++] = idx;
            }
            m_rawIdx.resize(io);
            if (m_primDiagOn && arcadexr::config::GetInt("m2.primDiag", 0) == 2) {
                char bands[256]; int o = 0;
                for (int b = 0; b < 24; ++b) o += std::snprintf(bands + o, sizeof bands - size_t(o), "%u ", m_clBand[b]);
                __android_log_print(ANDROID_LOG_INFO, "TCVR_PRIM",
                    "cluster z %d..%d: %u prims (not main %u) box x %.0f..%.0f y %.0f..%.0f buckets %u..%u passes none=%u sec=%u cut=%u glass=%u fast=%u | vertices per 16-row band: %s",
                    arcadexr::config::GetInt("m2.clusterZ0", 10), arcadexr::config::GetInt("m2.clusterZ1", 40), m_clN, m_clNotMain,
                    m_clBox[0], m_clBox[1], m_clBox[2], m_clBox[3], m_clBk[0], m_clBk[1],
                    m_clPass[0], m_clPass[1], m_clPass[2], m_clPass[3], m_clPass[4], bands);
            }
            GroundProbe(frame, mainCx, mainCy, mainB);
            SceneDepthProbe(frame, mainCx, mainCy);
            // Smooth motion for a Model 2 game (25/09, profile immersive.smoothMotion2: Sega Rally first): polygons carry
            // their identity in the prim (object address, rank, copy) instead of Model 1's u/v.
            m_m2Smooth = !m_directColour && !m_flatMode &&
                         arcadexr::profiles::GetInt("immersive.smoothMotion2", 0) != 0;   // off by default until the flashes are understood (25/09)
            if (m_directColour || m_m2Smooth || m_camDeltaWanted) {
                // Smooth motion (Model 1). Virtua Racing computes its 3D every OTHER arcade frame (30 Hz; measured
                // 24/09) and the same frame is rebuilt for several refreshes: the blend runs between the last two
                // frames that really changed. The match is done once per change, in MatchMotion().
                const size_t nv = m_rawVerts.size() / 5;
                bool same = m_newestPos.size() == nv * 3;
                for (size_t v = 0; same && v < nv; ++v) {
                    const float* d = &m_rawVerts[v * 5];
                    const float* q = &m_newestPos[v * 3];
                    same = d[0] == q[0] && d[1] == q[1] && d[2] == q[2];
                }
                if (!same) {
                    const auto now = std::chrono::steady_clock::now();
                    const float dt = std::chrono::duration<float>(now - m_stepTime).count();
                    if (dt > 0.01f && dt < 0.1f) m_stepPeriod += (dt - m_stepPeriod) * 0.2f;
                    m_stepTime = now;
                    m_newestPos.resize(nv * 3);
                    for (size_t v = 0; v < nv; ++v)
                        for (int c = 0; c < 3; ++c) m_newestPos[v * 3 + c] = m_rawVerts[v * 5 + c];
                    if (m_directColour) MatchMotion(nv); else if (m_m2Smooth) MatchMotionMatrices(frame, nv);
                    if (m_camDeltaWanted && !m_directColour) CameraDelta(frame);
                    if (m_mvWanted && !m_directColour && !m_m2Smooth) MotionPrevCertain(frame, nv);
                } else {
                    m_mvFresh = false;   // same arcade frame again: no motion this refresh
                }
                if (!m_directColour && !m_m2Smooth && !m_mvWanted) m_prevPosCpu.clear();   // camera delta only: no blend
                m_newGeometry = true;
            } else {
                m_prevPosCpu.clear();
            }
            if (arcadexr::config::GetInt("m2.overlayDiag", 0) != 0 && (++m_overlayDiagTick % 60u) == 0u) {
                // Which main-view polygons cover (nearly) the whole arcade frame? Camera-attached overlays (fades,
                // hit flashes) float as panels in immersive.
                const float fx = float(frame.crtc_xoffset + mainCx), fy = float((384 - mainCy) + frame.crtc_yoffset);
                for (uint32_t k = 0; k < m_rawPrims.size(); ++k) {
                    // vertices of rank k: scan m_rawPrimOfVertex
                    float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, zmin = 1e9f, zmax = -1e9f; int nv = 0; bool behind = false;
                    for (size_t v = 0; v < m_rawPrimOfVertex.size(); ++v) {
                        if (m_rawPrimOfVertex[v] != k) continue;
                        const float* q = &m_rawVerts[v * 5];
                        zmin = std::min(zmin, q[2]); zmax = std::max(zmax, q[2]); ++nv;
                        if (q[2] <= 1e-3f) { behind = true; continue; }
                        const float sx = fx + q[0] / q[2], sy = fy - q[1] / q[2];
                        x0 = std::min(x0, sx); x1 = std::max(x1, sx); y0 = std::min(y0, sy); y1 = std::max(y1, sy);
                    }
                    if (nv == 0 || behind) continue;
                    if (x1 - x0 < 0.8f * 496.0f || y1 - y0 < 0.6f * 384.0f) continue;
                    const tcvr_m2_prim& p = m_rawPrims[k];
                    Log::Write(Log::Level::Info, Fmt("TCVR_OVERLAY rank=%u/%zu z=%.2f..%.2f box=%.0f,%.0f..%.0f,%.0f tex=%u trans=%u checker=%u cb=%u luma=%u",
                        k, m_rawPrims.size(), zmin, zmax, x0, y0, x1, y1, p.textured, p.translucent, p.checker, p.colorbase, p.luma));
                }
            }

            // first_vertex is not read by any shader: it now carries the region slots
            // (main | microtexture << 16) to the vertex stage, which hands them on flat.
            // vertex_count, also unread by the shaders, carries the texture-array layers the same way.
            for (std::uint32_t k = 0; k < n; k++) {
                m_rawPrims[k].first_vertex = m_primSlot[k];
                m_rawPrims[k].vertex_count = m_primLayer[k];
            }
        }
    }

    // GPU-visible half: after the fence. Colour tables, texture sheets, region uploads,
    // descriptor updates, copies into the host-visible buffers, 2D layers.
    void CommitFrame(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        if (!m_initialized) return;
        if (m_rbPendingNext) { m_rbPendingNext = false; m_rbWaitFrames = 3; }
        if (m_rbWaitFrames > 0 && --m_rbWaitFrames == 0) m_rbPending = true;
        if (m_dbgMapped && (m_dbgMapped[0] | m_dbgMapped[1]) != 0u) {
            static unsigned s_ov = 0;
            if ((s_ov++ % 30u) == 0u)
                Log::Write(Log::Level::Info, Fmt("TCVR_M2VK overdraw frags opaque=%u other=%u eyePixels=%u -> opaque shaded x%.2f per eye pixel (MSAA counts once/pixel)",
                                                 m_dbgMapped[0], m_dbgMapped[1], m_lastEyePixels,
                                                 m_lastEyePixels ? double(m_dbgMapped[0]) / (2.0 * m_lastEyePixels) : 0.0));
            m_dbgMapped[0] = m_dbgMapped[1] = 0u;
        }
        // Geometry unchanged (Virtua Cop draws its 3D at 30 Hz: every other frame): the scene is the last built
        // one, but THIS frame slot's buffers still hold the scene from two frames ago. Returning here made the
        // image alternate between the current and the previous scene -- flashes and double images everywhere
        // (Guillaume, 23/09). Re-commit the last built arrays into this slot instead.
        const bool reuse = frame.geometry_unchanged != 0u;
        if (reuse && m_rawPrims.empty()) { UploadLayers(frame, cmd); return; }
        if (!reuse && !m_built) return;
        {   // debug.tcvr.m2_regionsReset=<n>: rebuild every region/layer now (each new value triggers once)
            const int rr = arcadexr::config::GetInt("m2.regionsReset", 0);
            if (rr != m_lastRegionsReset) {
                m_lastRegionsReset = rr;
                vkDeviceWaitIdle(m_vkDevice);
                m_regions.Clear();
                if (!reuse) BuildFrame(frame);
                Log::Write(Log::Level::Info, "TCVR_M2VK regions reset (debug)");
                if (!reuse && !m_built) return;
            }
        }
        UploadColourChain(frame);
        // A full upload on a REUSED frame (geometry unchanged: every other frame in 30 Hz games) cannot rebuild
        // the regions then; it must happen on the next built frame, even though the textures no longer change by
        // then. It never happened: stale regions and layers until a restart (The House of the Dead's tiled floor
        // drawn with the previous scene's texels, 25/09).
        if (UploadTextures(frame, cmd) == 1) m_regionsStale = true;
        if (const uint32_t st = m_regions.Validate(m_sheetCpu, 6u))
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK stale region/layer caught: %u now, %llu in total (e.g. sheet %u at %u,%u %ux%u), partial uploads %u",
                                             st, (unsigned long long)m_regions.StaleTotal(), m_regions.m_lastStale[0], m_regions.m_lastStale[1],
                                             m_regions.m_lastStale[2], m_regions.m_lastStale[3], m_regions.m_lastStale[4], m_partialTexLog));
        m_regions.ProcessRefills(m_sheetCpu);   // regions whose texels a partial update rewrote
        const bool texChanged = m_regionsStale;
        if (texChanged && !reuse) {
            m_regionsStale = false;
            // Rare (course / menu load): the regions were chosen on the old sheets -> rebuild. (Not on a reused
            // frame: its polygons still point at the current regions; the next built frame rebuilds.)
            if (m_regions.Count() > 0 || m_regions.LayerCount() > 0) {   // layers too: they can exist alone
                vkDeviceWaitIdle(m_vkDevice);
                m_regions.Clear();
            }
            if (!reuse) { BuildFrame(frame); if (!m_built) return; }
        }
        if (!m_rawPrims.empty()) {
            if (!m_rawVerts.empty()) {
                size_t vertBytes = std::min(m_rawVerts.size() * sizeof(float), m_vboSize);
                memcpy(m_vboMappedF[m_fs], m_rawVerts.data(), vertBytes);
            }
            if (!m_rawPrimOfVertex.empty()) {
                size_t primOfVertBytes = std::min(m_rawPrimOfVertex.size() * sizeof(uint32_t), m_primIndexSize);
                memcpy(m_primIndexMappedF[m_fs], m_rawPrimOfVertex.data(), primOfVertBytes);
            }
            if (!m_rawIdx.empty()) {
                size_t idxBytes = std::min(m_rawIdx.size() * sizeof(uint32_t), m_iboSize);
                memcpy(m_iboMappedF[m_fs], m_rawIdx.data(), idxBytes);
            }
            if (!m_prevPosCpu.empty()) {
                size_t pb = std::min(m_prevPosCpu.size() * sizeof(float), m_prevSize);
                memcpy(m_prevMappedF[m_fs], m_prevPosCpu.data(), pb);
            } else if (m_prevMappedF[m_fs]) {
                std::memset(m_prevMappedF[m_fs], 0, std::min(m_prevSize, size_t(16)));   // w = 0 on vertex 0: no blend
            }
            {   // always: the descriptor array must be fully valid (dummy image) before any draw
                m_regions.Flush(cmd);
                // New regions go into THIS slot's sets now, and into the other slot's sets the next
                // time that slot is recorded (its sets may still be in use by the GPU right now).
                const bool fullNow = m_regions.NeedsFullDescriptorWrite();
                std::vector<uint32_t> dirty = m_regions.TakeDirty();
                const int other = 1 - m_fs;
                const bool full = fullNow || m_fullPend[m_fs];
                std::vector<uint32_t> slots = m_pendSlots[m_fs];
                slots.insert(slots.end(), dirty.begin(), dirty.end());
                if (full || !slots.empty())
                    for (int e = 0; e < 2; ++e)
                        for (int ps = 0; ps < 2; ++ps) m_regions.WriteDescriptors(m_m2DescSetF[m_fs][e][ps], 12, 13, full, slots);
                m_fullPend[m_fs] = false;
                m_pendSlots[m_fs].clear();
                if (fullNow) { m_fullPend[other] = true; m_pendSlots[other].clear(); }
                else m_pendSlots[other].insert(m_pendSlots[other].end(), dirty.begin(), dirty.end());
                m_regions.DescriptorsDone();
                static unsigned s_regLog = 0;
                if ((s_regLog++ % 300u) == 0u)
                    Log::Write(Log::Level::Info, Fmt("TCVR_M2VK regions=%u created=%u", m_regions.Count(), m_regions.Created()));
            }
            size_t primBytes = std::min(m_rawPrims.size() * sizeof(tcvr_m2_prim), m_primsSize);
            memcpy(m_primsMappedF[m_fs], m_rawPrims.data(), primBytes);
        }
        UploadLayers(frame, cmd);
        m_preparedSeq = frame.sequence;
        // debug.tcvr.m2_rb=<n>: read back one region as layer AND as image (new value = once)
        if (m_rbPending) { m_regions.WriteReadback(arcadexr::config::ExternalDirectory()); m_rbPending = false; }
        const int rb = arcadexr::config::GetInt("m2.rb", 0);
        if (rb != m_lastRb) { m_lastRb = rb; if (rb != 0 && m_regions.RecordReadback(cmd, rb)) m_rbPendingNext = true; }
    }

    bool RenderImmersive(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView,
                         VkCommandBuffer cmd, VkExtent2D renderAreaExtent) {
        if (!m_initialized || m_opaqueIndexCount == 0) return false;

        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) return false;

        const float distance = std::max(0.25f, arcadexr::config::GetFloat("screen.distance", 2.0f));
        const float depthUnits = std::max(0.5f, arcadexr::config::GetFloat("m2.immersiveDepth", 9.4f));
        // World scale (23/09, Guillaume: "everything looks small and so do I"). The base was Sega Rally's
        // (distance / m2.immersiveDepth); gun games now put the player's eyes immersive.eyeHeight (1.65 m) above the
        // game's floor, measured every frame. The menu multiplier (immersive.worldScaleMul) applies to every game.
        const bool autoScale = arcadexr::profiles::GetInt("immersive.autoScale", arcadexr::profiles::IsDriving() ? 0 : 1) != 0;
        const float eyeHeight = arcadexr::profiles::GetFloat("immersive.eyeHeight", 1.65f);
        float baseScale = distance / depthUnits;
        // ISO scale (25/09): Sega Rally's units are metres (the Lancia measured on the PC MAME: 1.9 x 4.0 units for
        // 1.77 x 3.90 m). immersive.metresPerUnit > 0 sets the world scale directly; 0 = the old screen-based one.
        const float metresPerUnit = arcadexr::profiles::GetFloat("immersive.metresPerUnit",
            arcadexr::profiles::CurrentGame() == "srallyc" ? 1.0f : 0.0f);
        if (metresPerUnit > 0.0f) baseScale = metresPerUnit;
        if (autoScale && m_camHeightUnits > 1e-3f) baseScale = std::max(0.005f, std::min(5.0f, eyeHeight / m_camHeightUnits));
        const float worldScale = baseScale * std::max(0.1f, std::min(10.0f, arcadexr::profiles::GetFloat("immersive.worldScaleMul", 1.0f)));
        const arcadexr::gun::Vec3 camera{screen.center.x + screen.normal.x * distance,
                                         screen.center.y + screen.normal.y * distance,
                                         screen.center.z + screen.normal.z * distance};

        // Telephoto cap (23/09, Virtua Cop): a gun game films with a focus of 1500-1800 (17 degrees across the
        // arcade screen). At the headset's true angles its enemies were specks. Immersive uses at most focusMax
        // (700 = 39 degrees) for the metric conversion, which magnifies the scene by focus/focusMax and keeps the
        // game's own zooms; racing games (true angles) are not capped. Flat keeps the true focus.
        const float focusCap = m_flatMode ? 0.0f
            : arcadexr::profiles::GetFloat("immersive.focusMax", arcadexr::profiles::IsDriving() ? 0.0f : 700.0f);
        auto capF = [focusCap](float f) { return (focusCap > 1.0f) ? std::min(f, focusCap) : f; };
        const float focusY = capF((m_m2FocusY > 1.0f) ? m_m2FocusY : 512.0f);
        const float focusX = capF((m_m2FocusX > 1.0f) ? m_m2FocusX : 512.0f);
        if (!m_flatMode && viewIndex == 0) { m_aimFocus[0] = focusX; m_aimFocus[1] = focusY; }
        // Ground fill (gun games): below the game's horizon, the colour of the floor the arcade shows, not its 2D layer.
        const bool groundFill = !m_flatMode && m_groundProbed &&
            arcadexr::profiles::GetInt("immersive.groundFill", arcadexr::profiles::IsDriving() ? 0 : 1) != 0;
        const float* groundCol = groundFill ? m_groundProbe : m_groundColor;
        const float clipRow = (groundFill && m_horizonGeo >= 0.0f) ? m_horizonGeo + 2.0f : 0.0f;

        float pitchTarget = 0.0f;
        // Model 1: the horizon estimate swings with far mountains above the real horizon (0.01 to 0.35 rad measured
        // in a race, 24/09) and the "levelling" tilted the world by up to 20 degrees. Its camera is nearly level:
        // no levelling by default (immersive.pitchLevel=1 to re-enable).
        const bool levelPitch = m_directColour ? arcadexr::profiles::GetInt("immersive.pitchLevel", 0) != 0
                                               : arcadexr::config::GetInt("m2.immersivePitch", 1) != 0;
        if (levelPitch && m_haveMainView && m_horizonGeo >= 0.0f) {
            const float t = ((384.0f - float(m_mainCenter[1])) - m_horizonGeo) / std::max(focusY, 1.0f);
            pitchTarget = std::max(-0.35f, std::min(0.35f, ::atanf(t)));
        }
        // Horizon lost for a moment in a race (Model 1): keep the last pitch rather than swing back to level.
        if (m_directColour && m_haveMainView && m_horizonGeo < 0.0f && !m_isMenuM1) pitchTarget = m_lastPitchTarget;
        else if (viewIndex == 0) m_lastPitchTarget = pitchTarget;
        pitchTarget += arcadexr::config::GetFloat("m2.immersivePitchOffset", 0.0f);
        if (viewIndex == 0) m_m2Pitch += (pitchTarget - m_m2Pitch) * 0.1f;
        if (viewIndex == 0 && !m_flatMode && (++m_pitchLogTick % 30u) == 0u)
            Log::Write(Log::Level::Info, Fmt("TCVR_PITCH hz=%d horizon=%.1f target=%.4f applied=%.4f menu=%d zmax=%.0f prims=%zu", m_haveHorizon ? 1 : 0, m_horizonGeo, pitchTarget, m_m2Pitch, m_backNoTile ? 1 : 0, m_mainZMax, m_rawPrims.size()));

        const float cp = ::cosf(m_m2Pitch), sp = ::sinf(m_m2Pitch);
        const arcadexr::gun::Vec3 upP{cp * screen.up.x - sp * screen.normal.x, cp * screen.up.y - sp * screen.normal.y, cp * screen.up.z - sp * screen.normal.z};
        const arcadexr::gun::Vec3 normalP{cp * screen.normal.x + sp * screen.up.x, cp * screen.normal.y + sp * screen.up.y, cp * screen.normal.z + sp * screen.up.z};

        if (!m_flatMode && viewIndex == 0) {
            // Lightgun (Aim): the arcade-to-world transform of the image the player sees.
            m_aimXf = {true, camera, screen.right, upP, normalP, worldScale,
                       std::max(0.002f, std::min(1.0f, arcadexr::config::GetFloat("m2.immersiveNear", 0.25f)))};
        }
        XrMatrix4x4f arcadeToWorld{};
        arcadeToWorld.m[0] = screen.right.x * worldScale;  arcadeToWorld.m[1] = screen.right.y * worldScale;  arcadeToWorld.m[2] = screen.right.z * worldScale;
        arcadeToWorld.m[4] = upP.x * worldScale;          arcadeToWorld.m[5] = upP.y * worldScale;          arcadeToWorld.m[6] = upP.z * worldScale;
        arcadeToWorld.m[8] = -normalP.x * worldScale;     arcadeToWorld.m[9] = -normalP.y * worldScale;     arcadeToWorld.m[10] = -normalP.z * worldScale;
        arcadeToWorld.m[12] = camera.x; arcadeToWorld.m[13] = camera.y; arcadeToWorld.m[14] = camera.z; arcadeToWorld.m[15] = 1.0f;

        const float farMetres = std::max(200.0f, arcadexr::config::GetFloat("immersive.far", 20000.0f));
        const float nearMetres = std::max(0.002f, std::min(1.0f, arcadexr::config::GetFloat("m2.immersiveNear", 0.25f)));
        if (viewIndex == 0 && !m_flatMode) {
            m_a2wRight = screen.right; m_a2wUp = upP; m_a2wNormal = normalP; m_a2wCam = camera; m_a2wScale = worldScale;
            m_a2wValid = true; m_depthNear = nearMetres; m_depthFar = farMetres;
        }

        XrMatrix4x4f projection;
        XrMatrix4x4f_CreateProjectionFov(&projection, GRAPHICS_VULKAN, layerView.fov, nearMetres, farMetres);
        XrMatrix4x4f eyeToWorld;
        XrMatrix4x4f_CreateFromRigidTransform(&eyeToWorld, &layerView.pose);
        XrMatrix4x4f worldToEye;
        XrMatrix4x4f_InvertRigidBody(&worldToEye, &eyeToWorld);
        XrMatrix4x4f viewProjection;
        XrMatrix4x4f_Multiply(&viewProjection, &projection, &worldToEye);
        XrMatrix4x4f mvp;
        XrMatrix4x4f_Multiply(&mvp, &viewProjection, &arcadeToWorld);
        if (!m_flatMode) {   // AppSW depth pass (RenderAppSwDepth): the same placement as this eye's colour pass
            std::memcpy(m_swMvp[viewIndex & 1u], mvp.m, sizeof(mvp.m));
            m_swFocus[0] = focusX; m_swFocus[1] = focusY;
            m_swMvpValid[viewIndex & 1u] = true;
        }

        // HUD at 2 m (23/09-24/09): at 10 m it was drawn OVER 3D much nearer (a car at 3 m), which the eyes read as
        // "behind yet in front" -- the texts made Guillaume squint, in Virtua Racing and Virtua Cop. Same apparent
        // size and the same angle above the gaze (m2.hudLift is metres at 10 m). The sky keeps its own distance.
        const float hudDistance = std::max(0.5f, arcadexr::config::GetFloat("m2.hudDistance", 2.0f));
        const float hudK = hudDistance / distance;
        const float hudLiftAt10 = arcadexr::config::GetFloat("m2.hudLift", 1.8f);
        const float hudLift = hudLiftAt10 * hudDistance / 10.0f;
        const float backDistance = std::max(0.5f, arcadexr::config::GetFloat("m2.backDistance", 10.0f));
        const arcadexr::gun::Vec3 hudCenter{camera.x - screen.normal.x * hudDistance + screen.up.x * hudLift,
                                            camera.y - screen.normal.y * hudDistance + screen.up.y * hudLift,
                                            camera.z - screen.normal.z * hudDistance + screen.up.z * hudLift};

        XrMatrix4x4f hudToWorld{};
        hudToWorld.m[0] = screen.right.x * screen.width * hudK; hudToWorld.m[1] = screen.right.y * screen.width * hudK; hudToWorld.m[2] = screen.right.z * screen.width * hudK;
        hudToWorld.m[4] = screen.up.x * screen.height * hudK;   hudToWorld.m[5] = screen.up.y * screen.height * hudK;   hudToWorld.m[6] = screen.up.z * screen.height * hudK;
        hudToWorld.m[8] = screen.normal.x; hudToWorld.m[9] = screen.normal.y; hudToWorld.m[10] = screen.normal.z;
        hudToWorld.m[12] = hudCenter.x; hudToWorld.m[13] = hudCenter.y; hudToWorld.m[14] = hudCenter.z; hudToWorld.m[15] = 1.0f;
        // HUD ISO (25/09): each HUD pixel on the board camera's ray through that pixel, at the depth the player
        // looks at (SceneDepthProbe), through the same arcade-to-world transform as the 3D (scale, pitch). It covers
        // exactly what it covers on the cabinet (0 px), whatever the depth. Profile immersive.hudIso, Sega Rally only.
        // A menu screen (26/09) is ONE flat screen: all of it -- boxes, texts AND the 2D page behind, see hudMvpBack -- on
        // the rays at the depth the game draws its 3D (deepest main-view vertex: 28 units in Sega Rally's menus, i.e.
        // far and large as on 25/09). Brought to the 2 m HUD plane it sat "right in front of the eyes"; boxes at the
        // game depth with the page left at 2 m, it was a parallax.
        const bool hudIso = !m_flatMode && m_haveMainView && (m_isMenuM1 ? MenuIso() : m_sceneDepth > 0.0f) && HudIsoProfile();
        if (hudIso) {
            const float zh = std::max(1.0f, m_isMenuM1 ? m_mainZMax : m_sceneDepth);
            const float fcx = float(m_crtc[0]) + float(m_mainCenter[0]), fcy = float(384 - m_mainCenter[1]) + float(m_crtc[1]);
            XrMatrix4x4f H{};
            H.m[0] = 496.0f / focusX * zh;
            H.m[5] = 384.0f / focusY * zh;
            H.m[10] = 1.0f;
            H.m[12] = (248.0f - fcx) / focusX * zh;
            H.m[13] = (fcy - 192.0f) / focusY * zh;
            H.m[14] = zh;
            H.m[15] = 1.0f;
            XrMatrix4x4f_Multiply(&hudToWorld, &arcadeToWorld, &H);
        }
        XrMatrix4x4f hudMvp;
        XrMatrix4x4f_Multiply(&hudMvp, &viewProjection, &hudToWorld);
        if (!m_flatMode && viewIndex == 0) {
            m_aimHud = {true, hudCenter, screen.right, screen.up, screen.normal, screen.width * hudK, screen.height * hudK};
        }

        XrMatrix4x4f hudToWorldBack{};
        // Back layer (the game's 2D sky). It used to be tilted with the SMOOTHED pitch estimate while the game
        // scrolls its sky INSTANTLY for the same camera move: the difference made it slide like wallpaper going
        // up and down (Guillaume, 23/09). Now level, and re-anchored every frame so that the image row of the
        // game's horizon (where the game has just scrolled it) sits exactly at eye level.
        // A menu of the game (no main 3D view): its 2D back layer is part of the screen, not a sky -- same plane as the
        // HUD, level, not repeated (the course select showed ten copies of a course, and the selection frame of the
        // front layer did not sit on the course of the back one).
        const bool menuScreen = !m_flatMode && (!m_haveMainView || m_isMenuM1);
        m_backNoTile = menuScreen;
        const bool skyAnchor = !menuScreen && !m_flatMode && m_horizonGeo >= 0.0f &&
            arcadexr::profiles::GetInt("immersive.skyAnchor", arcadexr::profiles::IsDriving() ? 0 : 1) != 0;   // GOLD (Sega Rally) unchanged by default
        const float backD = menuScreen ? hudDistance : backDistance;
        const float backK = backD / distance;
        const float backH = screen.height * backK;
        const float backLift = menuScreen ? hudLift : (skyAnchor ? -(0.5f - m_horizonGeo / 384.0f) * backH : hudLiftAt10 * backD / 10.0f);
        const bool level = skyAnchor || menuScreen;
        const arcadexr::gun::Vec3 bN = level ? screen.normal : normalP, bU = level ? screen.up : upP;
        const arcadexr::gun::Vec3 backCenter{camera.x - bN.x * backD + screen.up.x * backLift,
                                             camera.y - bN.y * backD + screen.up.y * backLift,
                                             camera.z - bN.z * backD + screen.up.z * backLift};
        hudToWorldBack.m[0] = screen.right.x * screen.width * backK; hudToWorldBack.m[1] = screen.right.y * screen.width * backK; hudToWorldBack.m[2] = screen.right.z * screen.width * backK;
        hudToWorldBack.m[4] = bU.x * backH;                         hudToWorldBack.m[5] = bU.y * backH;                         hudToWorldBack.m[6] = bU.z * backH;
        hudToWorldBack.m[8] = bN.x; hudToWorldBack.m[9] = bN.y; hudToWorldBack.m[10] = bN.z;
        hudToWorldBack.m[12] = backCenter.x; hudToWorldBack.m[13] = backCenter.y; hudToWorldBack.m[14] = backCenter.z; hudToWorldBack.m[15] = 1.0f;
        XrMatrix4x4f hudMvpBack;
        XrMatrix4x4f_Multiply(&hudMvpBack, &viewProjection, &hudToWorldBack);
        if (hudIso && menuScreen) hudMvpBack = hudMvp;   // the page on the plane of its boxes (see hudIso)
        if (m_flatMode) {
            // FLAT: the board's own projection into the flat target (see RenderFlat), every view on the
            // screen plane mapped 1:1 onto it (plane (u,v) in -0.5..0.5 -> NDC (2u, -2v), Vulkan y down).
            mvp = m_flatMvp;
            XrMatrix4x4f plane{};
            plane.m[0] = 2.0f; plane.m[5] = -2.0f; plane.m[15] = 1.0f;
            hudMvp = plane;
            hudMvpBack = plane;
        }

        // Update UBO slices for this eye
        const uint32_t eye = (viewIndex < 2) ? viewIndex : 0;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            M2UniformBufferObject& ubo = *m_uboMappedF[m_fs][eye][pass];
            memcpy(ubo.uMvp, mvp.m, sizeof(mvp.m));
            memcpy(ubo.uHudMvp, hudMvp.m, sizeof(hudMvp.m));
            ubo.uViewport[0] = 496.0f; ubo.uViewport[1] = 384.0f;
            ubo.uFocus[0] = focusX; ubo.uFocus[1] = focusY;
            // The board's screen offsets (0 on Sega Rally, not on Virtua Cop): everything placed in board pixels (secondary
            // views, screen overlays) was about 128 rows off with zeros here (23/09).
            ubo.uCrtc[0] = m_crtc[0]; ubo.uCrtc[1] = m_crtc[1];
            // Menu screen (Virtua Racing course select): its small 3D objects (wheel, pedal, cursor) belong to the
            // 2D page; drawn in perspective with the game camera's angle they did not sit on the page (24/09).
            ubo.uMenuFlat = (m_backNoTile && (!m_haveMainView || MenuFlat())) ? 1 : 0;
            ubo.uMainClip[0] = m_mainClip[0]; ubo.uMainClip[1] = m_mainClip[1];
            ubo.uMainClip[2] = m_mainClip[2]; ubo.uMainClip[3] = m_mainClip[3];
            ubo.uMainCenter[0] = m_mainCenter[0]; ubo.uMainCenter[1] = m_mainCenter[1];
            ubo.uEyeOffset = 0.0f;
            ubo.uConvergence = 0.0f;
            ubo.uDepthBias = arcadexr::config::GetFloat("m2.immersiveDepthBias", 6.0e-7f);
            ubo.uPrimCount = float(m_rawPrims.size());
            ubo.uImmersive = 1;
            ubo.uDepthOrder = arcadexr::config::GetInt("m2.depthOrder", 1);
            ubo.uRaw = 1;
            ubo.uScale = 1;
            ubo.uStipple = 1;
            ubo.uPass = (pass == 0) ? 1 : 2;  // 1 = opaque, 2 = glass
            ubo.uPrepassClass = (m_lutValid && false) ? 1 : 0;  // colour table removed from the shader (slower on Adreno, see m2_frag.glsl)  // = uUseLut. OFF: measured SLOWER on the Adreno (3 texelFetch of a 64x32 table ~10 ms vs 7 SSBO loads ~5.8 ms, 22/09, interleaved A/B), even as a combined sampler
            ubo.uEdgeFade = 0.0f;
            ubo.uHorizonRow = m_horizonGeo;
            const bool popFade = m_directColour && !m_flatMode && !m_backNoTile && m_zMaxSmooth > 0.0f &&
                                 arcadexr::profiles::GetInt("immersive.popInFade", 1) != 0;
            ubo.uFogFar = popFade ? m_zMaxSmooth : 0.0f;
            if (m_smooth && (m_directColour || m_m2Smooth) && !m_prevPosCpu.empty()) {
                // Blend at THIS refresh: time since the newest moving frame over the measured step interval.
                const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - m_stepTime).count();
                ubo.uInterp = std::max(0.0f, std::min(1.0f, t / std::max(0.008f, m_stepPeriod)));
            } else {
                ubo.uInterp = 1.0f;
            }
            m_lastInterp = ubo.uInterp;
            if (popFade) { ubo.uSky[0] = m_fogColour[0]; ubo.uSky[1] = m_fogColour[1]; ubo.uSky[2] = m_fogColour[2]; }
            else { ubo.uSky[0] = m_voidColor[0]; ubo.uSky[1] = m_voidColor[1]; ubo.uSky[2] = m_voidColor[2]; }
            ubo.uGround[0] = groundCol[0]; ubo.uGround[1] = groundCol[1]; ubo.uGround[2] = groundCol[2];
            ubo.uAniso = std::max(1, std::min(8, arcadexr::config::GetInt("m2.aniso", 1)));
            ubo.uFilterMode = std::max(0, std::min(5, arcadexr::config::GetInt("m2.filter", 5)));
            ubo.uMipBias = std::max(0, std::min(512, arcadexr::config::GetInt("m2.mipBias", 0)));
            ubo.uAlphaCoverage = IsMsaa() ? arcadexr::config::GetInt("m2.alphaCoverage", 1) : 0;
            // 1.2 / -0.02 compensated the washed-out sRGB double encoding, fixed at the root (raw colour views):
            // neutral by default now; contrast/bright props remain for A/B.
            ubo.uContrast = m_flatMode ? 1.0f : arcadexr::config::GetFloat("contrast", 1.0f);
            ubo.uBright = m_flatMode ? 0.0f : arcadexr::config::GetFloat("bright", 0.0f);
            ubo.uLift = LiftExponent();
            ubo.uTestStage = arcadexr::config::GetInt("m2.stage", 0);
            ubo.uOverlayK = m_flatMode ? 1.0f : std::max(1.0f, arcadexr::config::GetFloat("m2.overlayScale", 8.0f));
            ubo.padEnd[1] = m_gammaFolded ? 1 : 0;   // = uGammaFolded
            ubo.padEnd[2] = arcadexr::config::GetInt("m2.texImplicit", 1) | (arcadexr::config::GetInt("m2.texArray", 1) != 0 ? 2 : 0);   // = uTexImplicit | 2: texture array
            ubo.padEnd[0] = arcadexr::config::GetInt("m2.hwAnisoOn", 1) != 0 ? 0 : 4;   // = uSmpBase (live A/B)
            ubo.uBoardLod = (arcadexr::config::GetInt("m2.boardLod", 1) != 0 &&
                             arcadexr::profiles::GetInt("immersive.boardLod", 0) != 0) ? 1 : 0;
            ubo.uCountOverdraw = (m_useRegions && arcadexr::config::GetInt("m2.regions", 1) != 0) ? 1 : 0;  // = uUseRegions
        }

        m_lastEyePixels = renderAreaExtent.width * renderAreaExtent.height;
        const float outW = float(renderAreaExtent.width);
        const float outH = float(renderAreaExtent.height);

        // Debug, live: debug.tcvr.m2.vkskip bitmask to time each pass on the headset.
        // 1 void, 2 back plane, 4 discard-free opaque, 8 cut-outs/secondary, 16 glass, 32 front HUD.
        const int skip = arcadexr::config::GetInt("m2.vkskip", 0);
        const bool bgFar = true;  // background is drawn after the geometry: far-plane, depth-tested
        // 3. Draw Model 2 Geometry
        if (m_opaqueIndexCount > 0) {
            VkBuffer vtxBufs[2] = { m_vboBufferF[m_fs].buf, m_primIndexBufferF[m_fs].buf };
            VkDeviceSize offsets[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vtxBufs, offsets);
            vkCmdBindIndexBuffer(cmd, m_iboBufferF[m_fs].buf, 0, VK_INDEX_TYPE_UINT32);

            // Pass 1: Opaque
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSetF[m_fs][eye][0], 0, nullptr);
            if (m_fastIndexCount > 0 && !(skip & 4)) {
                // LEAN: the same opaque polygons through a shader holding only their path (occupancy).
                const bool lean = (arcadexr::config::GetInt("m2.lean", 3) & 1) != 0 && ubo_texArray(eye) && m_edgeFadeOff;
                if (lean) {
                    if (m_leanIndexCount > 0) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineFastLean);
                        vkCmdDrawIndexed(cmd, m_leanIndexCount, 1, 0, 0, 0);
                    }
                    if (m_fastIndexCount > m_leanIndexCount) {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineFast);
                        vkCmdDrawIndexed(cmd, m_fastIndexCount - m_leanIndexCount, 1, m_leanIndexCount, 0, 0);
                    }
                } else {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineFast);
                    vkCmdDrawIndexed(cmd, m_fastIndexCount, 1, 0, 0, 0);
                }
            }
            const bool leanOn = (arcadexr::config::GetInt("m2.lean", 3) & 2) != 0 && ubo_texArray(eye) && m_edgeFadeOff;
            if (leanOn && m_secIndexStart > m_fastIndexCount && !(skip & 8)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineCutLean);
                vkCmdDrawIndexed(cmd, m_secIndexStart - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
            }
            const int cutModeEff = leanOn ? -1 : m_cutMode;   // lean handled above
            const bool cutNoWrite = (cutModeEff == 1);
            if (cutModeEff == 2 && m_opaqueIndexCount > m_fastIndexCount && !(skip & 8)) {
                // Cut-outs, depth pre-pass then colour of the visible sample only (before the background,
                // which then only fills what nothing covered).
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineCutDepth);
                vkCmdDrawIndexed(cmd, m_secIndexStart - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineCutColor);
                vkCmdDrawIndexed(cmd, m_secIndexStart - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
            }
            if (cutModeEff == 0) {
            if (m_opaqueIndexCount > m_fastIndexCount && !(skip & 8)) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineOpaque);
                    vkCmdDrawIndexed(cmd, m_secIndexStart - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
                }
    
    }
            // Background AFTER the opaque geometry, depth-tested on the far plane (m2.bgAfter=1):
            // void + back 2D layer now shade only the pixels no polygon covered.
        // 1. DrawVoid
            float invMvp[16];
            if (!(skip & 1) && !m_flatMode && InvertMatrix4x4(mvp.m, invMvp)) {
                VoidPushConstants voidPc{};
                voidPc.uLift = LiftExponent();
                memcpy(voidPc.uInvMvp, invMvp, sizeof(invMvp));
                voidPc.uSky[0] = m_voidColor[0]; voidPc.uSky[1] = m_voidColor[1]; voidPc.uSky[2] = m_voidColor[2]; voidPc.uSky[3] = 1.0f;
                voidPc.uGround[0] = groundCol[0]; voidPc.uGround[1] = groundCol[1]; voidPc.uGround[2] = groundCol[2]; voidPc.uGround[3] = 1.0f;
                voidPc.uOutSize[0] = outW; voidPc.uOutSize[1] = outH;
    
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (bgFar ? m_voidPipelineFar : m_voidPipeline));
                vkCmdPushConstants(cmd, m_voidPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(voidPc), &voidPc);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
    
            // 2. DrawPlaneLayer (Back 2D)
            if (m_haveLayer[1] && !(skip & 2)) {
                PlanePushConstants backPc{};
                backPc.uLift = LiftExponent();
                memcpy(backPc.uHudMvp, hudMvpBack.m, sizeof(hudMvpBack.m));
                backPc.uOutSize[0] = outW; backPc.uOutSize[1] = outH;
                backPc.uKeyZero = 0;
                backPc.uUvScaleX = m_layerUvScaleX[1];
                backPc.uClipRow = clipRow;
                backPc.uNoTile = m_backNoTile ? 1.0f : 0.0f;
    
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (bgFar ? m_planePipelineFar : m_planePipeline));
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipelineLayout, 0, 1, &m_layerDescSet[1], 0, nullptr);
                vkCmdPushConstants(cmd, m_planePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(backPc), &backPc);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
    
    
            // Secondary views (rear-view mirror, previews) on the arcade screen plane: own range, skip bit 64.
            if (m_opaqueIndexCount > m_secIndexStart && !(skip & 64)) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSetF[m_fs][eye][0], 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineOpaque);
                vkCmdDrawIndexed(cmd, m_opaqueIndexCount - m_secIndexStart, 1, m_secIndexStart, 0, 0);
            }
            // Cut-outs after the background, no depth write, far to near (see index building).
            if (cutNoWrite && m_opaqueIndexCount > m_fastIndexCount && !(skip & 8)) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSetF[m_fs][eye][0], 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineCutNoWrite);
                vkCmdDrawIndexed(cmd, m_secIndexStart - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
            }

            // Pass 2: Glass (after the background: it blends over what is really behind it)
            if (m_glassIndexCount > 0 && !(skip & 16)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineGlass);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSetF[m_fs][eye][1], 0, nullptr);
                vkCmdDrawIndexed(cmd, m_glassIndexCount, 1, m_opaqueIndexCount, 0, 0);
            }
        }

        // 4. DrawPlaneLayer (Front 2D HUD)
        m_hudMvAll[eye & 1u] = false; m_hudMvRects[eye & 1u].clear();   // where the HUD covers the eye (AppSW vectors)
        if (m_haveLayer[0] && arcadexr::config::GetInt("m2.hideHud", 0) == 0 && !(skip & 32)) {
            PlanePushConstants frontPc{};
            frontPc.uLift = LiftExponent();
            memcpy(frontPc.uHudMvp, hudMvp.m, sizeof(hudMvp.m));
            if (m_frontFullscreen && !m_flatMode) {
                // full-screen front layer (flash): the arcade plane enlarged over the whole view
                XrMatrix4x4f big = hudToWorld;
                const float k = std::max(1.0f, arcadexr::config::GetFloat("m2.overlayScale", 8.0f));
                for (int c = 0; c < 3; ++c) { big.m[c] *= k; big.m[4 + c] *= k; }
                XrMatrix4x4f bigMvp; XrMatrix4x4f_Multiply(&bigMvp, &viewProjection, &big);
                memcpy(frontPc.uHudMvp, bigMvp.m, sizeof(bigMvp.m));
            }
            frontPc.uOutSize[0] = outW; frontPc.uOutSize[1] = outH;
            frontPc.uKeyZero = 1;
            frontPc.uUvScaleX = m_layerUvScaleX[0];

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipelineLayout, 0, 1, &m_layerDescSet[0], 0, nullptr);
            vkCmdPushConstants(cmd, m_planePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(frontPc), &frontPc);
            // Only the tiles holding HUD texels (see UploadLayers), each run scissored to its projected rectangle.
            // Any corner behind the eye, a full-screen layer, or m2.hudTiles=0: the whole plane as before.
            bool tiled = !m_frontFullscreen && !m_flatMode && !m_frontRuns.empty() && m_frontRunsW > 0 &&
                         arcadexr::config::GetInt("m2.hudTiles", 1) != 0;
            std::vector<VkRect2D> rects;
            if (tiled) {
                const float* M = frontPc.uHudMvp;
                for (const FrontRun& r : m_frontRuns) {
                    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
                    for (int c = 0; c < 4; ++c) {
                        // three arcade texels of margin: the HUD filter reads up to 2 texels around (measured: 1.5 lost pixels)
                        const float px = (c & 1) ? float(r.x1) + 3.0f : float(r.x0) - 3.0f;
                        const float py = (c & 2) ? float(r.y1) + 3.0f : float(r.y0) - 3.0f;
                        const float u = px / float(m_frontRunsW) - 0.5f, v = 0.5f - py / float(m_frontRunsH);
                        const float cx = M[0] * u + M[4] * v + M[12], cy = M[1] * u + M[5] * v + M[13], cw = M[3] * u + M[7] * v + M[15];
                        if (cw <= 1e-4f) { tiled = false; break; }
                        const float sx = (cx / cw * 0.5f + 0.5f) * outW, sy = (cy / cw * 0.5f + 0.5f) * outH;
                        x0 = std::min(x0, sx); x1 = std::max(x1, sx); y0 = std::min(y0, sy); y1 = std::max(y1, sy);
                    }
                    if (!tiled) break;
                    const int ix0 = std::max(0, int(x0) - 2), iy0 = std::max(0, int(y0) - 2);
                    const int ix1 = std::min(int(outW), int(x1) + 3), iy1 = std::min(int(outH), int(y1) + 3);
                    if (ix1 > ix0 && iy1 > iy0) rects.push_back({{ix0, iy0}, {uint32_t(ix1 - ix0), uint32_t(iy1 - iy0)}});
                }
            }
            if (tiled) {
                // The expanded rectangles overlap (neighbouring rows, margins): a pixel drawn twice would blend its
                // anti-aliased HUD edge twice (measured: 4-7 k pixels off by up to 40). Snap them to a grid of
                // disjoint 32 px cells of the eye image and draw each marked row run once.
                const int cs = 32, gw = (int(outW) + cs - 1) / cs, gh = (int(outH) + cs - 1) / cs;
                m_hudCells.assign(size_t(gw) * size_t(gh), 0u);
                for (const VkRect2D& rc : rects) {
                    const int cx0 = rc.offset.x / cs, cy0 = rc.offset.y / cs;
                    const int cx1 = std::min(gw - 1, (rc.offset.x + int(rc.extent.width) - 1) / cs);
                    const int cy1 = std::min(gh - 1, (rc.offset.y + int(rc.extent.height) - 1) / cs);
                    for (int cy = cy0; cy <= cy1; ++cy)
                        for (int cx = cx0; cx <= cx1; ++cx) m_hudCells[size_t(cy) * size_t(gw) + size_t(cx)] = 1u;
                }
                for (int cy = 0; cy < gh; ++cy) {
                    int run = -1;
                    for (int cx = 0; cx <= gw; ++cx) {
                        const bool on = cx < gw && m_hudCells[size_t(cy) * size_t(gw) + size_t(cx)] != 0u;
                        if (on && run < 0) run = cx;
                        if (!on && run >= 0) {
                            const int x0 = run * cs, y0 = cy * cs;
                            const int x1 = std::min(int(outW), cx * cs), y1 = std::min(int(outH), (cy + 1) * cs);
                            const VkRect2D rc{{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
                            m_hudMvRects[eye & 1u].push_back({float(x0) / outW, float(y0) / outH, float(x1) / outW, float(y1) / outH});
                            vkCmdSetScissor(cmd, 0, 1, &rc);
                            vkCmdDraw(cmd, 3, 1, 0, 0);
                            run = -1;
                        }
                    }
                }
                const VkRect2D full{{0, 0}, {uint32_t(outW), uint32_t(outH)}};
                vkCmdSetScissor(cmd, 0, 1, &full);
            } else {
                m_hudMvAll[eye & 1u] = true;   // the whole plane: no vector anywhere under it
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
        }

        static unsigned s_immFrames = 0;
        if ((s_immFrames++ % 120u) == 0u) {
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK rendered eye=%u lean=%d layers=%u lut=%d prims=%zu fast=%u cutEnd=%u opq=%u gls=%u res=%ux%u",
                                             eye, int(arcadexr::config::GetInt("m2.lean", 3)), m_regions.LayerCount(), int(m_lutValid), m_rawPrims.size(), m_fastIndexCount, m_secIndexStart, m_opaqueIndexCount, m_glassIndexCount,
                                             renderAreaExtent.width, renderAreaExtent.height));
        }

        return true;
    }

    // Immersive lightgun (23/09, Virtua Cop): where does a controller ray meet the scene the player SEES, and
    // which board pixel is that? Exact ray/triangle test on the drawn main-view triangles (opaque + cut-outs;
    // mirrors and glass excluded), visible surface = lowest near-to-far rank like the renderer (depth = rank),
    // ray length only breaks ties. No hit (sky, void): the direction is projected at infinity, so pointing off
    // the scene lands off the screen and the game reloads, as on the cabinet. Same thread as BuildFrame.
    bool Aim(const XrVector3f& origin, const XrVector3f& dir, float& nx, float& ny, XrVector3f& hitWorld) const {
        if (!m_aimXf.valid || !m_haveMainView || m_rawIdx.empty()) return false;
        const auto& A = m_aimXf;
        const float rx = origin.x - A.cam.x, ry = origin.y - A.cam.y, rz = origin.z - A.cam.z;
        const auto dotv = [](float x, float y, float z, const arcadexr::gun::Vec3& b) { return x * b.x + y * b.y + z * b.z; };
        const float o[3] = {dotv(rx, ry, rz, A.R) / A.s, dotv(rx, ry, rz, A.U) / A.s, -dotv(rx, ry, rz, A.N) / A.s};
        const float d[3] = {dotv(dir.x, dir.y, dir.z, A.R), dotv(dir.x, dir.y, dir.z, A.U), -dotv(dir.x, dir.y, dir.z, A.N)};
        const float fx = m_aimFocus[0], fy = m_aimFocus[1];   // the focus the immersive image was built with
        const float nearZ = A.nearM / A.s;
        const size_t end = std::min<size_t>(m_secIndexStart, m_rawIdx.size());
        std::uint32_t bestRank = 0xffffffffu;
        float bestT = 0.0f;
        for (size_t i = 0; i + 2 < end; i += 3) {
            const std::uint32_t i0 = m_rawIdx[i], i1 = m_rawIdx[i + 1], i2 = m_rawIdx[i + 2];
            const std::uint32_t rank = m_rawPrimOfVertex[i0];
            if (rank > bestRank) continue;
            if (rank < m_rawPrims.size() && (m_rawPrims[rank].rgb & 0x2000000u) != 0u) continue;   // screen overlay: not in the room
            const float* a = &m_rawVerts[size_t(i0) * 5];
            const float* b = &m_rawVerts[size_t(i1) * 5];
            const float* c = &m_rawVerts[size_t(i2) * 5];
            const float A0[3] = {a[0] / fx, a[1] / fy, a[2]}, B0[3] = {b[0] / fx, b[1] / fy, b[2]}, C0[3] = {c[0] / fx, c[1] / fy, c[2]};
            const float e1[3] = {B0[0] - A0[0], B0[1] - A0[1], B0[2] - A0[2]}, e2[3] = {C0[0] - A0[0], C0[1] - A0[1], C0[2] - A0[2]};
            const float pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
            const float det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
            if (std::fabs(det) < 1e-12f) continue;
            const float inv = 1.0f / det;
            const float tv[3] = {o[0] - A0[0], o[1] - A0[1], o[2] - A0[2]};
            const float u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
            if (u < 0.0f || u > 1.0f) continue;
            const float qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
            const float v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
            if (v < 0.0f || u + v > 1.0f) continue;
            const float t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
            if (t <= 0.0f || o[2] + t * d[2] < nearZ) continue;
            if (rank < bestRank || t < bestT) { bestRank = rank; bestT = t; }
        }
        float P[3];
        m_aimOnPlane = false;
        if (bestRank == 0xffffffffu && m_aimHud.valid) {
            // No 3D of the game under the ray (menus, stage select: the panels are secondary views and overlays
            // on the arcade screen plane): aim on that plane, where they are drawn.
            const auto& H = m_aimHud;
            const float dn = dir.x * H.N.x + dir.y * H.N.y + dir.z * H.N.z;
            if (std::fabs(dn) > 1e-6f) {
                const float t = ((H.C.x - origin.x) * H.N.x + (H.C.y - origin.y) * H.N.y + (H.C.z - origin.z) * H.N.z) / dn;
                if (t > 0.0f) {
                    const XrVector3f Q{origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t};
                    const float u = ((Q.x - H.C.x) * H.R.x + (Q.y - H.C.y) * H.R.y + (Q.z - H.C.z) * H.R.z) / H.w;
                    const float v = ((Q.x - H.C.x) * H.U.x + (Q.y - H.C.y) * H.U.y + (Q.z - H.C.z) * H.U.z) / H.h;
                    nx = u + 0.5f;
                    ny = 0.5f - v;
                    hitWorld = Q;
                    m_aimOnPlane = true;
                    return true;
                }
            }
        }
        if (bestRank != 0xffffffffu) {
            P[0] = o[0] + bestT * d[0]; P[1] = o[1] + bestT * d[1]; P[2] = o[2] + bestT * d[2];
        } else {
            if (d[2] <= 1e-4f) return false;   // pointing behind the board camera: not on this screen
            const float far = 1.0e5f;           // "at infinity": only the direction matters for the pixel
            P[0] = o[0] + far * d[0]; P[1] = o[1] + far * d[1]; P[2] = o[2] + far * d[2];
        }
        const float sx = m_crtc[0] + float(m_mainCenter[0]) + fx * P[0] / P[2];
        const float sy = (384.0f - float(m_mainCenter[1])) + m_crtc[1] - fy * P[1] / P[2];
        nx = sx / 496.0f;
        ny = sy / 384.0f;
        const float hs = (bestRank != 0xffffffffu) ? A.s : 0.0f;
        hitWorld = (bestRank != 0xffffffffu)
            ? XrVector3f{A.cam.x + (A.R.x * P[0] + A.U.x * P[1] - A.N.x * P[2]) * hs,
                         A.cam.y + (A.R.y * P[0] + A.U.y * P[1] - A.N.y * P[2]) * hs,
                         A.cam.z + (A.R.z * P[0] + A.U.z * P[1] - A.N.z * P[2]) * hs}
            : XrVector3f{origin.x + dir.x * 50.0f, origin.y + dir.y * 50.0f, origin.z + dir.z * 50.0f};
        return true;
    }

    // Self-test (m2.aimTest=1, TCVR_AIMTEST m2 every ~2 s): the centroid of drawn triangles is placed in the room
    // as the renderer places it, a ray is cast at it, and Aim must return the pixel the board draws it at.
    // From the eye this checks the mapping; from a hand-like origin a nearer surface on the ray is not an error.
    void AimSelfTest() {
        if (arcadexr::config::GetInt("m2.aimTest", 0) == 0 || !m_aimXf.valid || !m_haveMainView) return;
        if ((++m_aimTestTick % 240u) != 0u) return;
        const auto& A = m_aimXf;
        const float fx = m_aimFocus[0], fy = m_aimFocus[1];
        const size_t end = std::min<size_t>(m_fastIndexCount, m_rawIdx.size());
        const size_t tris = end / 3;
        if (tris == 0) return;
        struct Org { const char* name; XrVector3f o; };
        const Org orgs[2] = {{"oeil", {A.cam.x, A.cam.y, A.cam.z}},
                             {"main", {A.cam.x + A.R.x * 0.15f - A.U.x * 0.35f - A.N.x * 0.30f, A.cam.y + A.R.y * 0.15f - A.U.y * 0.35f - A.N.y * 0.30f,
                                       A.cam.z + A.R.z * 0.15f - A.U.z * 0.35f - A.N.z * 0.30f}}};
        for (const Org& org : orgs) {
            std::vector<float> errs;
            int misses = 0, occluded = 0;
            const size_t step = std::max<size_t>(1, tris / 100);
            for (size_t tri = 0; tri < tris; tri += step) {
                float P[3] = {0, 0, 0};
                for (int k = 0; k < 3; ++k) {
                    const float* v = &m_rawVerts[size_t(m_rawIdx[tri * 3 + k]) * 5];
                    P[0] += v[0] / fx / 3.0f; P[1] += v[1] / fy / 3.0f; P[2] += v[2] / 3.0f;
                }
                if (P[2] < A.nearM / A.s) continue;
                const float ex = m_crtc[0] + float(m_mainCenter[0]) + fx * P[0] / P[2];
                const float ey = (384.0f - float(m_mainCenter[1])) + m_crtc[1] - fy * P[1] / P[2];
                if (ex < 0 || ex > 496 || ey < 0 || ey > 384) continue;
                const XrVector3f W{A.cam.x + (A.R.x * P[0] + A.U.x * P[1] - A.N.x * P[2]) * A.s,
                                   A.cam.y + (A.R.y * P[0] + A.U.y * P[1] - A.N.y * P[2]) * A.s,
                                   A.cam.z + (A.R.z * P[0] + A.U.z * P[1] - A.N.z * P[2]) * A.s};
                XrVector3f dir{W.x - org.o.x, W.y - org.o.y, W.z - org.o.z};
                const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                if (len < 1e-6f) continue;
                dir = {dir.x / len, dir.y / len, dir.z / len};
                float nx = 0, ny = 0; XrVector3f hw{};
                if (!Aim(org.o, dir, nx, ny, hw)) { ++misses; continue; }
                const float e = std::hypot(nx * 496.0f - ex, ny * 384.0f - ey);
                const float hd = std::sqrt((hw.x - org.o.x) * (hw.x - org.o.x) + (hw.y - org.o.y) * (hw.y - org.o.y) + (hw.z - org.o.z) * (hw.z - org.o.z));
                if (e > 1.5f && hd < 0.97f * len) { ++occluded; continue; }
                errs.push_back(e);
            }
            std::sort(errs.begin(), errs.end());
            auto q = [&](float f) { return errs.empty() ? -1.0f : errs[std::min(errs.size() - 1, size_t(f * float(errs.size())))]; };
            __android_log_print(ANDROID_LOG_INFO, "hello_xr", "TCVR_AIMTEST m2 %s: %zu points, erreur px mediane %.2f p90 %.2f max %.2f | masques par plus pres %d, rates %d",
                                org.name, errs.size(), q(0.5f), q(0.9f), errs.empty() ? -1.0f : errs.back(), occluded, misses);
        }
    }

    bool HaveMainView() const { return m_haveMainView; }
    VkImageView FlatView() const { return m_flat.view; }
    VkImage FlatImage() const { return m_flat.image; }
    VkFormat FlatFormat() const { return m_colorFormat; }
    uint32_t FlatWidth() const { return m_flat.w; }
    uint32_t FlatHeight() const { return m_flat.h; }

    // SCREEN presentation of a Model 2 game, drawn by the GPU: the recorded scene through the BOARD's own
    // projection into a k x (496x384) image (k = m2.gpuRaster, default 4), MSAA 4x, the same texture
    // regions and lean shaders as the immersive view, then mipmapped (it is minified on the virtual screen).
    // Lets MAME stop rasterising (scene mode 2) in screen mode too. Record outside any render pass.
    bool RenderFlat(VkCommandBuffer cmd) {
        if (!m_initialized || m_opaqueIndexCount == 0) return false;
        const int k = std::max(1, std::min(8, arcadexr::config::GetInt("m2.gpuRaster", 4)));
        const uint32_t W = 496u * uint32_t(k), H = 384u * uint32_t(k);
        if (m_flat.w != W || m_flat.h != H) {
            vkDeviceWaitIdle(m_vkDevice);
            if (m_flat.view) vkDestroyImageView(m_vkDevice, m_flat.view, nullptr);
            if (m_flat.image) vkDestroyImage(m_vkDevice, m_flat.image, nullptr);
            if (m_flat.mem) vkFreeMemory(m_vkDevice, m_flat.mem, nullptr);
            for (auto& f : m_fbs) if (f.eye == 2) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); f.fb = VK_NULL_HANDLE; }
            m_fbs.erase(std::remove_if(m_fbs.begin(), m_fbs.end(), [](const FbEntry& f) { return f.fb == VK_NULL_HANDLE; }), m_fbs.end());
            m_flat = FlatTarget{};
            m_flat.w = W; m_flat.h = H;
            uint32_t levels = 1; while ((std::max(W, H) >> levels) > 0) ++levels;
            m_flat.levels = levels;
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D; ii.format = m_colorFormat; ii.extent = {W, H, 1};
            ii.mipLevels = levels; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &ii, nullptr, &m_flat.image));
            VkMemoryRequirements req{}; vkGetImageMemoryRequirements(m_vkDevice, m_flat.image, &req);
            m_memAllocator->Allocate(req, &m_flat.mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, m_flat.image, m_flat.mem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = m_flat.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = m_colorFormat;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &vi, nullptr, &m_flat.view));
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK flat target %ux%u (x%d), %u mips", W, H, k, levels));
        }
        // Board projection, clip w = camera z (so the GPU clips behind the eye and interpolates in perspective):
        //   x_ndc = (crtc_x + cx + focus_x * X / z) / 248 - 1 ;  y_ndc = ((384 - cy) + crtc_y - focus_y * Y / z) / 192 - 1
        const float fx = (m_m2FocusX > 1.0f) ? m_m2FocusX : 512.0f, fy = (m_m2FocusY > 1.0f) ? m_m2FocusY : 512.0f;
        const float cx = float(m_mainCenter[0]), cy = float(m_mainCenter[1]);
        m_flatMvp = XrMatrix4x4f{};
        m_flatMvp.m[0] = fx / 248.0f;                                     // row 0 . X
        m_flatMvp.m[8] = (m_crtc[0] + cx - 248.0f) / 248.0f;              // row 0 . z
        m_flatMvp.m[5] = -fy / 192.0f;                                    // row 1 . Y
        m_flatMvp.m[9] = ((384.0f - cy) + m_crtc[1] - 192.0f) / 192.0f;   // row 1 . z
        m_flatMvp.m[11] = 1.0f;                                           // row 3 . z  (w = z)
        const VkRect2D area{{0, 0}, {W, H}};
        const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        BeginPass(cmd, 2, m_flat.image, {W, H}, area, black);
        VkViewport vp{0.0f, 0.0f, float(W), float(H), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &area);
        XrCompositionLayerProjectionView lv{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        lv.pose.orientation.w = 1.0f;
        lv.fov = {-0.8f, 0.8f, 0.8f, -0.8f};
        m_flatMode = true;
        const bool drawn = RenderImmersive(0, lv, cmd, {W, H});
        m_flatMode = false;
        vkCmdEndRenderPass(cmd);
        // Mip chain (the 4x image is minified on the virtual screen: without mips it shimmers).
        auto bar = [&](uint32_t lvl, uint32_t cnt, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da,
                       VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = sa; b.dstAccessMask = da; b.oldLayout = from; b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = m_flat.image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, lvl, cnt, 0, 1};
            vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        bar(0, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (m_flat.levels > 1)
            bar(1, m_flat.levels - 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        int32_t w = int32_t(W), h = int32_t(H);
        for (uint32_t l = 1; l < m_flat.levels; ++l) {
            VkImageBlit bl{};
            bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, 1};
            bl.srcOffsets[1] = {w, h, 1};
            const int32_t nw = std::max(1, w / 2), nh = std::max(1, h / 2);
            bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            bl.dstOffsets[1] = {nw, nh, 1};
            vkCmdBlitImage(cmd, m_flat.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_flat.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
            bar(l, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            w = nw; h = nh;
        }
        bar(0, m_flat.levels, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        return drawn;
    }
    // The LEAN shader assumes filter mode 5 + texture array + no edge fade + stage 0.
    bool ubo_texArray(uint32_t eye) const {
        const M2UniformBufferObject& u = *m_uboMappedF[m_fs][eye < 2 ? eye : 0][0];
        return u.uFilterMode == 5 && (u.padEnd[2] & 2) != 0 && (u.uTestStage == 0 || u.uTestStage >= 9);
    }
    void SetFrameSlot(int s) { m_fs = s & 1; m_regions.SetHalf(uint32_t(m_fs)); }

    // Does this polygon's texture (level 0 region) contain the transparent index 15 at all?
    // A "translucent" polygon whose texture has no hole can never discard a pixel, so it is
    // drawn with the discard-free pipeline. Scanned once per region, texture RAM is static in a race.
    bool RegionHasHoles(const tcvr_m2_prim& p) {
        if (p.textured == 0u || m_sheetCpu[0].empty()) return true;
        const uint32_t w = p.texwidth, h = p.texheight;
        if (w == 0 || h == 0 || w > 2048 || h > 1024) return true;
        const uint32_t ox = (p.texx - 2048u) & 2047u, oy = (p.texy - 1024u) & 1023u, sh = p.texsheet & 1u;
        const uint64_t key = (uint64_t(sh) << 63) | (uint64_t(ox) << 40) | (uint64_t(oy) << 24) | (uint64_t(w) << 12) | uint64_t(h);
        auto it = m_regionHasHoles.find(key);
        if (it != m_regionHasHoles.end()) return it->second;
        const std::vector<uint8_t>& cpu = m_sheetCpu[sh];
        bool holes = false;
        for (uint32_t y = 0; y < h && !holes; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                int x2 = int(ox + x), y2 = int(oy + y);
                if (x2 >= 1024) { x2 -= 1024; y2 ^= 1024; }
                const size_t at = size_t(y2) * 1024 + size_t(x2);
                if (at >= cpu.size() || cpu[at] == 15) { holes = true; break; }
            }
        m_regionHasHoles.emplace(key, holes);
        return holes;
    }
    // Ground colour (23/09, Virtua Cop): below its horizon the arcade always shows its 3D floor, never the 2D
    // layer behind it. In immersive the floor the game draws ends where its camera stops seeing it, and the 2D
    // layer showed through (a flat plum area in the desert stage). The ground there takes the colour of the
    // floor the ARCADE shows at the bottom centre of its screen: the frontmost main-view polygon under that pixel,
    // averaged through the Model 2 colour chain on the CPU (texel -> lumaram -> palette -> colorxlat -> gamma).
    // Scene depth where the player looks (25/09, HUD ISO): rays from the board camera through a grid of pixels in the
    // centre / lower centre of the main view (the car in a chase view, the road ahead), tested against the drawn
    // triangles exactly like GroundProbe; the median of the nearest hits, smoothed, in board units along the view
    // axis. The HUD is then laid at that depth ON the board camera's rays: it covers exactly what it covers on the
    // cabinet, and the eyes converge where they already look instead of on a plane 2 m away.
    void SceneDepthProbe(const tcvr_m2_frame& frame, int32_t mainCx, int32_t mainCy) {
        if (!m_haveMainView || (++m_depthProbeTick & 3u) != 0u) return;
        const auto tProbe = std::chrono::steady_clock::now();
        struct ProbeTimer { std::chrono::steady_clock::time_point t0; float* worst; ~ProbeTimer() {
            const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ms > *worst) *worst = ms; } } probeTimer{tProbe, &m_probeWorstMs};
        const float fx = float(frame.crtc_xoffset + mainCx), fy = float((384 - mainCy) + frame.crtc_yoffset);
        const float ffx = (m_m2FocusX > 1.0f) ? m_m2FocusX : 512.0f, ffy = (m_m2FocusY > 1.0f) ? m_m2FocusY : 512.0f;
        const size_t end = std::min<size_t>(m_secIndexStart, m_rawIdx.size());
        // Triangles outer, rays inner: each triangle's screen box selects the few grid rays that can hit it; the
        // exact ray test runs only for those (a triangle crossing the camera plane has no box: all rays). The
        // rays-outer version cost 3.4 ms median, 8 ms worst, every 4th frame: missed refreshes (25/09).
        constexpr int kGX = 9, kGY = 5;
        constexpr float kX0 = 150.0f, kDX = 24.5f, kY0 = 170.0f, kDY = 40.0f;
        float best[kGX * kGY];
        for (float& b : best) b = 1e30f;
        auto rayTest = [&](const float* A, const float* e1, const float* e2, int gx, int gy) {
            const float d[3] = {(kX0 + kDX * float(gx) - fx) / ffx, (fy - (kY0 + kDY * float(gy))) / ffy, 1.0f};
            const float pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
            const float det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
            if (std::fabs(det) < 1e-12f) return;
            const float inv = 1.0f / det;
            const float tv[3] = {-A[0], -A[1], -A[2]};
            const float u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
            if (u < 0.0f || u > 1.0f) return;
            const float qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
            const float v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
            if (v < 0.0f || u + v > 1.0f) return;
            const float t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
            float& bb = best[gy * kGX + gx];
            if (t > 0.05f && t < bb) bb = t;
        };
        for (size_t i = 0; i + 2 < end; i += 3) {
            const uint32_t rank = m_rawPrimOfVertex[m_rawIdx[i]];
            if (rank < m_rawPrims.size() && (m_rawPrims[rank].rgb & 0x2000000u) != 0u) continue;   // screen overlay
            const float* a = &m_rawVerts[size_t(m_rawIdx[i]) * 5];
            const float* b = &m_rawVerts[size_t(m_rawIdx[i + 1]) * 5];
            const float* c = &m_rawVerts[size_t(m_rawIdx[i + 2]) * 5];
            int gx0 = 0, gx1 = kGX - 1, gy0 = 0, gy1 = kGY - 1;
            if (a[2] > 0.05f && b[2] > 0.05f && c[2] > 0.05f) {
                const float ax = fx + a[0] / a[2], bx = fx + b[0] / b[2], cx = fx + c[0] / c[2];
                const float ay = fy - a[1] / a[2], by = fy - b[1] / b[2], cy = fy - c[1] / c[2];
                const float minx = std::min(ax, std::min(bx, cx)), maxx = std::max(ax, std::max(bx, cx));
                const float miny = std::min(ay, std::min(by, cy)), maxy = std::max(ay, std::max(by, cy));
                gx0 = std::max(0, int(std::ceil((minx - kX0) / kDX))); gx1 = std::min(kGX - 1, int(std::floor((maxx - kX0) / kDX)));
                gy0 = std::max(0, int(std::ceil((miny - kY0) / kDY))); gy1 = std::min(kGY - 1, int(std::floor((maxy - kY0) / kDY)));
                if (gx0 > gx1 || gy0 > gy1) continue;
            }
            const float A[3] = {a[0] / ffx, a[1] / ffy, a[2]};
            const float e1[3] = {b[0] / ffx - A[0], b[1] / ffy - A[1], b[2] - A[2]};
            const float e2[3] = {c[0] / ffx - A[0], c[1] / ffy - A[1], c[2] - A[2]};
            for (int gy = gy0; gy <= gy1; ++gy)
                for (int gx = gx0; gx <= gx1; ++gx) rayTest(A, e1, e2, gx, gy);
        }
        float hits[kGX * kGY]; int nh = 0;
        for (float bb : best) if (bb < 1e29f) hits[nh++] = bb;
        if (nh < 5) return;
        std::nth_element(hits, hits + nh / 2, hits + nh);
        const float med = hits[nh / 2];
        m_sceneDepth = (m_sceneDepth > 0.0f) ? m_sceneDepth + (med - m_sceneDepth) * 0.1f : med;
        if ((++m_depthLogTick % 60u) == 0u)
        {
            Log::Write(Log::Level::Info, Fmt("TCVR_DEPTH scene median %.2f units (smoothed %.2f), %d/45 rays hit, probe worst %.2f ms over %zu tris",
                                             med, m_sceneDepth, nh, m_probeWorstMs, end / 3));
            m_probeWorstMs = 0.0f;
        }
    }
    // Menu LUMINOSITE (profile immersive.lift, a display gamma >= 1): the shaders raise colours to 1/gamma. The
    // flat oracle target keeps the exact colours.
    float LiftExponent() const {
        if (m_flatMode) return 1.0f;
        const float g = std::max(1.0f, std::min(2.0f, arcadexr::profiles::GetFloat("immersive.lift", 1.0f)));
        return 1.0f / g;
    }
    float m_sceneDepth = 0.0f;
    float m_probeWorstMs = 0.0f;
    struct FrontRun { uint16_t x0, y0, x1, y1; };
    std::vector<FrontRun> m_frontRuns;
    std::vector<uint8_t> m_hudCells;
    uint32_t m_frontRunsW = 0, m_frontRunsH = 0;
    uint32_t m_depthProbeTick = 0, m_depthLogTick = 0;

    void GroundProbe(const tcvr_m2_frame& frame, int32_t mainCx, int32_t mainCy, int32_t mainB) {
        if (!m_haveMainView || !frame.palram || !frame.colorxlat || !frame.lumaram || !frame.gamma) return;
        const float fx = float(frame.crtc_xoffset + mainCx), fy = float((384 - mainCy) + frame.crtc_yoffset);
        const float px = 248.0f, py = float(std::max(0, mainB - 12));
        const size_t end = std::min<size_t>(m_secIndexStart, m_rawIdx.size());
        uint32_t best = 0xffffffffu;
        const bool diag = arcadexr::config::GetInt("m2.groundDiag", 0) != 0 && (m_groundLogTick % 120u) == 119u;
        // A ray from the board camera through that pixel (metric camera space: X = x/focus), tested exactly
        // against the drawn triangles: polygons that pass behind the camera -- the nearest floor always does --
        // are handled, where a projected 2D test had to skip them and found a far slab under the real floor.
        const float ffx = (m_m2FocusX > 1.0f) ? m_m2FocusX : 512.0f, ffy = (m_m2FocusY > 1.0f) ? m_m2FocusY : 512.0f;
        const float d[3] = {(px - fx) / ffx, (fy - py) / ffy, 1.0f};
        for (size_t i = 0; i + 2 < end; i += 3) {
            const uint32_t rank = m_rawPrimOfVertex[m_rawIdx[i]];
            if (rank >= best && !diag) continue;
            if (rank < m_rawPrims.size() && (m_rawPrims[rank].rgb & 0x2000000u) != 0u) continue;   // screen overlay
            const float* a = &m_rawVerts[size_t(m_rawIdx[i]) * 5];
            const float* b = &m_rawVerts[size_t(m_rawIdx[i + 1]) * 5];
            const float* c = &m_rawVerts[size_t(m_rawIdx[i + 2]) * 5];
            const float A[3] = {a[0] / ffx, a[1] / ffy, a[2]};
            const float e1[3] = {b[0] / ffx - A[0], b[1] / ffy - A[1], b[2] - A[2]};
            const float e2[3] = {c[0] / ffx - A[0], c[1] / ffy - A[1], c[2] - A[2]};
            // floor only: nearly horizontal
            const float nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
            const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len < 1e-12f || std::fabs(ny) < 0.8f * len) continue;
            const float pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
            const float det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
            if (std::fabs(det) < 1e-12f) continue;
            const float inv = 1.0f / det;
            const float tv[3] = {-A[0], -A[1], -A[2]};
            const float u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
            if (u < 0.0f || u > 1.0f) continue;
            const float qv[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
            const float v = (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]) * inv;
            if (v < 0.0f || u + v > 1.0f) continue;
            const float t = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
            if (t <= 0.0f) continue;
            if (diag && rank < m_rawPrims.size()) {
                float cc[3]; const int nn = GroundColourOf(frame, m_rawPrims[rank], cc);
                const tcvr_m2_prim& q = m_rawPrims[rank];
                Log::Write(Log::Level::Info, Fmt("TCVR_GROUNDDIAG rank=%u z=%.1f tex=%u xy=%u,%u wh=%ux%u cb=%u n=%d -> %.2f %.2f %.2f",
                    rank, t, q.textured, q.texx, q.texy, q.texwidth, q.texheight, q.colorbase, nn, cc[0], cc[1], cc[2]));
            }
            if (rank < best) best = rank;
        }
        if (best == 0xffffffffu || best >= m_rawPrims.size()) return;
        {   // Camera height above that floor (world scale, 23/09): distance from the camera to the floor's PLANE,
            // in the metric space the immersive image is built in (capped focus) -- independent of the pitch.
            const float fe0 = m_aimFocus[0] > 1.0f ? m_aimFocus[0] : ffx, fe1 = m_aimFocus[1] > 1.0f ? m_aimFocus[1] : ffy;
            for (size_t i = 0; i + 2 < end; i += 3) {
                if (m_rawPrimOfVertex[m_rawIdx[i]] != best) continue;
                const float* a = &m_rawVerts[size_t(m_rawIdx[i]) * 5];
                const float* b = &m_rawVerts[size_t(m_rawIdx[i + 1]) * 5];
                const float* c = &m_rawVerts[size_t(m_rawIdx[i + 2]) * 5];
                const float A[3] = {a[0] / fe0, a[1] / fe1, a[2]};
                const float e1[3] = {b[0] / fe0 - A[0], b[1] / fe1 - A[1], b[2] - A[2]};
                const float e2[3] = {c[0] / fe0 - A[0], c[1] / fe1 - A[1], c[2] - A[2]};
                const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                if (len < 1e-12f) continue;
                const float h = std::fabs(n[0] * A[0] + n[1] * A[1] + n[2] * A[2]) / len;
                if (h > 1e-3f) {
                    m_camHeightUnits = (m_camHeightUnits > 0.0f) ? m_camHeightUnits + (h - m_camHeightUnits) * 0.05f : h;
                }
                break;
            }
        }
        const tcvr_m2_prim& p = m_rawPrims[best];
        float acc[3] = {0, 0, 0};
        const int n = GroundColourOf(frame, p, acc);
        if (n == 0) return;
        const float a = m_groundProbed ? 0.15f : 1.0f;   // settle quickly, then follow the stage smoothly
        for (int k = 0; k < 3; ++k) m_groundProbe[k] += (acc[k] - m_groundProbe[k]) * a;
        m_groundProbed = true;
        if ((++m_groundLogTick % 120u) == 0u)
            Log::Write(Log::Level::Info, Fmt("TCVR_GROUND rank=%u/%zu tex=%u trans=%u sheet=%u xy=%u,%u wh=%ux%u cb=%u lb=%u luma=%u n=%d -> %.2f %.2f %.2f",
                best, m_rawPrims.size(), p.textured, p.translucent, p.texsheet, p.texx, p.texy, p.texwidth, p.texheight, p.colorbase, p.lumabase, p.luma, n,
                m_groundProbe[0], m_groundProbe[1], m_groundProbe[2]));
    }
    unsigned m_groundLogTick = 0;
    unsigned m_overlayDiagTick = 0;
    bool m_overlayOn = false;
    // Average display colour of a polygon through the Model 2 colour chain (CPU), 0 if it cannot be computed.
    int GroundColourOf(const tcvr_m2_frame& frame, const tcvr_m2_prim& p, float acc[3]) {
        auto u16 = [](const uint16_t* t, uint32_t n, uint32_t i) -> uint32_t { return i < n ? t[i] : 0u; };
        auto shadeCpu = [&](uint32_t luma, float out[3]) {
            const uint32_t color = u16(frame.palram, frame.palram_entries, p.colorbase + 0x1000u) & 0x7fffu;
            const uint32_t c[3] = {color & 0x1fu, (color >> 5) & 0x1fu, (color >> 10) & 0x1fu};
            const uint32_t base[3] = {0x0000u / 2u, 0x4000u / 2u, 0x8000u / 2u};
            for (int k = 0; k < 3; ++k) {
                const uint32_t x = u16(frame.colorxlat, frame.colorxlat_entries, base[k] + (c[k] << 8) + luma) & 0xffu;
                out[k] = float(x < frame.gamma_entries ? frame.gamma[x] : x) / 255.0f;
            }
        };
        acc[0] = acc[1] = acc[2] = 0.0f; int n = 0;
        if ((p.rgb & 0x1000000u) != 0u) {
            acc[0] = float((p.rgb >> 16) & 0xffu) / 255.0f; acc[1] = float((p.rgb >> 8) & 0xffu) / 255.0f; acc[2] = float(p.rgb & 0xffu) / 255.0f; n = 1;
        } else if (p.textured == 0u) {
            shadeCpu(std::min(p.luma >> 2, 0x3fu), acc); n = 1;
        } else {
            const uint32_t sh = p.texsheet & 1u, w = p.texwidth, h = p.texheight;
            if (m_sheetCpu[sh].empty() || w == 0 || h == 0 || w > 2048 || h > 1024) return 0;
            const uint32_t ox = (p.texx - 2048u) & 2047u, oy = (p.texy - 1024u) & 1023u;
            const std::vector<uint8_t>& cpu = m_sheetCpu[sh];
            for (uint32_t gy = 0; gy < 16; ++gy)
                for (uint32_t gx = 0; gx < 16; ++gx) {
                    int x2 = int(ox + (gx * w) / 16), y2 = int(oy + (gy * h) / 16);
                    if (x2 >= 1024) { x2 -= 1024; y2 ^= 1024; }
                    const size_t at = size_t(y2) * 1024 + size_t(x2);
                    if (at >= cpu.size()) continue;
                    const uint32_t texel = cpu[at];
                    if (p.translucent != 0u && texel == 15u) continue;
                    const uint32_t t8 = texel << 4;
                    const uint32_t li = p.lumabase + (t8 >> 1);
                    const uint32_t lr = li < frame.lumaram_entries ? frame.lumaram[li] : 0u;
                    float c[3]; shadeCpu(std::min((lr * p.luma) / 256u, 0x3fu), c);
                    acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]; ++n;
                }
        }
        if (n > 1) for (int k = 0; k < 3; ++k) acc[k] /= float(n);
        if (arcadexr::config::GetInt("m2.groundDiag", 0) != 0 && p.textured != 0u && (m_groundLogTick % 120u) == 119u) {
            std::string line; int hist[16] = {};
            const uint32_t sh = p.texsheet & 1u, ox = (p.texx - 2048u) & 2047u, oy = (p.texy - 1024u) & 1023u;
            for (uint32_t y = 0; y < p.texheight; ++y) for (uint32_t x = 0; x < p.texwidth; ++x) {
                int x2 = int(ox + x), y2 = int(oy + y); if (x2 >= 1024) { x2 -= 1024; y2 ^= 1024; }
                const size_t at = size_t(y2) * 1024 + size_t(x2); if (at < m_sheetCpu[sh].size()) hist[m_sheetCpu[sh][at] & 15]++;
            }
            for (uint32_t t = 0; t < 16; ++t) {
                const uint32_t li = p.lumabase + ((t << 4) >> 1);
                const uint32_t lr = li < frame.lumaram_entries ? frame.lumaram[li] : 0u;
                float c[3]; shadeCpu(std::min((lr * p.luma) / 256u, 0x3fu), c);
                line += Fmt(" %u:%d(%02x%02x%02x)", t, hist[t], int(c[0] * 255), int(c[1] * 255), int(c[2] * 255));
            }
            Log::Write(Log::Level::Info, "TCVR_GROUNDTEX" + line);
        }
        return n;
    }
    float m_groundProbe[3] = {0.18f, 0.16f, 0.14f};
    float m_camHeightUnits = 0.0f;   // game camera height above its floor, in immersive metric units (smoothed)
    bool m_groundProbed = false;

    bool HasGeometry() const { return m_initialized && m_opaqueIndexCount > 0; }
    bool IsMsaa() const { return m_samples != VK_SAMPLE_COUNT_1_BIT; }

    // Begin the immersive render pass on swapchain image `target` (eye `eye`).
    void BeginPass(VkCommandBuffer cmd, uint32_t eye, VkImage target, VkExtent2D ext, const VkRect2D& area, const float clear[4],
                   VkImage fdmImage = VK_NULL_HANDLE, VkExtent2D fdmExt = {0, 0}, VkImage depthTarget = VK_NULL_HANDLE) {
        eye = eye < 3 ? eye : 0;
        if (!m_depthResolveOn) depthTarget = VK_NULL_HANDLE;
        EyeTargets& et = m_eyeTargets[eye];
        if (et.ext.width != ext.width || et.ext.height != ext.height) DestroyEyeTargets(et), CreateEyeTargets(et, ext);
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_fbs) if (f.image == target && f.eye == eye && f.depthImage == depthTarget) fb = f.fb;
        if (fb == VK_NULL_HANDLE) {
            FbEntry e{};
            e.image = target; e.eye = eye; e.depthImage = depthTarget;
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = target; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = m_colorFormat;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &vi, nullptr, &e.view));
            std::array<VkImageView, 5> att{};
            uint32_t n = 0;
            if (IsMsaa()) { att[n++] = et.colorView; att[n++] = et.depthView; att[n++] = e.view; }
            else { att[n++] = e.view; att[n++] = et.depthView; }
            if (m_depthResolveOn) {
                // The headset's depth image: the MSAA depth resolved into it on-chip (sample 0), for AppSW (26/09).
                VkImageViewCreateInfo dv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                dv.image = depthTarget; dv.viewType = VK_IMAGE_VIEW_TYPE_2D; dv.format = kDepthFormat;
                dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &dv, nullptr, &e.depthResolveView));
                att[n++] = e.depthResolveView;
            }
            if (m_useFdm) {
                // The runtime's density map for this image: the periphery is shaded at a lower rate.
                VkImageViewCreateInfo fv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                fv.image = fdmImage; fv.viewType = VK_IMAGE_VIEW_TYPE_2D; fv.format = VK_FORMAT_R8G8_UNORM;
                fv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                if (fdmImage == VK_NULL_HANDLE || vkCreateImageView(m_vkDevice, &fv, nullptr, &e.fdmView) != VK_SUCCESS) {
                    Log::Write(Log::Level::Error, "TCVR_M2VK FDM view missing -> foveation off for this image");
                } else {
                    att[n++] = e.fdmView;
                    static bool s_logged = false;
                    if (!s_logged) { s_logged = true; Log::Write(Log::Level::Info, Fmt("TCVR_M2VK foveation: density map %ux%u", fdmExt.width, fdmExt.height)); }
                }
            }
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_pass; fi.attachmentCount = n; fi.pAttachments = att.data();
            fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_vkDevice, &fi, nullptr, &e.fb));
            m_fbs.push_back(e);
            fb = e.fb;
        }
        std::array<VkClearValue, 3> cv{};
        for (int i = 0; i < 4; i++) cv[0].color.float32[i] = clear[i];
        cv[0].color.float32[3] = 1.0f;
        cv[1].depthStencil = {1.0f, 0};
        cv[2] = cv[0];
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_pass; bi.framebuffer = fb; bi.renderArea = area;
        bi.clearValueCount = IsMsaa() ? 3u : 2u; bi.pClearValues = cv.data();
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    // AppSW (26/09): the same pass with the MSAA depth resolved into the headset's depth image (VK_KHR_depth_stencil_resolve
    // through VK_KHR_create_renderpass2), only when AppSW is on: otherwise the pass is exactly the previous one.
    bool CreateRenderPassDepthResolve() {
        auto create2 = reinterpret_cast<PFN_vkCreateRenderPass2KHR>(vkGetDeviceProcAddr(m_vkDevice, "vkCreateRenderPass2KHR"));
        if (!create2 || !IsMsaa()) {
            Log::Write(Log::Level::Warning, Fmt("TCVR_M2VK depth resolve unavailable: vkCreateRenderPass2KHR=%p msaa=%d", (void*)create2, int(IsMsaa())));
            return false;
        }
        std::array<VkAttachmentDescription2, 5> at{};
        for (auto& a : at) { a.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2; a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; }
        at[0].format = m_colorFormat; at[0].samples = m_samples;
        at[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[0].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        at[1].format = kDepthFormat; at[1].samples = m_samples;
        at[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        at[2].format = m_colorFormat; at[2].samples = VK_SAMPLE_COUNT_1_BIT;
        at[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[2].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        at[3].format = kDepthFormat; at[3].samples = VK_SAMPLE_COUNT_1_BIT;   // the headset's depth image
        at[3].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        const uint32_t fdmIndex = 4u;
        at[4].format = VK_FORMAT_R8G8_UNORM; at[4].samples = VK_SAMPLE_COUNT_1_BIT;
        at[4].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[4].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[4].initialLayout = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
        at[4].finalLayout = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
        VkAttachmentReference2 colorRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
        VkAttachmentReference2 depthRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT};
        VkAttachmentReference2 resolveRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
        VkAttachmentReference2 depthResolveRef{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT};
        VkSubpassDescriptionDepthStencilResolve dsr{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE};
        dsr.depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT; dsr.stencilResolveMode = VK_RESOLVE_MODE_NONE;
        dsr.pDepthStencilResolveAttachment = &depthResolveRef;
        VkSubpassDescription2 sp{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
        sp.pNext = &dsr;
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = &colorRef;
        sp.pResolveAttachments = &resolveRef;
        sp.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency2 dep{VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassFragmentDensityMapCreateInfoEXT fdmInfo{VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT};
        fdmInfo.fragmentDensityMapAttachment = {fdmIndex, VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT};
        VkRenderPassCreateInfo2 ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
        ri.attachmentCount = 4u + (m_useFdm ? 1u : 0u); ri.pAttachments = at.data();
        if (m_useFdm) ri.pNext = &fdmInfo;
        ri.subpassCount = 1; ri.pSubpasses = &sp;
        ri.dependencyCount = 1; ri.pDependencies = &dep;
        const VkResult res = create2(m_vkDevice, &ri, nullptr, &m_pass);
        if (res != VK_SUCCESS) {
            Log::Write(Log::Level::Warning, Fmt("TCVR_M2VK depth resolve render pass failed: VkResult %d", int(res)));
            m_pass = VK_NULL_HANDLE;
            return false;
        }
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK render pass: MSAA x%d, depth resolved into the headset's depth image (AppSW), foveation %s",
                                         int(m_samples), m_useFdm ? "ON" : "off"));
        return true;
    }

    void CreateRenderPass() {
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK CreateRenderPass depthResolveWanted=%d", int(m_depthResolveWanted)));
        m_depthResolveOn = m_depthResolveWanted && CreateRenderPassDepthResolve();
        if (m_depthResolveOn) return;
        std::array<VkAttachmentDescription, 3> at{};
        const bool ms = IsMsaa();
        // 0: colour (MSAA transient, or the swapchain itself at 1x)
        at[0].format = m_colorFormat; at[0].samples = m_samples;
        at[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        at[0].storeOp = ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
        at[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        // 1: depth, transient, never stored
        at[1].format = kDepthFormat; at[1].samples = m_samples;
        at[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        // 2: resolve target = the swapchain image (MSAA only)
        at[2].format = m_colorFormat; at[2].samples = VK_SAMPLE_COUNT_1_BIT;
        at[2].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[2].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference resolveRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        // Fragment density map (fixed foveation): last attachment, read-only for the whole pass.
        std::array<VkAttachmentDescription, 4> at4{};
        for (int i = 0; i < 3; ++i) at4[i] = at[i];
        const uint32_t fdmIndex = ms ? 3u : 2u;
        at4[fdmIndex].format = VK_FORMAT_R8G8_UNORM; at4[fdmIndex].samples = VK_SAMPLE_COUNT_1_BIT;
        at4[fdmIndex].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at4[fdmIndex].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at4[fdmIndex].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at4[fdmIndex].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at4[fdmIndex].initialLayout = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
        at4[fdmIndex].finalLayout = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
        VkRenderPassFragmentDensityMapCreateInfoEXT fdmInfo{VK_STRUCTURE_TYPE_RENDER_PASS_FRAGMENT_DENSITY_MAP_CREATE_INFO_EXT};
        fdmInfo.fragmentDensityMapAttachment = {fdmIndex, VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = &colorRef;
        sp.pResolveAttachments = ms ? &resolveRef : nullptr;
        sp.pDepthStencilAttachment = &depthRef;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = (ms ? 3u : 2u) + (m_useFdm ? 1u : 0u); ri.pAttachments = at4.data();
        if (m_useFdm) ri.pNext = &fdmInfo;
        ri.subpassCount = 1; ri.pSubpasses = &sp;
        ri.dependencyCount = 1; ri.pDependencies = &dep;
        XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_vkDevice, &ri, nullptr, &m_pass));
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK render pass: MSAA x%d, transient colour/depth, on-tile resolve, foveation %s",
                                         int(m_samples), m_useFdm ? "ON (density map)" : "off"));
    }

    struct EyeTargets {
        VkExtent2D ext{0, 0};
        VkImage color = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
        VkDeviceMemory colorMem = VK_NULL_HANDLE, depthMem = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE, depthView = VK_NULL_HANDLE;
    };
    struct FbEntry { VkImage image; uint32_t eye; VkImageView view; VkFramebuffer fb; VkImageView fdmView = VK_NULL_HANDLE;
                     VkImage depthImage = VK_NULL_HANDLE; VkImageView depthResolveView = VK_NULL_HANDLE; };
    bool m_depthResolveWanted = false, m_depthResolveOn = false;

    // ---- AppSW motion vectors (26/09) -------------------------------------------------------------------------------
    // The headset shows each 60 Hz frame twice at 120 Hz: an object moving on screen (the player's car turning in the
    // chase view) is seen doubled, i.e. blurred (Guillaume, A/B LISSAGE CASQUE/NON). Motion vectors let the headset move
    // it in the synthesised frame. ONLY where the motion is certain: an object with one copy in both frames (its own
    // matrices, exact), and the objects within appsw.mvBorrow units of one (its wheels, swapped models) take its motion.
    // Everything else: zero -- exactly what was shown before. prevPos (binding 16) carries the previous positions; the
    // colour pass ignores them (uInterp = 1 without smooth motion).
    struct MvMat { float m[14]; };
    // The HUD drawn over the scene (normalised eye rectangles of its cells): its pixels do not move with the car behind
    // them (26/09: the HUD and the menu smeared with the intro's cars passing behind).
    std::vector<std::array<float, 4>> m_hudMvRects[2];
    bool m_hudMvAll[2] = {false, false};
    bool m_mvWanted = false, m_mvFresh = false;
    std::unordered_map<uint32_t, MvMat> m_mvPrevMats;   // addr -> matrix of single-copy main-view objects, previous frame
    uint32_t m_mvCertain = 0, m_mvBorrowed = 0, m_mvVerts = 0, m_mvScenery = 0;
    VkRenderPass m_mvPass = VK_NULL_HANDLE;
    VkPipeline m_mvPipe = VK_NULL_HANDLE;
    VkPipelineLayout m_mvLayout = VK_NULL_HANDLE;   // the Model 2 set layout + a fragment push constant (scale)
    struct MvFb { VkImage image; VkImageView view; VkFramebuffer fb; VkExtent2D ext; };
    std::vector<MvFb> m_mvFbs;
    VkImage m_mvDepth = VK_NULL_HANDLE; VkDeviceMemory m_mvDepthMem = VK_NULL_HANDLE; VkImageView m_mvDepthView = VK_NULL_HANDLE;
    VkExtent2D m_mvDepthExt{0, 0};

    bool IsMainPrim(const tcvr_m2_prim& p) const {
        return p.center_x == m_mainCenter[0] && p.center_y == m_mainCenter[1] &&
               std::abs(p.clip_l - m_mainClip[0]) <= 2 && std::abs(p.clip_t - m_mainClip[1]) <= 2 &&
               std::abs(p.clip_r - m_mainClip[2]) <= 2 && std::abs(p.clip_b - m_mainClip[3]) <= 2;
    }

    void MotionPrevCertain(const tcvr_m2_frame& frame, size_t nv) {
        m_prevPosCpu.assign(nv * 4, 0.0f);
        m_mvFresh = true;
        m_mvCertain = m_mvBorrowed = m_mvVerts = m_mvScenery = 0;
        if (!frame.raw_motion || !m_haveMainView || m_isMenuM1) { m_mvPrevMats.clear(); return; }
        std::unordered_map<uint32_t, MvMat> cur;
        std::unordered_map<uint32_t, uint32_t> copies;
        for (size_t k = 0; k < m_rawPrims.size() && k < m_rawPrimSrc.size(); ++k) {
            const tcvr_m2_prim& p = m_rawPrims[k];
            if (!IsMainPrim(p)) continue;
            const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
            if (mo[14] < 0.5f) continue;
            uint32_t& n = copies[p.motion_addr];
            n = std::max(n, p.motion_serial + 1u);
            MvMat om; for (int i = 0; i < 14; ++i) om.m[i] = mo[i];
            cur.emplace(p.motion_addr, om);
        }
        std::unordered_map<uint32_t, std::pair<const float*, const float*>> src;   // addr -> (cur, prev) motion source
        std::vector<uint32_t> certain;
        // Only objects that do NOT move like the scenery (the camera delta measured by CameraDelta, same frame): the
        // static world stays uniformly at zero -- vectors on the single-copy road pieces and not on the many-copy
        // trees would tear the scenery apart in the synthesised frame. No reliable camera delta: no vectors at all.
        if (!m_cdValid) { m_mvPrevMats.clear(); for (const auto& kv : cur) if (copies[kv.first] == 1u) m_mvPrevMats.emplace(kv.first, kv.second); return; }
        for (const auto& kv : cur) {
            if (copies[kv.first] != 1u) continue;
            auto it = m_mvPrevMats.find(kv.first);
            if (it == m_mvPrevMats.end()) continue;
            const float* C = kv.second.m; const float* P = it->second.m;
            float Ci[9], R[9];
            if (!Inverse3(C, Ci)) continue;
            Mul3(P, Ci, R);
            float T[3];
            for (int rr = 0; rr < 3; ++rr) T[rr] = P[9 + rr] - (R[rr] * C[9] + R[3 + rr] * C[10] + R[6 + rr] * C[11]);
            float dr = 0.0f, dt = 0.0f;
            for (int q = 0; q < 9; ++q) dr = std::max(dr, std::fabs(R[q] - m_cdR[q]));
            for (int q = 0; q < 3; ++q) dt = std::max(dt, std::fabs(T[q] - m_cdT[q]));
            const float tol = 0.02f + 0.01f * std::sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);
            if (dr < 0.002f && dt < tol) continue;   // moves like the scenery: stays at zero with it
            src.emplace(kv.first, std::make_pair(kv.second.m, it->second.m));
            certain.push_back(kv.first);
        }
        m_mvCertain = uint32_t(certain.size());
        const float borrow = std::max(0.0f, arcadexr::config::GetFloat("appsw.mvBorrow", 3.0f));
        for (const auto& kv : cur) {
            if (src.count(kv.first)) continue;
            const float* a = kv.second.m;
            float bestD = borrow * borrow; uint32_t best = 0; bool found = false;
            for (uint32_t c : certain) {
                const float* b = cur[c].m;
                const float d = (a[9] - b[9]) * (a[9] - b[9]) + (a[10] - b[10]) * (a[10] - b[10]) + (a[11] - b[11]) * (a[11] - b[11]);
                if (d < bestD) { bestD = d; best = c; found = true; }
            }
            if (found) { src.emplace(kv.first, src[best]); ++m_mvBorrowed; }
        }
        std::unordered_map<uint32_t, std::pair<const float*, const float*>> primBorrow;   // prim -> borrowed motion
        const bool sceneryVectors = arcadexr::config::GetInt("appsw_mvScenery", 1) != 0;   // live A/B
        for (size_t v = 0; v < nv; ++v) {
            const uint32_t k = m_rawPrimOfVertex[v];
            if (k >= m_rawPrims.size() || k >= m_rawPrimSrc.size()) continue;
            const tcvr_m2_prim& p = m_rawPrims[k];
            if (!IsMainPrim(p)) continue;
            const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
            if (std::fabs(mo[12]) < 1e-6f || std::fabs(mo[13]) < 1e-6f) continue;
            const float* cm = nullptr; const float* pm = nullptr;
            // The scenery (26/09): every polygon with no motion of its own (not a moving certain object, not near one)
            // moves like the camera -- the SAME measured delta for all of it, trees and road alike, so it stays whole.
            // A vector for every pixel, as an engine gives AppSW; zero left the whole moving picture doubled at 120 Hz.
            auto sceneryPrev = [&](const float* dd, float* o4) {
                if (!sceneryVectors) return false;
                const float o[3] = {dd[0] / mo[12], dd[1] / mo[13], dd[2]};
                const float px = m_cdR[0] * o[0] + m_cdR[3] * o[1] + m_cdR[6] * o[2] + m_cdT[0];
                const float py = m_cdR[1] * o[0] + m_cdR[4] * o[1] + m_cdR[7] * o[2] + m_cdT[1];
                const float pz = m_cdR[2] * o[0] + m_cdR[5] * o[1] + m_cdR[8] * o[2] + m_cdT[2];
                o4[0] = px * mo[12]; o4[1] = py * mo[13]; o4[2] = pz; o4[3] = 1.0f;
                return true;
            };
            if (mo[14] >= 0.5f) {
                auto it = src.find(p.motion_addr);
                if (it == src.end()) { if (sceneryPrev(&m_rawVerts[v * 5], &m_prevPosCpu[v * 4])) ++m_mvScenery; continue; }
                cm = it->second.first; pm = it->second.second;
            } else {
                // No matrix (direct polygons: the car's shadow, its dust): the motion of the moving certain object
                // whose origin is nearest to the polygon's centroid, within appsw.mvBorrow (26/09: the shadow stayed
                // doubled under a sharp car).
                auto pb = primBorrow.find(uint32_t(k));
                if (pb == primBorrow.end()) {
                    std::pair<const float*, const float*> got{nullptr, nullptr};
                    if (p.vertex_count > 0 && !certain.empty()) {
                        float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                        for (uint32_t vi = 0; vi < p.vertex_count; ++vi) {
                            const float* dd = &m_rawVerts[size_t(p.first_vertex + vi) * 5];
                            cx += dd[0] / mo[12]; cy += dd[1] / mo[13]; cz += dd[2];
                        }
                        cx /= float(p.vertex_count); cy /= float(p.vertex_count); cz /= float(p.vertex_count);
                        float bestD = borrow * borrow; uint32_t best = 0; bool found = false;
                        for (uint32_t c : certain) {
                            const float* b = cur[c].m;
                            const float dd = (cx - b[9]) * (cx - b[9]) + (cy - b[10]) * (cy - b[10]) + (cz - b[11]) * (cz - b[11]);
                            if (dd < bestD) { bestD = dd; best = c; found = true; }
                        }
                        if (found) got = src[best];
                    }
                    pb = primBorrow.emplace(uint32_t(k), got).first;
                }
                cm = pb->second.first; pm = pb->second.second;
                if (!cm || !pm) { if (sceneryPrev(&m_rawVerts[v * 5], &m_prevPosCpu[v * 4])) ++m_mvScenery; continue; }
            }
            const float* d = &m_rawVerts[v * 5];
            const float o[3] = {d[0] / mo[12] - cm[9], d[1] / mo[13] - cm[10], d[2] - cm[11]};
            float ci[9];
            if (!Inverse3(cm, ci)) continue;
            const float ob[3] = {ci[0] * o[0] + ci[3] * o[1] + ci[6] * o[2], ci[1] * o[0] + ci[4] * o[1] + ci[7] * o[2],
                                 ci[2] * o[0] + ci[5] * o[1] + ci[8] * o[2]};
            const float px = ob[0] * pm[0] + ob[1] * pm[3] + ob[2] * pm[6] + pm[9];
            const float py = ob[0] * pm[1] + ob[1] * pm[4] + ob[2] * pm[7] + pm[10];
            const float pz = ob[0] * pm[2] + ob[1] * pm[5] + ob[2] * pm[8] + pm[11];
            float* o4 = &m_prevPosCpu[v * 4];
            o4[0] = px * pm[12]; o4[1] = py * pm[13]; o4[2] = pz; o4[3] = 1.0f;
            ++m_mvVerts;
        }
        m_mvPrevMats.clear();
        for (const auto& kv : cur) if (copies[kv.first] == 1u) m_mvPrevMats.emplace(kv.first, kv.second);
    }

    bool CreateMvObjects() {
        if (m_mvPipe != VK_NULL_HANDLE) return true;
        std::array<VkAttachmentDescription, 2> at{};
        at[0].format = VK_FORMAT_R16G16B16A16_SFLOAT; at[0].samples = VK_SAMPLE_COUNT_1_BIT;
        at[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        at[1].format = kDepthFormat; at[1].samples = VK_SAMPLE_COUNT_1_BIT;
        at[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = &cref; sp.pDepthStencilAttachment = &dref;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = 2; ri.pAttachments = at.data(); ri.subpassCount = 1; ri.pSubpasses = &sp;
        if (vkCreateRenderPass(m_vkDevice, &ri, nullptr, &m_mvPass) != VK_SUCCESS) return false;
        VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(float)};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = 1; li.pSetLayouts = &m_m2DescLayout; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(m_vkDevice, &li, nullptr, &m_mvLayout) != VK_SUCCESS) return false;
        VkShaderModule vs = CreateShaderModule(c_m2VertAppswMvSpv, sizeof(c_m2VertAppswMvSpv));
        VkShaderModule fs = CreateShaderModule(c_appswMvFragSpv, sizeof(c_appswMvFragSpv));
        std::array<VkPipelineShaderStageCreateInfo, 2> st{};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = vs; st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
        std::array<VkVertexInputBindingDescription, 2> binds{{{0, 5 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX},
                                                              {1, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX}}};
        std::array<VkVertexInputAttributeDescription, 3> attrs{{{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                                                 {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 2 * sizeof(float)},
                                                                 {2, 1, VK_FORMAT_R32_UINT, 0}}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = binds.data();
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1; cb.pAttachments = &cba;
        std::array<VkDynamicState, 2> dyn{{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(dyn.size()); dy.pDynamicStates = dyn.data();
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 2; pi.pStages = st.data();
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp; pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds; pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
        pi.layout = m_mvLayout; pi.renderPass = m_mvPass; pi.subpass = 0;
        const VkResult res = vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &m_mvPipe);
        vkDestroyShaderModule(m_vkDevice, vs, nullptr);
        vkDestroyShaderModule(m_vkDevice, fs, nullptr);
        Log::Write(Log::Level::Info, Fmt("TCVR_APPSW motion-vector pass created: %d", int(res)));
        return res == VK_SUCCESS;
    }
    void DestroyMvTargets() {
        for (auto& f : m_mvFbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); }
        m_mvFbs.clear();
        if (m_mvDepthView) vkDestroyImageView(m_vkDevice, m_mvDepthView, nullptr);
        if (m_mvDepth) vkDestroyImage(m_vkDevice, m_mvDepth, nullptr);
        if (m_mvDepthMem) vkFreeMemory(m_vkDevice, m_mvDepthMem, nullptr);
        m_mvDepthView = VK_NULL_HANDLE; m_mvDepth = VK_NULL_HANDLE; m_mvDepthMem = VK_NULL_HANDLE; m_mvDepthExt = {0, 0};
    }
    void DestroyMvObjects() {
        if (m_vkDevice == VK_NULL_HANDLE) return;
        DestroyMvTargets();
        if (m_mvPipe) vkDestroyPipeline(m_vkDevice, m_mvPipe, nullptr);
        if (m_mvPass) vkDestroyRenderPass(m_vkDevice, m_mvPass, nullptr);
        if (m_mvLayout) vkDestroyPipelineLayout(m_vkDevice, m_mvLayout, nullptr);
        m_mvPipe = VK_NULL_HANDLE; m_mvPass = VK_NULL_HANDLE; m_mvLayout = VK_NULL_HANDLE;
    }
    // Records the motion-vector pass (outside any render pass) into the headset's motion-vector image: cleared to zero,
    // then the main view's polygons with their CurrNDC - PrevNDC where the motion is certain.
    bool RenderAppSwMotion(VkCommandBuffer cmd, uint32_t eye, VkImage image, VkExtent2D ext, bool drawGeometry) {
        if (!m_initialized || image == VK_NULL_HANDLE || !CreateMvObjects()) return false;
        if (m_mvDepthExt.width != ext.width || m_mvDepthExt.height != ext.height) {
            DestroyMvTargets();
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D; ii.format = kDepthFormat; ii.extent = {ext.width, ext.height, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (vkCreateImage(m_vkDevice, &ii, nullptr, &m_mvDepth) != VK_SUCCESS) return false;
            AllocTransient(m_mvDepth, &m_mvDepthMem);
            VkImageViewCreateInfo dv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            dv.image = m_mvDepth; dv.viewType = VK_IMAGE_VIEW_TYPE_2D; dv.format = kDepthFormat;
            dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(m_vkDevice, &dv, nullptr, &m_mvDepthView) != VK_SUCCESS) return false;
            m_mvDepthExt = ext;
        }
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_mvFbs) if (f.image == image) fb = f.fb;
        if (fb == VK_NULL_HANDLE) {
            MvFb f{image, VK_NULL_HANDLE, VK_NULL_HANDLE, ext};
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(m_vkDevice, &vi, nullptr, &f.view) != VK_SUCCESS) return false;
            std::array<VkImageView, 2> att{{f.view, m_mvDepthView}};
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_mvPass; fi.attachmentCount = 2; fi.pAttachments = att.data();
            fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
            if (vkCreateFramebuffer(m_vkDevice, &fi, nullptr, &f.fb) != VK_SUCCESS) return false;
            m_mvFbs.push_back(f);
            fb = f.fb;
        }
        std::array<VkClearValue, 2> cv{};
        cv[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_mvPass; bi.framebuffer = fb; bi.renderArea = {{0, 0}, ext};
        bi.clearValueCount = 2; bi.pClearValues = cv.data();
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        eye &= 1u;
        if (drawGeometry && m_mvFresh && m_opaqueIndexCount > 0 && (m_mvVerts + m_mvScenery) > 0) {
            VkViewport v{0.0f, 0.0f, float(ext.width), float(ext.height), 0.0f, 1.0f};
            VkRect2D sc{{0, 0}, ext};
            vkCmdSetViewport(cmd, 0, 1, &v);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_mvPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_mvLayout, 0, 1, &m_m2DescSetF[m_fs][eye][0], 0, nullptr);
            const float k = arcadexr::config::GetFloat("appsw_mvScale", 1.0f);
            const float sc4[4] = {k, (arcadexr::config::GetInt("appsw_mvFlipY", 0) != 0 ? -k : k), k, 0.0f};
            vkCmdPushConstants(cmd, m_mvLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(sc4), sc4);
            VkBuffer vtxBufs[2] = {m_vboBufferF[m_fs].buf, m_primIndexBufferF[m_fs].buf};
            VkDeviceSize offs[2] = {0, 0};
            vkCmdBindVertexBuffers(cmd, 0, 2, vtxBufs, offs);
            vkCmdBindIndexBuffer(cmd, m_iboBufferF[m_fs].buf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m_opaqueIndexCount, 1, 0, 0, 0);
            // the HUD's cells back to zero
            VkClearAttachment ca{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
            std::vector<VkClearRect> crs;
            if (m_hudMvAll[eye]) crs.push_back({{{0, 0}, ext}, 0, 1});
            else
                for (const auto& h : m_hudMvRects[eye]) {
                    const int x0 = std::max(0, int(h[0] * float(ext.width)) - 1), y0 = std::max(0, int(h[1] * float(ext.height)) - 1);
                    const int x1 = std::min(int(ext.width), int(h[2] * float(ext.width)) + 2), y1 = std::min(int(ext.height), int(h[3] * float(ext.height)) + 2);
                    if (x1 > x0 && y1 > y0) crs.push_back({{{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}}, 0, 1});
                }
            if (!crs.empty()) vkCmdClearAttachments(cmd, 1, &ca, uint32_t(crs.size()), crs.data());
        }
        vkCmdEndRenderPass(cmd);
        return true;
    }
    void SetMotionVectorsWanted(bool on) { m_mvWanted = on; }
    void MotionStats(uint32_t* certain, uint32_t* borrowed, uint32_t* verts) const { *certain = m_mvCertain; *borrowed = m_mvBorrowed; *verts = m_mvVerts + m_mvScenery; }

    // ---- AppSW depth pass (26/09) -----------------------------------------------------------------------------------
    // The real distance of the main view's polygons (opaque + cut-outs, index range [0, m_secIndexStart)) into the
    // headset's low-resolution AppSW depth image; far where nothing (sky, flat menus). Depth only, no colour.
    float m_swMvp[2][16] = {}, m_swFocus[2] = {512.0f, 512.0f};
    bool m_swMvpValid[2] = {false, false};
    VkRenderPass m_swDepthPass = VK_NULL_HANDLE;
    VkPipeline m_swDepthPipe = VK_NULL_HANDLE;
    VkFormat m_swDepthFormat = VK_FORMAT_UNDEFINED;
    struct SwDepthFb { VkImage image; VkImageView view; VkFramebuffer fb; VkExtent2D ext; };
    std::vector<SwDepthFb> m_swDepthFbs;

    bool CreateSwDepthObjects(VkFormat format) {
        if (m_swDepthPipe != VK_NULL_HANDLE && m_swDepthFormat == format) return true;
        DestroySwDepthObjects();
        m_swDepthFormat = format;
        VkAttachmentDescription at{};
        at.format = format; at.samples = VK_SAMPLE_COUNT_1_BIT;
        at.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference dref{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; sp.pDepthStencilAttachment = &dref;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = 1; ri.pAttachments = &at; ri.subpassCount = 1; ri.pSubpasses = &sp;
        if (vkCreateRenderPass(m_vkDevice, &ri, nullptr, &m_swDepthPass) != VK_SUCCESS) return false;
        // The colour pass's own vertex shader (m2_vert) built with APPSW_DEPTH, its layout and descriptor sets: every
        // polygon lands exactly where the player sees it (a minimal shader left the camera-attached fades at their raw
        // place, 0.4-0.9 m in front of the eyes over the whole picture: everything smeared when the head moved).
        VkShaderModule vs = CreateShaderModule(c_m2VertAppswSpv, sizeof(c_m2VertAppswSpv));
        VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        st.stage = VK_SHADER_STAGE_VERTEX_BIT; st.module = vs; st.pName = "main";
        std::array<VkVertexInputBindingDescription, 2> binds{{{0, 5 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX},
                                                              {1, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX}}};
        std::array<VkVertexInputAttributeDescription, 3> attrs{{{0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                                                                 {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 2 * sizeof(float)},
                                                                 {2, 1, VK_FORMAT_R32_UINT, 0}}};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = binds.data();
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        std::array<VkDynamicState, 2> dyn{{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(dyn.size()); dy.pDynamicStates = dyn.data();
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 1; pi.pStages = &st;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp; pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds; pi.pColorBlendState = &cb; pi.pDynamicState = &dy;
        pi.layout = m_m2PipelineLayout; pi.renderPass = m_swDepthPass; pi.subpass = 0;
        const VkResult res = vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr, &m_swDepthPipe);
        vkDestroyShaderModule(m_vkDevice, vs, nullptr);
        Log::Write(Log::Level::Info, Fmt("TCVR_APPSW depth pass created (format %d): %d", int(format), int(res)));
        return res == VK_SUCCESS;
    }
    void DestroySwDepthObjects() {
        if (m_vkDevice == VK_NULL_HANDLE) return;
        for (auto& f : m_swDepthFbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); }
        m_swDepthFbs.clear();
        if (m_swDepthPipe) vkDestroyPipeline(m_vkDevice, m_swDepthPipe, nullptr);
        if (m_swDepthPass) vkDestroyRenderPass(m_vkDevice, m_swDepthPass, nullptr);
        m_swDepthPipe = VK_NULL_HANDLE; m_swDepthPass = VK_NULL_HANDLE;
    }

    // Records the depth pass into cmd (outside any render pass). drawGeometry false: only cleared to far.
    bool RenderAppSwDepth(VkCommandBuffer cmd, uint32_t eye, VkImage image, VkExtent2D ext, VkFormat format, bool drawGeometry) {
        if (!m_initialized || image == VK_NULL_HANDLE || !CreateSwDepthObjects(format)) return false;
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_swDepthFbs) if (f.image == image && f.ext.width == ext.width && f.ext.height == ext.height) fb = f.fb;
        if (fb == VK_NULL_HANDLE) {
            SwDepthFb f{image, VK_NULL_HANDLE, VK_NULL_HANDLE, ext};
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(m_vkDevice, &vi, nullptr, &f.view) != VK_SUCCESS) return false;
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_swDepthPass; fi.attachmentCount = 1; fi.pAttachments = &f.view;
            fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
            if (vkCreateFramebuffer(m_vkDevice, &fi, nullptr, &f.fb) != VK_SUCCESS) return false;
            m_swDepthFbs.push_back(f);
            fb = f.fb;
        }
        VkClearValue cv{}; cv.depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_swDepthPass; bi.framebuffer = fb; bi.renderArea = {{0, 0}, ext};
        bi.clearValueCount = 1; bi.pClearValues = &cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        eye &= 1u;
        // Everything the colour pass draws with depth (main view, cut-outs, secondary views/HUD on their plane),
        // with this eye's own uniforms.
        if (drawGeometry && m_swMvpValid[eye] && m_opaqueIndexCount > 0) {
            VkViewport v{0.0f, 0.0f, float(ext.width), float(ext.height), 0.0f, 1.0f};
            VkRect2D sc{{0, 0}, ext};
            vkCmdSetViewport(cmd, 0, 1, &v);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_swDepthPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSetF[m_fs][eye][0], 0, nullptr);
            VkBuffer vtxBufs[2] = {m_vboBufferF[m_fs].buf, m_primIndexBufferF[m_fs].buf};
            VkDeviceSize offs[2] = {0, 0};
            vkCmdBindVertexBuffers(cmd, 0, 2, vtxBufs, offs);
            vkCmdBindIndexBuffer(cmd, m_iboBufferF[m_fs].buf, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, m_opaqueIndexCount, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
        return true;
    }
    void SetDepthResolveWanted(bool on) { m_depthResolveWanted = on; }   // before Initialize
    bool DepthResolveOn() const { return m_depthResolveOn; }
    // A menu screen of the game (car or course select...): the menu rule of the renderer (no horizon, 30 frames).
    bool GameMenuScreen() const { return m_initialized && m_isMenuM1; }
    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

    void AllocTransient(VkImage img, VkDeviceMemory* mem) {
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(m_vkDevice, img, &req);
        try {
            m_memAllocator->Allocate(req, mem, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT);
        } catch (...) {
            m_memAllocator->Allocate(req, mem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, img, *mem, 0));
    }

    void CreateEyeTargets(EyeTargets& et, VkExtent2D ext) {
        et.ext = ext;
        auto mk = [&](VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage* img, VkDeviceMemory* mem, VkImageView* view) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {ext.width, ext.height, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = m_samples; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = usage | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
            // With foveation, the attachments must be SUBSAMPLED or the driver may render them at full density.
            if (m_useFdm && arcadexr::config::GetInt("m2.fdmSubsampled", 1) != 0) ii.flags |= VK_IMAGE_CREATE_SUBSAMPLED_BIT_EXT;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &ii, nullptr, img));
            AllocTransient(*img, mem);
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = *img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
            vi.subresourceRange = {aspect, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &vi, nullptr, view));
        };
        if (IsMsaa()) mk(m_colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, &et.color, &et.colorMem, &et.colorView);
        mk(kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, &et.depth, &et.depthMem, &et.depthView);
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK eye targets %ux%u MSAA x%d", ext.width, ext.height, int(m_samples)));
    }

    void DestroyEyeTargets(EyeTargets& et) {
        if (et.ext.width == 0) return;
        vkDeviceWaitIdle(m_vkDevice);
        // framebuffers reference these views: drop them all, they are rebuilt lazily
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); if (f.fdmView) vkDestroyImageView(m_vkDevice, f.fdmView, nullptr); if (f.depthResolveView) vkDestroyImageView(m_vkDevice, f.depthResolveView, nullptr); }
        m_fbs.clear();
        if (et.colorView) vkDestroyImageView(m_vkDevice, et.colorView, nullptr);
        if (et.depthView) vkDestroyImageView(m_vkDevice, et.depthView, nullptr);
        if (et.color) vkDestroyImage(m_vkDevice, et.color, nullptr);
        if (et.depth) vkDestroyImage(m_vkDevice, et.depth, nullptr);
        if (et.colorMem) vkFreeMemory(m_vkDevice, et.colorMem, nullptr);
        if (et.depthMem) vkFreeMemory(m_vkDevice, et.depthMem, nullptr);
        et = EyeTargets{};
    }

    void Cleanup() {
        if (!m_initialized) return;
        m_initialized = false;
        DestroySwDepthObjects();
        DestroyMvObjects();

        vkDeviceWaitIdle(m_vkDevice);
        m_regions.Destroy();
        DestroyEyeTargets(m_eyeTargets[0]);
        DestroyEyeTargets(m_eyeTargets[1]);
        DestroyEyeTargets(m_eyeTargets[2]);
        if (m_flat.view) vkDestroyImageView(m_vkDevice, m_flat.view, nullptr);
        if (m_flat.image) vkDestroyImage(m_vkDevice, m_flat.image, nullptr);
        if (m_flat.mem) vkFreeMemory(m_vkDevice, m_flat.mem, nullptr);
        m_flat = FlatTarget{};
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); if (f.fdmView) vkDestroyImageView(m_vkDevice, f.fdmView, nullptr); if (f.depthResolveView) vkDestroyImageView(m_vkDevice, f.depthResolveView, nullptr); }
        m_fbs.clear();
        if (m_pass != VK_NULL_HANDLE) { vkDestroyRenderPass(m_vkDevice, m_pass, nullptr); m_pass = VK_NULL_HANDLE; }

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_vkDevice, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }

        auto destroyPipe = [&](VkPipeline& p) { if (p != VK_NULL_HANDLE) { vkDestroyPipeline(m_vkDevice, p, nullptr); p = VK_NULL_HANDLE; } };
        destroyPipe(m_m2PipelineOpaque);
        destroyPipe(m_m2PipelineFast);
        destroyPipe(m_m2PipelineCutNoWrite);
        destroyPipe(m_m2PipelineCutDepth);
        destroyPipe(m_m2PipelineFastLean);
        destroyPipe(m_m2PipelineCutLean);
        destroyPipe(m_m2PipelineCutColor);
        destroyPipe(m_voidPipelineFar);
        destroyPipe(m_planePipelineFar);
        destroyPipe(m_m2PipelineGlass);
        destroyPipe(m_voidPipeline);
        destroyPipe(m_planePipeline);

        auto destroyLayout = [&](VkPipelineLayout& l) { if (l != VK_NULL_HANDLE) { vkDestroyPipelineLayout(m_vkDevice, l, nullptr); l = VK_NULL_HANDLE; } };
        destroyLayout(m_m2PipelineLayout);
        destroyLayout(m_voidPipelineLayout);
        destroyLayout(m_planePipelineLayout);

        auto destroyDescLayout = [&](VkDescriptorSetLayout& l) { if (l != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(m_vkDevice, l, nullptr); l = VK_NULL_HANDLE; } };
        destroyDescLayout(m_m2DescLayout);
        destroyDescLayout(m_planeDescLayout);

        auto destroyMod = [&](VkShaderModule& m) { if (m != VK_NULL_HANDLE) { vkDestroyShaderModule(m_vkDevice, m, nullptr); m = VK_NULL_HANDLE; } };
        destroyMod(m_m2VertModule);
        destroyMod(m_m2FragModule);
        destroyMod(m_m2FragNdModule);
        destroyMod(m_m2FragCutDModule);
        destroyMod(m_m2FragLeanModule);
        destroyMod(m_m2FragLeanCutModule);
        destroyMod(m_m2FragCutCModule);
        destroyMod(m_quadFarVertModule);
        destroyMod(m_quadVertModule);
        destroyMod(m_voidFragModule);
        destroyMod(m_planeFragModule);

        if (m_sheetSampler != VK_NULL_HANDLE) { vkDestroySampler(m_vkDevice, m_sheetSampler, nullptr); m_sheetSampler = VK_NULL_HANDLE; }
        for (int i = 2; i < 4; ++i) {
            if (m_sheetView[i] != VK_NULL_HANDLE) { vkDestroyImageView(m_vkDevice, m_sheetView[i], nullptr); m_sheetView[i] = VK_NULL_HANDLE; }
            if (m_sheetImage[i] != VK_NULL_HANDLE) { vkDestroyImage(m_vkDevice, m_sheetImage[i], nullptr); m_sheetImage[i] = VK_NULL_HANDLE; }
            if (m_sheetMem[i] != VK_NULL_HANDLE) { vkFreeMemory(m_vkDevice, m_sheetMem[i], nullptr); m_sheetMem[i] = VK_NULL_HANDLE; }
        }
        for (int i = 0; i < 2; ++i) {
            if (m_layerSampler[i] != VK_NULL_HANDLE) { vkDestroySampler(m_vkDevice, m_layerSampler[i], nullptr); m_layerSampler[i] = VK_NULL_HANDLE; }
            if (m_sheetView[i] != VK_NULL_HANDLE) { vkDestroyImageView(m_vkDevice, m_sheetView[i], nullptr); m_sheetView[i] = VK_NULL_HANDLE; }
            if (m_sheetImage[i] != VK_NULL_HANDLE) { vkDestroyImage(m_vkDevice, m_sheetImage[i], nullptr); m_sheetImage[i] = VK_NULL_HANDLE; }
            if (m_sheetMem[i] != VK_NULL_HANDLE) { vkFreeMemory(m_vkDevice, m_sheetMem[i], nullptr); m_sheetMem[i] = VK_NULL_HANDLE; }
            if (m_layerView[i] != VK_NULL_HANDLE) { vkDestroyImageView(m_vkDevice, m_layerView[i], nullptr); m_layerView[i] = VK_NULL_HANDLE; }
            if (m_layerImage[i] != VK_NULL_HANDLE) { vkDestroyImage(m_vkDevice, m_layerImage[i], nullptr); m_layerImage[i] = VK_NULL_HANDLE; }
            if (m_layerMem[i] != VK_NULL_HANDLE) { vkFreeMemory(m_vkDevice, m_layerMem[i], nullptr); m_layerMem[i] = VK_NULL_HANDLE; }
        }

        for (m_fs = 0; m_fs < kFrames; ++m_fs) {
        m_vboBufferF[m_fs].Reset(m_vkDevice);
        m_prevBufferF[m_fs].Reset(m_vkDevice);
        m_primIndexBufferF[m_fs].Reset(m_vkDevice);
        m_iboBufferF[m_fs].Reset(m_vkDevice);
        m_primsBufferF[m_fs].Reset(m_vkDevice);
        m_palramBufferF[m_fs].Reset(m_vkDevice);
        m_colorxlatBufferF[m_fs].Reset(m_vkDevice);
        m_lumaramBufferF[m_fs].Reset(m_vkDevice);
        m_gammaBufferF[m_fs].Reset(m_vkDevice);
        for (int i = 0; i < 2; ++i) m_layerStagingBufferF[m_fs][i].Reset(m_vkDevice);
        for (int e = 0; e < 2; ++e)
            for (int p = 0; p < 2; ++p) m_uboBufferF[m_fs][e][p].Reset(m_vkDevice);
        }
        m_fs = 0;
        m_dummyBuffer.Reset(m_vkDevice);
        m_lutBuffer.Reset(m_vkDevice);
        for (int i = 0; i < 4; ++i) {
            m_sheetStagingBuffer[i].Reset(m_vkDevice);
            m_sheetStagingMapped[i] = nullptr;
        }
        m_texturesUploaded = false;
        m_texHash = 0;


    }

private:
    VkShaderModule CreateShaderModule(const unsigned char* code, size_t size) {
        VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize = size;
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule mod = VK_NULL_HANDLE;
        XRC_CHECK_THROW_VKCMD(vkCreateShaderModule(m_vkDevice, &createInfo, nullptr, &mod));
        return mod;
    }

    void CreatePipelines(VkRenderPass renderPass) {
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineViewportStateCreateInfo vpState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vpState.viewportCount = 1;
        vpState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterState{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterState.cullMode = VK_CULL_MODE_NONE;
        rasterState.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterState.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo msState{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        msState.rasterizationSamples = m_samples;

        // --- 1. Void Pipeline ---
        {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = m_quadVertModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = m_voidFragModule;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo viState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo iaState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineDepthStencilStateCreateInfo dsState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsState.depthTestEnable = VK_FALSE;
            dsState.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState cbAtt{};
            cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            cbAtt.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo cbState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbState.attachmentCount = 1;
            cbState.pAttachments = &cbAtt;

            VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeInfo.stageCount = 2;
            pipeInfo.pStages = stages;
            pipeInfo.pVertexInputState = &viState;
            pipeInfo.pInputAssemblyState = &iaState;
            pipeInfo.pViewportState = &vpState;
            pipeInfo.pRasterizationState = &rasterState;
            pipeInfo.pMultisampleState = &msState;
            pipeInfo.pDepthStencilState = &dsState;
            pipeInfo.pColorBlendState = &cbState;
            pipeInfo.pDynamicState = &dynamicState;
            pipeInfo.layout = m_voidPipelineLayout;
            pipeInfo.renderPass = renderPass;

            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_voidPipeline));
            // Far-plane variant, depth-tested: drawn after the geometry, shades only the sky.
            stages[0].module = m_quadFarVertModule;
            dsState.depthTestEnable = VK_TRUE;
            dsState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_voidPipelineFar));
        }

        // --- 2. Plane Pipeline ---
        {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = m_quadVertModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = m_planeFragModule;
            stages[1].pName = "main";

            VkPipelineVertexInputStateCreateInfo viState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineInputAssemblyStateCreateInfo iaState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineDepthStencilStateCreateInfo dsState{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsState.depthTestEnable = VK_FALSE;
            dsState.depthWriteEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState cbAtt{};
            cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            cbAtt.blendEnable = VK_TRUE;
            cbAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbAtt.colorBlendOp = VK_BLEND_OP_ADD;
            cbAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbAtt.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo cbState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbState.attachmentCount = 1;
            cbState.pAttachments = &cbAtt;

            VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeInfo.stageCount = 2;
            pipeInfo.pStages = stages;
            pipeInfo.pVertexInputState = &viState;
            pipeInfo.pInputAssemblyState = &iaState;
            pipeInfo.pViewportState = &vpState;
            pipeInfo.pRasterizationState = &rasterState;
            pipeInfo.pMultisampleState = &msState;
            pipeInfo.pDepthStencilState = &dsState;
            pipeInfo.pColorBlendState = &cbState;
            pipeInfo.pDynamicState = &dynamicState;
            pipeInfo.layout = m_planePipelineLayout;
            pipeInfo.renderPass = renderPass;

            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_planePipeline));
            stages[0].module = m_quadFarVertModule;
            dsState.depthTestEnable = VK_TRUE;
            dsState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_planePipelineFar));
        }

        // --- 3. Model 2 Pipelines (Opaque & Glass) ---
        {
            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = m_m2VertModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = m_m2FragModule;
            stages[1].pName = "main";

            std::array<VkVertexInputBindingDescription, 2> bindings{{
                {0, 5 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX}, // pos (xy) + param (zuv)
                {1, sizeof(uint32_t), VK_VERTEX_INPUT_RATE_VERTEX}   // primIndex
            }};

            std::array<VkVertexInputAttributeDescription, 3> attributes{{
                {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},                  // location 0: aPos
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 2 * sizeof(float)}, // location 1: aParam
                {2, 1, VK_FORMAT_R32_UINT, 0}                         // location 2: aPrim
            }};

            VkPipelineVertexInputStateCreateInfo viState{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            viState.vertexBindingDescriptionCount = (uint32_t)bindings.size();
            viState.pVertexBindingDescriptions = bindings.data();
            viState.vertexAttributeDescriptionCount = (uint32_t)attributes.size();
            viState.pVertexAttributeDescriptions = attributes.data();

            VkPipelineInputAssemblyStateCreateInfo iaState{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            // Opaque depth state: test=TRUE, write=TRUE, LESS_OR_EQUAL
            VkPipelineDepthStencilStateCreateInfo dsOpaque{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsOpaque.depthTestEnable = VK_TRUE;
            dsOpaque.depthWriteEnable = VK_TRUE;
            dsOpaque.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            // Opaque blend state: disabled
            VkPipelineColorBlendAttachmentState cbOpaqueAtt{};
            cbOpaqueAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            cbOpaqueAtt.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo cbOpaqueState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbOpaqueState.attachmentCount = 1;
            cbOpaqueState.pAttachments = &cbOpaqueAtt;

            VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeInfo.stageCount = 2;
            pipeInfo.pStages = stages;
            pipeInfo.pVertexInputState = &viState;
            pipeInfo.pInputAssemblyState = &iaState;
            pipeInfo.pViewportState = &vpState;
            pipeInfo.pRasterizationState = &rasterState;
            pipeInfo.pMultisampleState = &msState;
            pipeInfo.pDepthStencilState = &dsOpaque;
            pipeInfo.pColorBlendState = &cbOpaqueState;
            pipeInfo.pDynamicState = &dynamicState;
            pipeInfo.layout = m_m2PipelineLayout;
            pipeInfo.renderPass = renderPass;

            // Cut-outs (trees, fences, decals with holes): alpha-to-coverage spreads the edge over
            // the MSAA samples instead of a binary per-pixel discard that crawls frame to frame.
            VkPipelineMultisampleStateCreateInfo msCut = msState;
            msCut.alphaToCoverageEnable = (m_samples != VK_SAMPLE_COUNT_1_BIT) ? VK_TRUE : VK_FALSE;
            pipeInfo.pMultisampleState = &msCut;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineOpaque));
            stages[1].module = m_m2FragLeanCutModule;   // same state (depth write, A2C), lean cut-out shader
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineCutLean));
            stages[1].module = m_m2FragModule;
            {   // same, depth test but NO depth write: early depth rejection stays on despite discard
                VkPipelineDepthStencilStateCreateInfo dsCut = dsOpaque;
                dsCut.depthWriteEnable = VK_FALSE;
                pipeInfo.pDepthStencilState = &dsCut;
                XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineCutNoWrite));
                pipeInfo.pDepthStencilState = &dsOpaque;
            }
            {   // cut-out DEPTH pre-pass: opacity-only shader, colour masked, A2C coverage -> per-sample depth
                VkPipelineColorBlendAttachmentState noColor = cbOpaqueAtt;
                noColor.colorWriteMask = 0;
                VkPipelineColorBlendStateCreateInfo cbNo = cbOpaqueState;
                cbNo.pAttachments = &noColor;
                pipeInfo.pColorBlendState = &cbNo;
                stages[1].module = m_m2FragCutDModule;
                XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineCutDepth));
                pipeInfo.pColorBlendState = &cbOpaqueState;
                // cut-out COLOUR pass: depth EQUAL, no write, no discard, early tests, no A2C
                VkPipelineDepthStencilStateCreateInfo dsEq = dsOpaque;
                dsEq.depthWriteEnable = VK_FALSE;
                dsEq.depthCompareOp = VK_COMPARE_OP_EQUAL;
                pipeInfo.pDepthStencilState = &dsEq;
                pipeInfo.pMultisampleState = &msState;
                stages[1].module = m_m2FragCutCModule;
                XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineCutColor));
                pipeInfo.pDepthStencilState = &dsOpaque;
                stages[1].module = m_m2FragModule;
            }
            pipeInfo.pMultisampleState = &msState;
            // Same state, fragment shader compiled WITHOUT any discard: the Adreno keeps its
            // early depth rejection (LRZ) on for these, so hidden fragments are never shaded.
            stages[1].module = m_m2FragNdModule;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineFast));
            stages[1].module = m_m2FragLeanModule;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineFastLean));
            stages[1].module = m_m2FragModule;

            // Glass depth state: test=TRUE, write=FALSE, LESS_OR_EQUAL
            VkPipelineDepthStencilStateCreateInfo dsGlass{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dsGlass.depthTestEnable = VK_TRUE;
            dsGlass.depthWriteEnable = VK_FALSE;
            dsGlass.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            // Glass blend state: SRC_ALPHA / ONE_MINUS_SRC_ALPHA
            VkPipelineColorBlendAttachmentState cbGlassAtt{};
            cbGlassAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            cbGlassAtt.blendEnable = VK_TRUE;
            cbGlassAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            cbGlassAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cbGlassAtt.colorBlendOp = VK_BLEND_OP_ADD;
            cbGlassAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            cbGlassAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            cbGlassAtt.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo cbGlassState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cbGlassState.attachmentCount = 1;
            cbGlassState.pAttachments = &cbGlassAtt;

            pipeInfo.pDepthStencilState = &dsGlass;
            pipeInfo.pColorBlendState = &cbGlassState;

            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineGlass));
        }
    }

    void AllocateBuffers() {
        auto createMappedBuffer = [&](BufferAndMemory& bam, VkDeviceSize size, VkBufferUsageFlags usage, void** mapped) {
            VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufInfo.size = size;
            bufInfo.usage = usage;
            bam.Create(m_vkDevice, *m_memAllocator, bufInfo);
            XRC_CHECK_THROW_VKCMD(vkMapMemory(m_vkDevice, bam.mem, 0, size, 0, mapped));
        };

        for (m_fs = 0; m_fs < kFrames; ++m_fs) {
        // 1. UBO buffers: 4 slices (2 eyes x 2 passes)
        for (int e = 0; e < 2; ++e) {
            for (int p = 0; p < 2; ++p) {
                createMappedBuffer(m_uboBufferF[m_fs][e][p], sizeof(M2UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   reinterpret_cast<void**>(&m_uboMappedF[m_fs][e][p]));
            }
        }

        // 2. SSBOs
        m_primsSize = 16384 * sizeof(tcvr_m2_prim);   // = the recorder cap (k_m2_max_prims): Virtua Racing sends ~3 200 quads
        createMappedBuffer(m_primsBufferF[m_fs], m_primsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_primsMappedF[m_fs]));

        m_palramSize = 65536 * sizeof(uint32_t);
        createMappedBuffer(m_palramBufferF[m_fs], m_palramSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_palramMappedF[m_fs]));

        m_colorxlatSize = 16384 * sizeof(uint32_t);
        createMappedBuffer(m_colorxlatBufferF[m_fs], m_colorxlatSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_colorxlatMappedF[m_fs]));

        m_lumaramSize = 4096 * sizeof(uint32_t);
        createMappedBuffer(m_lumaramBufferF[m_fs], m_lumaramSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_lumaramMappedF[m_fs]));

        m_gammaSize = 256 * sizeof(uint32_t);
        createMappedBuffer(m_gammaBufferF[m_fs], m_gammaSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_gammaMappedF[m_fs]));

        // 3. Geometry Buffers
        m_vboSize = 65536 * 5 * sizeof(float); // 32k verts
        createMappedBuffer(m_vboBufferF[m_fs], m_vboSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_vboMappedF[m_fs]));

        m_primIndexSize = 65536 * sizeof(uint32_t);
        createMappedBuffer(m_primIndexBufferF[m_fs], m_primIndexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_primIndexMappedF[m_fs]));

        m_prevSize = 65536 * 4 * sizeof(float);
        createMappedBuffer(m_prevBufferF[m_fs], m_prevSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_prevMappedF[m_fs]));
        std::memset(m_prevMappedF[m_fs], 0, m_prevSize);

        m_iboSize = 131072 * sizeof(uint32_t); // 65k indices
        createMappedBuffer(m_iboBufferF[m_fs], m_iboSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_iboMappedF[m_fs]));

        for (int i = 0; i < 2; ++i) {
            createMappedBuffer(m_layerStagingBufferF[m_fs][i], 512 * 384 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               reinterpret_cast<void**>(&m_layerStagingMappedF[m_fs][i]));
        }
        }
        m_fs = 0;
        void* dummyMapped = nullptr;
        createMappedBuffer(m_dummyBuffer, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dummyMapped);
        {
            void* lm = nullptr;
            createMappedBuffer(m_lutBuffer, 64 * 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &lm);
            m_lutMapped = reinterpret_cast<uint8_t*>(lm);
            std::memset(m_lutMapped, 0, 64 * 32 * 4);
        }
        m_dbgMapped = reinterpret_cast<uint32_t*>(dummyMapped);
        std::memset(m_dbgMapped, 0, 256);

        // 4. Staging Buffers
        for (int i = 0; i < 4; ++i) {
            createMappedBuffer(m_sheetStagingBuffer[i], (i < 2 ? 1u : 2u) * 1024 * 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               reinterpret_cast<void**>(&m_sheetStagingMapped[i]));
        }


    }

    void AllocateTextures() {
        // Sheet textures: 1024x4096, [0..1] R8 t*17, [2..3] RG8 premultiplied cut-out form
        for (int s = 0; s < 4; ++s) {
            const VkFormat sheetFmt = (s < 2) ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
            VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imgInfo.imageType = VK_IMAGE_TYPE_2D;
            imgInfo.extent.width = 1024;
            imgInfo.extent.height = 4096;
            imgInfo.extent.depth = 1;
            imgInfo.mipLevels = 1;
            imgInfo.arrayLayers = 1;
            imgInfo.format = sheetFmt;
            imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &imgInfo, nullptr, &m_sheetImage[s]));

            VkMemoryRequirements memReq{};
            vkGetImageMemoryRequirements(m_vkDevice, m_sheetImage[s], &memReq);
            m_memAllocator->Allocate(memReq, &m_sheetMem[s], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, m_sheetImage[s], m_sheetMem[s], 0));

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = m_sheetImage[s];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = sheetFmt;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &m_sheetView[s]));
            m_sheetLayout[s] = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        // 2D Layer textures: 2 images (front2d, back2d), 512x384, B8G8R8A8_UNORM
        for (int l = 0; l < 2; ++l) {
            VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imgInfo.imageType = VK_IMAGE_TYPE_2D;
            imgInfo.extent.width = 512;
            imgInfo.extent.height = 384;
            imgInfo.extent.depth = 1;
            imgInfo.mipLevels = 1;
            imgInfo.arrayLayers = 1;
            imgInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
            imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &imgInfo, nullptr, &m_layerImage[l]));

            VkMemoryRequirements memReq{};
            vkGetImageMemoryRequirements(m_vkDevice, m_layerImage[l], &memReq);
            m_memAllocator->Allocate(memReq, &m_layerMem[l], VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, m_layerImage[l], m_layerMem[l], 0));

            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = m_layerImage[l];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_B8G8R8A8_UNORM;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &m_layerView[l]));
            m_layerLayout[l] = VK_IMAGE_LAYOUT_UNDEFINED;

            // Allocate descriptor set for each layer
            VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocInfo.descriptorPool = m_descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &m_planeDescLayout;
            XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_layerDescSet[l]));

            VkDescriptorImageInfo descImgInfo{};
            descImgInfo.sampler = m_layerSampler[l];
            descImgInfo.imageView = m_layerView[l];
            descImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkWriteDescriptorSet writeDesc{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writeDesc.dstSet = m_layerDescSet[l];
            writeDesc.dstBinding = 0;
            writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writeDesc.descriptorCount = 1;
            writeDesc.pImageInfo = &descImgInfo;
            vkUpdateDescriptorSets(m_vkDevice, 1, &writeDesc, 0, nullptr);
        }

        // Allocate Model 2 descriptor sets (2 eyes x 2 passes = 4 sets) per frame slot
        for (m_fs = 0; m_fs < kFrames; ++m_fs)
        for (int e = 0; e < 2; ++e) {
            for (int p = 0; p < 2; ++p) {
                VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocInfo.descriptorPool = m_descriptorPool;
                allocInfo.descriptorSetCount = 1;
                allocInfo.pSetLayouts = &m_m2DescLayout;
                XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_m2DescSetF[m_fs][e][p]));

                std::vector<VkWriteDescriptorSet> writes;

                // Binding 0: UBO
                VkDescriptorBufferInfo uboInfo{m_uboBufferF[m_fs][e][p].buf, 0, sizeof(M2UniformBufferObject)};
                VkWriteDescriptorSet uboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                uboWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                uboWrite.dstBinding = 0;
                uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                uboWrite.descriptorCount = 1;
                uboWrite.pBufferInfo = &uboInfo;
                writes.push_back(uboWrite);

                // Binding 1: Prims
                VkDescriptorBufferInfo primsInfo{m_primsBufferF[m_fs].buf, 0, m_primsSize};
                VkWriteDescriptorSet primsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                primsWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                primsWrite.dstBinding = 1;
                primsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                primsWrite.descriptorCount = 1;
                primsWrite.pBufferInfo = &primsInfo;
                writes.push_back(primsWrite);

                // Binding 2: Palram
                VkDescriptorBufferInfo palramInfo{m_palramBufferF[m_fs].buf, 0, m_palramSize};
                VkWriteDescriptorSet palramWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                palramWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                palramWrite.dstBinding = 2;
                palramWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                palramWrite.descriptorCount = 1;
                palramWrite.pBufferInfo = &palramInfo;
                writes.push_back(palramWrite);

                // Binding 3: Colorxlat
                VkDescriptorBufferInfo colInfo{m_colorxlatBufferF[m_fs].buf, 0, m_colorxlatSize};
                VkWriteDescriptorSet colWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                colWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                colWrite.dstBinding = 3;
                colWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                colWrite.descriptorCount = 1;
                colWrite.pBufferInfo = &colInfo;
                writes.push_back(colWrite);

                // Binding 4: Lumaram
                VkDescriptorBufferInfo lumInfo{m_lumaramBufferF[m_fs].buf, 0, m_lumaramSize};
                VkWriteDescriptorSet lumWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                lumWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                lumWrite.dstBinding = 4;
                lumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                lumWrite.descriptorCount = 1;
                lumWrite.pBufferInfo = &lumInfo;
                writes.push_back(lumWrite);

                // Binding 5: Gamma
                VkDescriptorBufferInfo gamInfo{m_gammaBufferF[m_fs].buf, 0, m_gammaSize};
                VkWriteDescriptorSet gamWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                gamWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                gamWrite.dstBinding = 5;
                gamWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                gamWrite.descriptorCount = 1;
                gamWrite.pBufferInfo = &gamInfo;
                writes.push_back(gamWrite);

                // Binding 16: previous vertex positions (smooth motion)
                VkDescriptorBufferInfo prevInfo{m_prevBufferF[m_fs].buf, 0, m_prevSize};
                VkWriteDescriptorSet prevWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                prevWrite.dstSet = m_m2DescSetF[m_fs][e][p];
                prevWrite.dstBinding = 16;
                prevWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                prevWrite.descriptorCount = 1;
                prevWrite.pBufferInfo = &prevInfo;
                writes.push_back(prevWrite);

                // Binding 6, 7: Dummy SSBOs
                VkDescriptorBufferInfo dummyInfo{m_dummyBuffer.buf, 0, 256};
                VkWriteDescriptorSet dum6Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                dum6Write.dstSet = m_m2DescSetF[m_fs][e][p];
                dum6Write.dstBinding = 6;
                dum6Write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dum6Write.descriptorCount = 1;
                dum6Write.pBufferInfo = &dummyInfo;
                writes.push_back(dum6Write);

                VkWriteDescriptorSet dum7Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                dum7Write.dstSet = m_m2DescSetF[m_fs][e][p];
                dum7Write.dstBinding = 7;
                // binding 7 = the colour table as a storage buffer (uint RGBA per (component, luma))
                VkDescriptorBufferInfo lutInfo{m_lutBuffer.buf, 0, 64 * 32 * 4};
                dum7Write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dum7Write.descriptorCount = 1;
                dum7Write.pBufferInfo = &lutInfo;
                writes.push_back(dum7Write);

                // Binding 8, 9: Sheet texture samplers
                VkDescriptorImageInfo sheet0Info{m_sheetSampler, m_sheetView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet s0Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                s0Write.dstSet = m_m2DescSetF[m_fs][e][p];
                s0Write.dstBinding = 8;
                s0Write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                s0Write.descriptorCount = 1;
                s0Write.pImageInfo = &sheet0Info;
                writes.push_back(s0Write);

                VkDescriptorImageInfo sheet1Info{m_sheetSampler, m_sheetView[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet s1Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                s1Write.dstSet = m_m2DescSetF[m_fs][e][p];
                s1Write.dstBinding = 9;
                s1Write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                s1Write.descriptorCount = 1;
                s1Write.pImageInfo = &sheet1Info;
                writes.push_back(s1Write);

                VkDescriptorImageInfo cut0Info{m_sheetSampler, m_sheetView[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkDescriptorImageInfo cut1Info{m_sheetSampler, m_sheetView[3], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet c0Write = s1Write; c0Write.dstBinding = 10; c0Write.pImageInfo = &cut0Info;
                VkWriteDescriptorSet c1Write = s1Write; c1Write.dstBinding = 11; c1Write.pImageInfo = &cut1Info;
                writes.push_back(c0Write);
                writes.push_back(c1Write);

                vkUpdateDescriptorSets(m_vkDevice, (uint32_t)writes.size(), writes.data(), 0, nullptr);
            }
        }
        m_fs = 0;
    }

    // gamma(colorxlat) for every (channel, 5-bit component, luma): 3 texel reads per pixel
    // instead of 6 dependent storage-buffer reads. Rebuilt only when the tables change.
    void BuildColourLut(const tcvr_m2_frame& frame) {
        if (!frame.colorxlat || !frame.gamma || frame.colorxlat_entries < 0x6000u || frame.gamma_entries < 256u) return;
        const size_t cxBytes = 0x6000u * sizeof(uint16_t);
        if (m_lutValid && std::memcmp(m_lutSrcCx.data(), frame.colorxlat, cxBytes) == 0 &&
            std::memcmp(m_lutSrcGamma.data(), frame.gamma, 256) == 0)
            return;
        m_lutSrcCx.assign(reinterpret_cast<const uint8_t*>(frame.colorxlat), reinterpret_cast<const uint8_t*>(frame.colorxlat) + cxBytes);
        m_lutSrcGamma.assign(frame.gamma, frame.gamma + 256);
        uint8_t lut[64 * 32 * 4];
        for (uint32_t comp = 0; comp < 32; ++comp)
            for (uint32_t luma = 0; luma < 64; ++luma) {
                uint8_t* px = lut + (comp * 64 + luma) * 4;
                for (uint32_t c = 0; c < 3; ++c) {
                    const uint32_t idx = c * 0x2000u + (comp << 8) + luma;
                    px[c] = frame.gamma[frame.colorxlat[idx] & 0xffu];
                }
                px[3] = 255;
            }
        m_regions.SetLut(lut);
        if (m_lutMapped) std::memcpy(m_lutMapped, lut, sizeof(lut));  // SSBO copy (binding 7), fenced
        m_lutValid = true;
        static unsigned s_lutBuilds = 0;
        if ((s_lutBuilds++ % 60u) == 0u) Log::Write(Log::Level::Info, Fmt("TCVR_M2VK colour LUT rebuilt (%u)", s_lutBuilds));
    }

    void UploadColourChain(const tcvr_m2_frame& frame) {
        if (frame.palram && frame.palram_entries) {
            size_t bytes = std::min(size_t(frame.palram_entries) * sizeof(uint16_t), m_palramSize);
            memcpy(m_palramMappedF[m_fs], frame.palram, bytes);
        }
        m_gammaFolded = arcadexr::config::GetInt("m2.foldGamma", 1) != 0 && frame.gamma && frame.gamma_entries >= 256;
        if (frame.colorxlat && frame.colorxlat_entries) {
            size_t bytes = std::min(size_t(frame.colorxlat_entries) * sizeof(uint16_t), m_colorxlatSize);
            if (m_gammaFolded) {
                // gamma folded INTO colorxlat (same buffer, same shader code shape): one dependent
                // read level less per pixel -- the shader then skips gamma8().
                uint16_t* dst = reinterpret_cast<uint16_t*>(m_colorxlatMappedF[m_fs]);
                const size_t n = bytes / sizeof(uint16_t);
                for (size_t i = 0; i < n; ++i) dst[i] = frame.gamma[frame.colorxlat[i] & 0xffu];
            } else {
                memcpy(m_colorxlatMappedF[m_fs], frame.colorxlat, bytes);
            }
        }
        if (frame.lumaram && frame.lumaram_entries) {
            size_t bytes = std::min(size_t(frame.lumaram_entries) * sizeof(uint8_t), m_lumaramSize);
            memcpy(m_lumaramMappedF[m_fs], frame.lumaram, bytes);
        }
        if (frame.gamma && frame.gamma_entries) {
            size_t bytes = std::min(size_t(frame.gamma_entries) * sizeof(uint8_t), m_gammaSize);
            memcpy(m_gammaMappedF[m_fs], frame.gamma, bytes);
        }
    }

    // 0 = unchanged, 1 = full upload (regions must be rebuilt), 2 = partial update done in place.
    int UploadTextures(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        if (!frame.textureram[0] || !frame.textureram[1] || frame.textureram_words == 0) return 0;

        // Model 2B (Sega Rally) does not update dirty_generation.
        // Fingerprint both sheets using FNV-1a hash (sampling every 4 words)
        // exactly matching gpu_renderer.cpp lines 716-724.
        bool hashChanged = false;
        {
            uint64_t h = 1469598103934665603ull;
            for (int sheet = 0; sheet < 2; sheet++) {
                const uint32_t* w = frame.textureram[sheet];
                const uint32_t n = frame.textureram_words;
                if (!w) continue;
                for (uint32_t i = 0; i < n; i += 4) { h ^= w[i]; h *= 1099511628211ull; }
            }
            if (h != m_texHash) {
                hashChanged = m_texturesUploaded;
                m_texHash = h;
            }
        }

        {   // debug.tcvr.m2_texReupload=<n>: force one FULL upload of both sheets (each new value triggers once)
            const int ru = arcadexr::config::GetInt("m2.texReupload", 0);
            if (ru != m_lastTexReupload) { m_lastTexReupload = ru; if (ru != 0) { m_texShadow[0].clear(); m_texShadow[1].clear(); hashChanged = true; } }
        }
        // The sampled hash (1 word in 4) missed rewrites of the other words: The House of the Dead's tiled floor
        // stayed on the previous scene's texels in the sheets (25/09). Once uploaded, the block comparison below is
        // the change detector (4 MB of memcmp, ~0.4 ms); the hash only decides the very first upload.
        const bool needsUpload = !m_texturesUploaded || hashChanged ||
                                 (m_texShadow[0].size() == std::min(frame.textureram_words, 524288u));
        if (!needsUpload) return 0;

        // Partial update (24/09): Super GT 24h rewrites a few 4 KB blocks of texture RAM every frame (Model 2B/2C
        // texture RAM is plain shared RAM: no write handler, no dirty marks). A full upload + every region dropped +
        // the frame rebuilt cost ~35 ms a frame (30-40 fps in the headset). Compare with the previous copy block
        // by block; up to a quarter of the blocks changed -> only those rows go up, only the regions reading them
        // are re-filled. More (a course / menu load) -> the full path below.
        const uint32_t totalWords = std::min(frame.textureram_words, 524288u);
        if (m_texturesUploaded && m_texShadow[0].size() == totalWords && m_texShadow[1].size() == totalWords) {
            constexpr uint32_t kBlock = 1024;   // words: 8 rows of the 1024-texel sheet
            std::vector<uint32_t> dirty[2];
            size_t nDirty = 0;
            for (int sheet = 0; sheet < 2; ++sheet)
                for (uint32_t b = 0; b * kBlock < totalWords; ++b) {
                    const uint32_t n = std::min(kBlock, totalWords - b * kBlock);
                    if (std::memcmp(frame.textureram[sheet] + b * kBlock, m_texShadow[sheet].data() + b * kBlock, n * 4) != 0) {
                        dirty[sheet].push_back(b); ++nDirty;
                    }
                }
            const size_t nBlocks = 2 * ((totalWords + kBlock - 1) / kBlock);
            if (nDirty == 0) return 0;
            if (nDirty * 4 <= nBlocks) {
                for (int sheet = 0; sheet < 2; ++sheet) {
                    if (dirty[sheet].empty()) continue;
                    const uint32_t* words = frame.textureram[sheet];
                    std::vector<uint8_t>& cpu = m_sheetCpu[sheet];
                    uint8_t* dst = reinterpret_cast<uint8_t*>(m_sheetStagingMapped[sheet]);
                    uint8_t* cut = reinterpret_cast<uint8_t*>(m_sheetStagingMapped[sheet + 2]);
                    std::vector<uint8_t> rowDirty(4096, 0);
                    for (uint32_t b : dirty[sheet]) {
                        const uint32_t w0 = b * kBlock, w1 = std::min(totalWords, w0 + kBlock);
                        std::memcpy(m_texShadow[sheet].data() + w0, words + w0, size_t(w1 - w0) * 4);
                        for (uint32_t word_idx = w0; word_idx < w1; ++word_idx) {
                            const uint32_t w = words[word_idx];
                            for (uint32_t half = 0; half < 2; ++half) {
                                const uint32_t off = word_idx * 2 + half;
                                const uint32_t x0 = (off % 512) * 2, y0 = (off / 512) * 2;
                                const uint32_t hw = (w >> (16 * half)) & 0xffff;
                                cpu[y0 * 1024 + x0] = (hw >> 12) & 0xf;
                                cpu[y0 * 1024 + x0 + 1] = (hw >> 8) & 0xf;
                                cpu[(y0 + 1) * 1024 + x0] = (hw >> 4) & 0xf;
                                cpu[(y0 + 1) * 1024 + x0 + 1] = hw & 0xf;
                            }
                        }
                        for (uint32_t row = b * 8; row < b * 8 + 8 && row < 4096; ++row) {
                            rowDirty[row] = 1;
                            for (uint32_t x = 0; x < 1024; ++x) {
                                const size_t i = size_t(row) * 1024 + x;
                                const uint8_t t = cpu[i];
                                dst[i] = uint8_t(t * 17);
                                cut[2 * i] = (t == 15) ? 0 : uint8_t(t * 17);
                                cut[2 * i + 1] = (t == 15) ? 0 : 255;
                            }
                        }
                    }
                    for (int img = sheet; img < 4; img += 2) {
                        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.oldLayout = m_sheetLayout[img];
                        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barrier.image = m_sheetImage[img];
                        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                        const size_t bpp = (img >= 2) ? 2 : 1;
                        for (size_t k = 0; k < dirty[sheet].size();) {   // one copy per run of contiguous blocks
                            size_t e = k + 1;
                            while (e < dirty[sheet].size() && dirty[sheet][e] == dirty[sheet][e - 1] + 1) ++e;
                            const uint32_t r0 = dirty[sheet][k] * 8, r1 = std::min(4096u, (dirty[sheet][e - 1] + 1) * 8);
                            VkBufferImageCopy region{};
                            region.bufferOffset = VkDeviceSize(r0) * 1024 * bpp;
                            region.bufferRowLength = 1024;
                            region.bufferImageHeight = r1 - r0;
                            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                            region.imageOffset = {0, int32_t(r0), 0};
                            region.imageExtent = {1024, r1 - r0, 1};
                            vkCmdCopyBufferToImage(cmd, m_sheetStagingBuffer[img].buf, m_sheetImage[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                            k = e;
                        }
                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
                        m_sheetLayout[img] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    }
                    m_regions.InvalidateRows(uint32_t(sheet), rowDirty);
                }
                m_regionHasHoles.clear();
                if ((++m_partialTexLog % 120u) == 0u)
                    Log::Write(Log::Level::Info, Fmt("TCVR_M2VK partial texture update: %zu of %zu blocks", nDirty, nBlocks));
                return 2;
            }
        }

        // The sheet staging buffers are shared by both frame slots: let any in-flight copy finish (rare: loads).
        if (m_texturesUploaded) vkDeviceWaitIdle(m_vkDevice);
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK: Uploading texture sheets (hashChanged=%d, words=%u)",
                                         hashChanged ? 1 : 0, frame.textureram_words));

        for (int sheet = 0; sheet < 2; sheet++) {
            const uint32_t* words = frame.textureram[sheet];
            const uint32_t wordCount = frame.textureram_words;
            if (!words || wordCount == 0) continue;

            std::vector<uint8_t>& cpu = m_sheetCpu[sheet];
            cpu.assign(size_t(1024) * 4096, 0);
            const uint32_t totalWords = std::min(wordCount, 524288u);
            for (uint32_t word_idx = 0; word_idx < totalWords; ++word_idx) {
                const uint32_t w = words[word_idx];
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = word_idx * 2 + half;
                    const uint32_t x0 = (off % 512) * 2, y0 = (off / 512) * 2;
                    const uint32_t hw = (w >> (16 * half)) & 0xffff;
                    cpu[y0 * 1024 + x0] = (hw >> 12) & 0xf;
                    cpu[y0 * 1024 + x0 + 1] = (hw >> 8) & 0xf;
                    cpu[(y0 + 1) * 1024 + x0] = (hw >> 4) & 0xf;
                    cpu[(y0 + 1) * 1024 + x0 + 1] = hw & 0xf;
                }
            }
            uint8_t* dst = reinterpret_cast<uint8_t*>(m_sheetStagingMapped[sheet]);
            uint8_t* cut = reinterpret_cast<uint8_t*>(m_sheetStagingMapped[sheet + 2]);
            for (size_t i = 0; i < cpu.size(); ++i) {
                const uint8_t t = cpu[i];
                dst[i] = uint8_t(t * 17);
                cut[2 * i] = (t == 15) ? 0 : uint8_t(t * 17);
                cut[2 * i + 1] = (t == 15) ? 0 : 255;
            }
            for (int img = sheet; img < 4; img += 2) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = (m_sheetLayout[img] == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = m_sheetLayout[img];
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.image = m_sheetImage[img];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd,
                                 (m_sheetLayout[img] == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkBufferImageCopy region{};
            region.bufferRowLength = 1024;
            region.bufferImageHeight = 4096;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {1024, 4096, 1};
            vkCmdCopyBufferToImage(cmd, m_sheetStagingBuffer[img].buf, m_sheetImage[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            m_sheetLayout[img] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        for (int sheet = 0; sheet < 2; ++sheet)
            m_texShadow[sheet].assign(frame.textureram[sheet], frame.textureram[sheet] + totalWords);
        m_texturesUploaded = true;
        m_sheetGeneration = frame.dirty_generation;
        m_regionHasHoles.clear();
        return 1;
    }

    void UploadLayers(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        // 0 = front2d, 1 = back2d
        auto uploadOne = [&](int idx, const uint32_t* pixels, uint32_t stride, int w, int h) {
            if (!pixels || w <= 0 || h <= 0 || stride == 0) {
                m_haveLayer[idx] = false;
                return;
            }
            const uint32_t copyW = std::min(uint32_t(w), 512u);
            const uint32_t copyH = std::min(uint32_t(h), 384u);
            if (idx == 1) {
                // Haze colour for the pop-in fade: the sky just above the game's horizon (middle half of the row).
                const float hr = (m_horizonGeo >= 0.0f) ? m_horizonGeo : 100.0f;
                const uint32_t row = uint32_t(std::max(0.0f, std::min(float(copyH - 1), hr - 6.0f)));
                float acc[3] = {0, 0, 0}; int n = 0;
                for (uint32_t x = copyW / 4; x < copyW * 3 / 4; x += 4) {
                    const uint32_t px = pixels[row * stride + x];
                    if (px == 0u) continue;
                    acc[0] += float((px >> 16) & 0xffu); acc[1] += float((px >> 8) & 0xffu); acc[2] += float(px & 0xffu); ++n;
                }
                if (n > 0) for (int k = 0; k < 3; ++k) m_fogColour[k] += (acc[k] / (255.0f * float(n)) - m_fogColour[k]) * 0.2f;
            }
            if (idx == 0) {
                // A front layer that covers (nearly) the whole screen is a flash (the gun's white flash: the
                // cabinet's optical gun needs the whole screen lit) or a full-screen page, not a HUD. On the arcade
                // plane it showed as a white square in front of the player (Guillaume, 23/09): flag it, the front
                // pass then enlarges it over the whole view.
                uint32_t opaque = 0, white = 0, total = 0;
                for (uint32_t y = 0; y < copyH; y += 8)
                    for (uint32_t x = 0; x < copyW; x += 8, ++total) {
                        const uint32_t px = pixels[y * stride + x];
                        if (px == 0u) continue;
                        ++opaque;
                        if (((px >> 16) & 0xffu) > 200u && ((px >> 8) & 0xffu) > 200u && (px & 0xffu) > 200u) ++white;
                    }
                // the flash only: nearly all of the screen, nearly all white (a title page on black is not enlarged)
                m_frontFullscreen = total > 0 && opaque * 10u >= total * 8u && white * 10u >= total * 8u;
                // Tiles that hold HUD texels (25/09): the front pass is a full-view plane whose fragments cost even where
                // the layer is transparent -- most of it. Only 16x16 tiles with an opaque texel are drawn (runs per row,
                // scissored), same pixels, a fraction of the fragments.
                m_frontRuns.clear();
                for (uint32_t ty = 0; ty * 16u < copyH; ++ty) {
                    int runStart = -1;
                    for (uint32_t tx = 0; tx * 16u <= copyW; ++tx) {
                        bool any = false;
                        if (tx * 16u < copyW)
                            for (uint32_t y = ty * 16u; y < std::min(copyH, ty * 16u + 16u) && !any; ++y)
                                for (uint32_t x = tx * 16u; x < std::min(copyW, tx * 16u + 16u); ++x)
                                    if (pixels[y * stride + x] != 0u) { any = true; break; }
                        if (any && runStart < 0) runStart = int(tx);
                        if (!any && runStart >= 0) { m_frontRuns.push_back({uint16_t(runStart * 16), uint16_t(ty * 16u), uint16_t(std::min(copyW, tx * 16u)), uint16_t(std::min(copyH, ty * 16u + 16u))}); runStart = -1; }
                    }
                }
                m_frontRunsW = copyW; m_frontRunsH = copyH;
                static unsigned s_ffLog = 0;
                if (m_frontFullscreen && (s_ffLog++ % 30u) == 0u)
                    Log::Write(Log::Level::Info, Fmt("TCVR_M2VK front layer full-screen (%u/%u opaque): drawn over the whole view", opaque, total));
            }
            for (uint32_t y = 0; y < copyH; ++y) {
                memcpy(&m_layerStagingMappedF[m_fs][idx][y * 512], &pixels[y * stride], copyW * sizeof(uint32_t));
            }

            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = (m_layerLayout[idx] == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = m_layerLayout[idx];
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.image = m_layerImage[idx];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd,
                                 (m_layerLayout[idx] == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 512;
            region.bufferImageHeight = 384;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {copyW, copyH, 1};
            vkCmdCopyBufferToImage(cmd, m_layerStagingBufferF[m_fs][idx].buf, m_layerImage[idx],
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            m_layerLayout[idx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            m_haveLayer[idx] = true;
            m_layerUvScaleX[idx] = float(copyW) / 512.0f;
        };

        uploadOne(0, frame.front2d, frame.front2d_stride, frame.width, frame.height);
        uploadOne(1, frame.back2d, frame.back2d_stride, frame.width, frame.height);
    }

private:
    bool m_initialized = false;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    const MemoryAllocator* m_memAllocator = nullptr;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_m2DescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_planeDescLayout = VK_NULL_HANDLE;

    VkPipelineLayout m_m2PipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_voidPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_planePipelineLayout = VK_NULL_HANDLE;

    VkShaderModule m_quadVertModule = VK_NULL_HANDLE;
    VkShaderModule m_voidFragModule = VK_NULL_HANDLE;
    VkShaderModule m_planeFragModule = VK_NULL_HANDLE;
    VkShaderModule m_m2VertModule = VK_NULL_HANDLE;
    VkShaderModule m_m2FragModule = VK_NULL_HANDLE;
    VkShaderModule m_m2FragNdModule = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineFast = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineCutNoWrite = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineCutDepth = VK_NULL_HANDLE, m_m2PipelineCutColor = VK_NULL_HANDLE;
    VkShaderModule m_m2FragCutDModule = VK_NULL_HANDLE, m_m2FragCutCModule = VK_NULL_HANDLE;
    int m_cutMode = 2;
    VkPipeline m_m2PipelineFastLean = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineCutLean = VK_NULL_HANDLE;
    VkShaderModule m_m2FragLeanCutModule = VK_NULL_HANDLE;
    VkShaderModule m_m2FragLeanModule = VK_NULL_HANDLE;
    unsigned m_secIndexStart = 0;
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_samples = VK_SAMPLE_COUNT_1_BIT;
    VkRenderPass m_pass = VK_NULL_HANDLE;
    bool m_useFdm = false;
    EyeTargets m_eyeTargets[3];   // [2] = the flat (screen presentation) target
    std::vector<FbEntry> m_fbs;
    VkShaderModule m_quadFarVertModule = VK_NULL_HANDLE;
    VkPipeline m_voidPipelineFar = VK_NULL_HANDLE;
    VkPipeline m_planePipelineFar = VK_NULL_HANDLE;
    uint32_t m_fastIndexCount = 0;
    uint32_t m_leanIndexCount = 0;
    uint32_t m_viewDiagTick = 0;
    struct MainLike { std::int32_t l, t, r, b, cx, cy; };
    std::vector<MainLike> m_mainLike;   // full-screen, centred views of this frame (main camera candidates)   // [0, lean) lean shader; [lean, fast) full shader (no region image nor layer)

    VkPipeline m_voidPipeline = VK_NULL_HANDLE;
    VkPipeline m_planePipeline = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineOpaque = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineGlass = VK_NULL_HANDLE;

    VkSampler m_sheetSampler = VK_NULL_HANDLE;
    VkSampler m_layerSampler[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Host-visible mapped buffers
    // Two frames in flight: everything the CPU rewrites every frame exists twice, indexed by
    // m_fs (the frame slot the plugin is recording). The GPU reads slot s while the CPU fills 1-s.
    static constexpr int kFrames = 2;
    int m_fs = 0;
    std::vector<uint32_t> m_pendSlots[kFrames];
    bool m_fullPend[kFrames] = {true, true};
    BufferAndMemory m_uboBufferF[kFrames][2][2]; // [eye][pass]
    M2UniformBufferObject* m_uboMappedF[kFrames][2][2] = {};

    BufferAndMemory m_primsBufferF[kFrames];
    tcvr_m2_prim* m_primsMappedF[kFrames] = {};
    size_t m_primsSize = 0;

    BufferAndMemory m_palramBufferF[kFrames];
    uint32_t* m_palramMappedF[kFrames] = {};
    size_t m_palramSize = 0;

    BufferAndMemory m_colorxlatBufferF[kFrames];
    uint32_t* m_colorxlatMappedF[kFrames] = {};
    size_t m_colorxlatSize = 0;

    BufferAndMemory m_lumaramBufferF[kFrames];
    uint32_t* m_lumaramMappedF[kFrames] = {};
    size_t m_lumaramSize = 0;

    BufferAndMemory m_gammaBufferF[kFrames];
    uint32_t* m_gammaMappedF[kFrames] = {};
    size_t m_gammaSize = 0;

    BufferAndMemory m_dummyBuffer;
    uint32_t* m_dbgMapped = nullptr;
    BufferAndMemory m_lutBuffer;
    uint8_t* m_lutMapped = nullptr;
    uint32_t m_lastEyePixels = 0;
    bool m_built = false;
    int m_lastRegionsReset = 0;
    int m_lastRb = 0, m_rbWaitFrames = 0;
    bool m_rbPending = false, m_rbPendingNext = false;
    bool m_edgeFadeOff = true;
    bool m_gammaFolded = false;

    BufferAndMemory m_vboBufferF[kFrames];
    BufferAndMemory m_prevBufferF[kFrames];
    float* m_prevMappedF[kFrames] = {};
    size_t m_prevSize = 0;
    std::vector<float> m_prevPosCpu;                         // per vertex: previous x, y, z, matched
    std::vector<float> m_newestPos;                          // positions of the newest changed frame, by vertex
    struct MotionQuad { uint32_t key, obj; uint32_t v0, n; float c[3]; float p[4][3]; };
    std::vector<MotionQuad> m_lastQuads, m_curQuads;         // sorted by key
    std::vector<std::pair<const MotionQuad*, const MotionQuad*>> m_motionPairs;
    uint32_t m_motionDiagTick = 0;

    // Match every quad of the new frame with the same quad in the previous changed frame. Copies of a model share
    // one key (road sections, trackside objects): estimate the game camera's rigid motion on the keys seen exactly
    // once in both frames, then give each quad the copy that lands nearest once moved by it. An occurrence number
    // in MAME did not work: it shifts by one whenever a copy leaves the view (24/09, half the scene mismatched).
    // Model 2 smooth motion by the GAME's matrices (25/09). Each raw vertex is focus(M * object vertex), M the object
    // matrix the geometriser used (captured per prim). One frame earlier the same object (address + copy) had M_prev:
    // prev = focus_prev(M_prev * M^-1 * unfocus(vertex)). Exact motion, nothing estimated: no cracks (the whole
    // scenery shares the camera part of its matrices), no class to decide, nothing on a menu (no main view), and an
    // object absent from the previous frame (a wheel model swapped to spin it) simply is not blended.
    struct ObjMat { float m[14]; };

    // ---- Camera delta for Application SpaceWarp (26/09) ------------------------------------------------------------
    // ONE motion for the whole picture, the game camera's, so the headset reprojects every pixel alike (no per-object
    // guess: no isolated flash is possible). Measured on the objects whose identity is certain (one copy of the model
    // in both frames, main view): for every static object prev = C_prev W, cur = C_cur W, so prev * cur^-1 is the
    // same camera delta D for all of them. The D the most objects agree on wins; too few agree -> no delta this frame.
    bool m_camDeltaWanted = false;
    std::unordered_map<uint32_t, ObjMat> m_cdPrev;   // object address -> matrix, previous changed frame
    float m_cdR[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, m_cdT[3] = {0, 0, 0};   // p_prev = R p_cur + T (camera space, arcade units)
    bool m_cdValid = false;
    uint32_t m_cdSeq = 0, m_cdAgree = 0, m_cdPairs = 0;
    arcadexr::gun::Vec3 m_a2wRight{}, m_a2wUp{}, m_a2wNormal{}, m_a2wCam{};
    float m_a2wScale = 1.0f, m_depthNear = 0.25f, m_depthFar = 20000.0f;
    bool m_a2wValid = false;

    static bool Inverse3(const float* c, float* o) {   // column-major 3x3 (m0..m8)
        const float a = c[0], b = c[3], cc = c[6], d = c[1], e = c[4], f = c[7], g = c[2], h = c[5], i = c[8];
        const float det = a * (e * i - f * h) - b * (d * i - f * g) + cc * (d * h - e * g);
        if (std::fabs(det) < 1e-12f) return false;
        const float id = 1.0f / det;
        o[0] = (e * i - f * h) * id; o[3] = (cc * h - b * i) * id; o[6] = (b * f - cc * e) * id;
        o[1] = (f * g - d * i) * id; o[4] = (a * i - cc * g) * id; o[7] = (cc * d - a * f) * id;
        o[2] = (d * h - e * g) * id; o[5] = (b * g - a * h) * id; o[8] = (a * e - b * d) * id;
        return true;
    }
    static void Mul3(const float* x, const float* y, float* o) {   // o = x * y, column-major
        for (int c = 0; c < 3; ++c)
            for (int rr = 0; rr < 3; ++rr) o[c * 3 + rr] = x[rr] * y[c * 3] + x[3 + rr] * y[c * 3 + 1] + x[6 + rr] * y[c * 3 + 2];
    }

    void CameraDelta(const tcvr_m2_frame& frame) {
        ++m_cdSeq;
        m_cdValid = false;
        std::unordered_map<uint32_t, ObjMat> cur;
        std::unordered_map<uint32_t, uint32_t> serials;   // addr -> 1 + highest serial seen (copies)
        if (!frame.raw_motion || !m_haveMainView) { m_cdPrev.clear(); return; }
        for (size_t k = 0; k < m_rawPrims.size() && k < m_rawPrimSrc.size(); ++k) {
            const tcvr_m2_prim& p = m_rawPrims[k];
            if (p.center_x != m_mainCenter[0] || p.center_y != m_mainCenter[1] ||
                std::abs(p.clip_l - m_mainClip[0]) > 2 || std::abs(p.clip_t - m_mainClip[1]) > 2 ||
                std::abs(p.clip_r - m_mainClip[2]) > 2 || std::abs(p.clip_b - m_mainClip[3]) > 2) continue;
            const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
            if (mo[14] < 0.5f) continue;
            uint32_t& n = serials[p.motion_addr];
            n = std::max(n, p.motion_serial + 1u);
            ObjMat om; for (int i = 0; i < 14; ++i) om.m[i] = mo[i];
            cur.emplace(p.motion_addr, om);
        }
        struct Cand { float R[9], T[3]; };
        std::vector<Cand> cands;
        for (const auto& kv : cur) {
            if (serials[kv.first] != 1u) continue;          // several copies: identity not certain
            auto it = m_cdPrev.find(kv.first);
            if (it == m_cdPrev.end()) continue;
            const float* C = kv.second.m; const float* P = it->second.m;
            float Ci[9];
            if (!Inverse3(C, Ci)) continue;
            Cand c;
            Mul3(P, Ci, c.R);
            for (int rr = 0; rr < 3; ++rr)
                c.T[rr] = P[9 + rr] - (c.R[rr] * C[9] + c.R[3 + rr] * C[10] + c.R[6 + rr] * C[11]);
            cands.push_back(c);
        }
        m_cdPrev.clear();
        for (const auto& kv : cur) if (serials[kv.first] == 1u) m_cdPrev.emplace(kv.first, kv.second);
        m_cdPairs = uint32_t(cands.size());
        size_t best = 0, bestN = 0;
        for (size_t i = 0; i < cands.size(); ++i) {
            size_t n = 0;
            const float tol = 0.02f + 0.01f * std::sqrt(cands[i].T[0] * cands[i].T[0] + cands[i].T[1] * cands[i].T[1] + cands[i].T[2] * cands[i].T[2]);
            for (size_t j = 0; j < cands.size(); ++j) {
                float dr = 0.0f, dt = 0.0f;
                for (int q = 0; q < 9; ++q) dr = std::max(dr, std::fabs(cands[i].R[q] - cands[j].R[q]));
                for (int q = 0; q < 3; ++q) dt = std::max(dt, std::fabs(cands[i].T[q] - cands[j].T[q]));
                if (dr < 0.002f && dt < tol) ++n;
            }
            if (n > bestN) { bestN = n; best = i; }
        }
        m_cdAgree = uint32_t(bestN);
        if (bestN >= 6 && bestN * 10 >= cands.size() * 3) {
            std::memcpy(m_cdR, cands[best].R, sizeof m_cdR);
            std::memcpy(m_cdT, cands[best].T, sizeof m_cdT);
            m_cdValid = true;
        }
    }
  public:
    void SetCameraDeltaWanted(bool on) { m_camDeltaWanted = on; }
    void DepthRange(float* n, float* f) const { *n = m_depthNear; *f = m_depthFar; }
    uint32_t CameraDeltaSeq() const { return m_cdSeq; }
    // The camera delta as the headset's appSpaceDeltaPose: the transform taking CURRENT app-space coordinates of the
    // (static) world to its PREVIOUS ones. With A = arcadeToWorld (rotation R_A = [right | up | -normal], scale s,
    // origin c): q_prev = R_A D_R R_A^T (q - c) + c + s R_A D_T. False when no reliable delta.
    bool AppSpaceDelta(XrPosef* pose, uint32_t* agree, uint32_t* pairs) const {
        *agree = m_cdAgree; *pairs = m_cdPairs;
        if (!m_cdValid || !m_a2wValid) return false;
        const float RA[9] = {m_a2wRight.x, m_a2wRight.y, m_a2wRight.z, m_a2wUp.x, m_a2wUp.y, m_a2wUp.z,
                             -m_a2wNormal.x, -m_a2wNormal.y, -m_a2wNormal.z};
        const float RAt[9] = {RA[0], RA[3], RA[6], RA[1], RA[4], RA[7], RA[2], RA[5], RA[8]};
        float tmp[9], RF[9];
        Mul3(RA, m_cdR, tmp);
        Mul3(tmp, RAt, RF);
        const float c[3] = {m_a2wCam.x, m_a2wCam.y, m_a2wCam.z};
        float pos[3];
        for (int rr = 0; rr < 3; ++rr) {
            const float rc = RF[rr] * c[0] + RF[3 + rr] * c[1] + RF[6 + rr] * c[2];
            const float at = RA[rr] * m_cdT[0] + RA[3 + rr] * m_cdT[1] + RA[6 + rr] * m_cdT[2];
            pos[rr] = c[rr] - rc + m_a2wScale * at;
        }
        // rotation matrix (column-major) -> quaternion
        const float m00 = RF[0], m11 = RF[4], m22 = RF[8], tr = m00 + m11 + m22;
        float qw, qx, qy, qz;
        if (tr > 0.0f) {
            const float S = std::sqrt(tr + 1.0f) * 2.0f;
            qw = 0.25f * S; qx = (RF[5] - RF[7]) / S; qy = (RF[6] - RF[2]) / S; qz = (RF[1] - RF[3]) / S;
        } else if (m00 > m11 && m00 > m22) {
            const float S = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
            qw = (RF[5] - RF[7]) / S; qx = 0.25f * S; qy = (RF[3] + RF[1]) / S; qz = (RF[6] + RF[2]) / S;
        } else if (m11 > m22) {
            const float S = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
            qw = (RF[6] - RF[2]) / S; qx = (RF[3] + RF[1]) / S; qy = 0.25f * S; qz = (RF[7] + RF[5]) / S;
        } else {
            const float S = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
            qw = (RF[1] - RF[3]) / S; qx = (RF[6] + RF[2]) / S; qy = (RF[7] + RF[5]) / S; qz = 0.25f * S;
        }
        const float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (!(qn > 0.5f)) return false;
        pose->orientation = {qx / qn, qy / qn, qz / qn, qw / qn};
        pose->position = {pos[0], pos[1], pos[2]};
        return true;
    }
  private:
    std::unordered_map<uint32_t, ObjMat> m_prevMats, m_curMats;
    std::vector<uint32_t> m_rawPrimSrc;
    uint32_t m_matCuts = 0;
    void MatchMotionMatrices(const tcvr_m2_frame& frame, size_t nv) {
        m_prevPosCpu.assign(nv * 4, 0.0f);
        m_curMats.clear();
        // The VIEW is part of an object's identity (26/09): the car-select screen draws a car in each box, each box
        // its own view centred on its car -- the same model in two boxes sits at the same camera-space place, and
        // pairing across views made the cars flash. View = window + projection centre + clip rectangle.
        auto viewKey = [](const tcvr_m2_prim& p) {
            uint32_t h = 2166136261u;
            for (uint32_t w : {p.window, uint32_t(p.center_x), uint32_t(p.center_y), uint32_t(p.clip_l), uint32_t(p.clip_t),
                               uint32_t(p.clip_r), uint32_t(p.clip_b)}) { h ^= w; h *= 16777619u; }
            return h;
        };
        auto objKey = [&viewKey](const tcvr_m2_prim& p) {
            uint32_t h = viewKey(p);
            for (uint32_t w : {p.motion_addr, p.motion_serial}) { h ^= w; h *= 16777619u; }
            return h;
        };
        if (!frame.raw_motion) { m_prevMats.clear(); m_prevCopies.clear(); return; }
        // Copies of one model (the crowd: the same spectator many times) are numbered in drawing order, and that
        // order changes when the game drops one out of view: "copy 5" was another spectator one frame earlier, which
        // flew across the road (Guillaume, 26/09). Copies are paired with the NEAREST copy of the previous frame
        // (object origin in camera space), within a third of its distance; unpaired copies are not blended.
        std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, ObjMat>>> groups;   // (addr, window) -> copies
        for (size_t k = 0; k < m_rawPrims.size() && k < m_rawPrimSrc.size(); ++k) {
            const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
            if (mo[14] < 0.5f) continue;
            const tcvr_m2_prim& rp = m_rawPrims[k];
            ObjMat om; for (int i = 0; i < 14; ++i) om.m[i] = mo[i];
            if (m_curMats.emplace(objKey(rp), om).second)
                groups[(uint64_t(rp.motion_addr) << 32) | viewKey(rp)].push_back({objKey(rp), om});
        }
        // pair copies -> m_pairedPrev: current object key -> previous matrix
        std::unordered_map<uint32_t, ObjMat> pairedPrev;
        for (auto& g : groups) {
            auto pg = m_prevCopies.find(g.first);
            if (pg == m_prevCopies.end()) continue;
            std::vector<ObjMat> prevList = pg->second;
            std::vector<char> used(prevList.size(), 0);
            struct Cand { float d; uint32_t ci, pi; };
            std::vector<Cand> cands;
            for (uint32_t ci = 0; ci < g.second.size(); ++ci)
                for (uint32_t pi = 0; pi < prevList.size(); ++pi) {
                    const float* a = g.second[ci].second.m; const float* b = prevList[pi].m;
                    const float dx = a[9] - b[9], dy = a[10] - b[10], dz = a[11] - b[11];
                    const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const float r = std::sqrt(a[9] * a[9] + a[10] * a[10] + a[11] * a[11]);
                    if (d <= 0.33f * std::max(r, 1.0f)) cands.push_back({d, ci, pi});
                }
            std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.d < y.d; });
            // No guessing (26/09): a pair is kept only if nothing else is nearly as close, on EITHER side. Gravel grains
            // of one model a few centimetres apart were paired with a neighbour's grain: long vertical streaks. An
            // ambiguous copy is left unpaired and borrows the motion of a sure neighbour (below).
            std::vector<float> c1(g.second.size(), 1e30f), c2(g.second.size(), 1e30f), p1(prevList.size(), 1e30f), p2(prevList.size(), 1e30f);
            for (const Cand& c : cands) {
                if (c.d < c1[c.ci]) { c2[c.ci] = c1[c.ci]; c1[c.ci] = c.d; } else if (c.d < c2[c.ci]) c2[c.ci] = c.d;
                if (c.d < p1[c.pi]) { p2[c.pi] = p1[c.pi]; p1[c.pi] = c.d; } else if (c.d < p2[c.pi]) p2[c.pi] = c.d;
            }
            std::vector<char> cdone(g.second.size(), 0);
            for (const Cand& c : cands) {
                if (cdone[c.ci] || used[c.pi]) continue;
                if (c.d * 2.0f > c2[c.ci] || c.d * 2.0f > p2[c.pi]) continue;
                cdone[c.ci] = 1; used[c.pi] = 1;
                pairedPrev.emplace(g.second[c.ci].first, prevList[c.pi]);
            }
        }
        // Everything on screen is drawn at the SAME instant (26/09): an object with no partner (a wheel -- the game
        // swaps wheel models to spin them) left at its current position while its car, blended, sat up to a frame
        // earlier: wheels off the body. An unpaired object borrows the frame-to-frame motion of the nearest paired
        // object of its view (object origins in camera space): the wheel follows its car, a lone prop the scenery.
        // Motion of a paired object n, as a map on camera space: prev = A_pn * A_cn^-1 * (x - b_cn) + b_pn.
        std::unordered_map<uint32_t, std::pair<ObjMat, ObjMat>> borrowed;   // unpaired key -> (neighbour cur, neighbour prev)
        std::unordered_map<uint32_t, std::vector<uint32_t>> pairedByView;   // view key -> paired object keys
        {
            std::unordered_map<uint32_t, uint32_t> viewOf;
            for (size_t k = 0; k < m_rawPrims.size() && k < m_rawPrimSrc.size(); ++k) {
                if (frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16 + 14] < 0.5f) continue;   // no object matrix
                const uint32_t ok = objKey(m_rawPrims[k]);
                if (viewOf.count(ok)) continue;
                viewOf[ok] = viewKey(m_rawPrims[k]);
                if (pairedPrev.count(ok) && m_curMats.count(ok)) pairedByView[viewOf[ok]].push_back(ok);
            }
            for (const auto& kv : m_curMats) {
                if (pairedPrev.count(kv.first)) continue;
                auto pv = pairedByView.find(viewOf[kv.first]);
                if (pv == pairedByView.end()) continue;
                const float* a = kv.second.m;
                float bestD = 1e30f; uint32_t best = 0;
                for (uint32_t nk : pv->second) {
                    const float* b = m_curMats[nk].m;
                    const float dx = a[9] - b[9], dy = a[10] - b[10], dz = a[11] - b[11];
                    const float d = dx * dx + dy * dy + dz * dz;
                    if (d < bestD) { bestD = d; best = nk; }
                }
                if (bestD < 1e29f) borrowed.emplace(kv.first, std::make_pair(m_curMats[best], pairedPrev[best]));
            }
        }
        std::unordered_map<uint64_t, std::vector<ObjMat>> copiesNow;
        for (auto& g : groups) { auto& v = copiesNow[g.first]; for (auto& c : g.second) v.push_back(c.second); }
        const bool blendable = m_haveMainView && !m_isMenuM1 && !pairedPrev.empty();
        // Camera cut: most objects' origins jump by more than a third of their distance -> no blend this step.
        size_t objs = 0, jumped = 0;
        if (blendable)
            for (const auto& kv : m_curMats) {
                auto it = pairedPrev.find(kv.first);
                if (it == pairedPrev.end()) continue;
                const float* a = kv.second.m; const float* b = it->second.m;
                const float dx = a[9] - b[9], dy = a[10] - b[10], dz = a[11] - b[11];
                const float d = std::sqrt(a[9] * a[9] + a[10] * a[10] + a[11] * a[11]);
                ++objs;
                if (std::sqrt(dx * dx + dy * dy + dz * dz) > 0.33f * std::max(d, 1.0f)) ++jumped;
            }
        const bool cut = objs > 4 && jumped * 2 > objs;
        if (cut) ++m_matCuts;
        // SAME INSTANT FOR EVERYTHING (26/09): a polygon left at its current position while the rest is drawn up to a
        // frame earlier is a picture that exists in neither of the game's frames -- the flashes of the wheels and of the
        // gravel. Every polygon of a view with a paired object therefore gets a motion: its own pair, else the nearest
        // sure neighbour's. Polygons with no object matrix (direct polygons: particles, gravel; Top Skater's skinned
        // meshes) pick the neighbour nearest to their centroid in camera space.
        std::vector<std::pair<const float*, const float*>> primSrc(m_rawPrims.size(), {nullptr, nullptr});
        if (blendable && !cut) {
            for (size_t k = 0; k < m_rawPrims.size() && k < m_rawPrimSrc.size(); ++k) {
                const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
                const tcvr_m2_prim& rp = m_rawPrims[k];
                if (mo[14] >= 0.5f) {
                    const uint32_t okey = objKey(rp);
                    auto it = pairedPrev.find(okey);
                    if (it != pairedPrev.end()) { primSrc[k] = {mo, it->second.m}; continue; }
                    auto bo = borrowed.find(okey);
                    if (bo != borrowed.end()) primSrc[k] = {bo->second.first.m, bo->second.second.m};
                    continue;
                }
                auto pv = pairedByView.find(viewKey(rp));
                if (pv == pairedByView.end() || rp.vertex_count == 0u || std::fabs(mo[12]) < 1e-6f || std::fabs(mo[13]) < 1e-6f) continue;
                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                for (uint32_t vi = 0; vi < rp.vertex_count; ++vi) {
                    const float* d = &m_rawVerts[size_t(rp.first_vertex + vi) * 5];
                    cx += d[0] / mo[12]; cy += d[1] / mo[13]; cz += d[2];
                }
                cx /= float(rp.vertex_count); cy /= float(rp.vertex_count); cz /= float(rp.vertex_count);
                float bestD = 1e30f; uint32_t best = 0;
                for (uint32_t nk : pv->second) {
                    const float* b = m_curMats[nk].m;
                    const float d = (cx - b[9]) * (cx - b[9]) + (cy - b[10]) * (cy - b[10]) + (cz - b[11]) * (cz - b[11]);
                    if (d < bestD) { bestD = d; best = nk; }
                }
                if (bestD < 1e29f) primSrc[k] = {m_curMats[best].m, pairedPrev[best].m};
            }
            for (size_t v = 0; v < nv; ++v) {
                const uint32_t k = m_rawPrimOfVertex[v];
                if (k >= m_rawPrims.size() || k >= m_rawPrimSrc.size()) continue;
                const float* mo = &frame.raw_motion[size_t(m_rawPrimSrc[k]) * 16];
                const float* cm = primSrc[k].first;    // the matrix whose inverse brings the vertex to "object" space
                const float* pm = primSrc[k].second;   // ... and the one that brings it to the previous frame
                if (!cm || !pm) continue;
                // unfocus (this object's focus), then the neighbour-or-own object space: A^-1 (o - b)
                const float* d = &m_rawVerts[v * 5];
                if (std::fabs(mo[12]) < 1e-6f || std::fabs(mo[13]) < 1e-6f) continue;
                const float o[3] = {d[0] / mo[12] - cm[9], d[1] / mo[13] - cm[10], d[2] - cm[11]};
                const float a = cm[0], b = cm[3], c = cm[6], e = cm[1], f = cm[4], g = cm[7], h = cm[2], i2 = cm[5], j = cm[8];
                const float det = a * (f * j - g * i2) - b * (e * j - g * h) + c * (e * i2 - f * h);
                if (std::fabs(det) < 1e-9f) continue;
                const float id = 1.0f / det;
                const float ob[3] = {
                    ((f * j - g * i2) * o[0] + (c * i2 - b * j) * o[1] + (b * g - c * f) * o[2]) * id,
                    ((g * h - e * j) * o[0] + (a * j - c * h) * o[1] + (c * e - a * g) * o[2]) * id,
                    ((e * i2 - f * h) * o[0] + (b * h - a * i2) * o[1] + (a * f - b * e) * o[2]) * id};
                const float px = ob[0] * pm[0] + ob[1] * pm[3] + ob[2] * pm[6] + pm[9];
                const float py = ob[0] * pm[1] + ob[1] * pm[4] + ob[2] * pm[7] + pm[10];
                const float pz = ob[0] * pm[2] + ob[1] * pm[5] + ob[2] * pm[8] + pm[11];
                float* o4 = &m_prevPosCpu[v * 4];
                o4[0] = px * pm[12]; o4[1] = py * pm[13]; o4[2] = pz; o4[3] = 1.0f;
            }
        }
        if (arcadexr::config::GetInt("m2.motionDiag", 0) != 0 && (++m_motionDiagTick % 60u) == 0u) {
            size_t blended = 0;
            for (size_t v = 0; v < nv; ++v) blended += m_prevPosCpu[v * 4 + 3] > 0.5f ? 1u : 0u;
            __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION", "matrices: objects=%zu with previous=%zu jumped=%zu cuts=%u | vertices %zu/%zu blended | main=%d",
                                m_curMats.size(), objs, jumped, m_matCuts, blended, nv, int(m_haveMainView));
        }
        m_prevMats.swap(m_curMats);
        m_prevCopies.swap(copiesNow);
    }
    std::unordered_map<uint64_t, std::vector<ObjMat>> m_prevCopies;

    void MatchMotion(size_t nv) {
        m_curQuads.clear();
        for (size_t v = 0; v < nv;) {
            const uint32_t k = m_rawPrimOfVertex[v];
            size_t e = v + 1;
            while (e < nv && m_rawPrimOfVertex[e] == k) ++e;
            MotionQuad q{};
            if (m_directColour) {
                q.key = uint32_t(m_rawVerts[v * 5 + 3]);
                q.obj = uint32_t(m_rawVerts[v * 5 + 4]) >> 2;   // which push_object call of the frame
            } else {
                // Model 2: identity from the prim -- object address, rank in the object, copy of the object in the frame.
                const tcvr_m2_prim& mp = m_rawPrims[k];
                uint32_t h = 2166136261u;
                for (uint32_t w : {mp.motion_addr, mp.motion_poly, mp.motion_serial}) { h ^= w; h *= 16777619u; }
                q.key = h;
                uint32_t o = 2166136261u;
                for (uint32_t w : {mp.motion_addr, mp.motion_serial}) { o ^= w; o *= 16777619u; }
                q.obj = o;
            }
            q.v0 = uint32_t(v); q.n = uint32_t(e - v);
            for (size_t i = v; i < e; ++i) {
                const float* d = &m_rawVerts[i * 5];
                const uint32_t corner = m_directColour ? (uint32_t(d[4]) & 3u) : uint32_t(std::min<size_t>(i - v, 3));
                for (int c = 0; c < 3; ++c) { q.p[corner][c] = d[c]; q.c[c] += d[c] / float(e - v); }
            }
            m_curQuads.push_back(q);
            v = e;
        }
        std::sort(m_curQuads.begin(), m_curQuads.end(), [](const MotionQuad& x, const MotionQuad& y) { return x.key < y.key; });

        // Keys present exactly once in both frames.
        m_motionPairs.clear();
        for (size_t i = 0, j = 0; i < m_curQuads.size() && j < m_lastQuads.size();) {
            const uint32_t ki = m_curQuads[i].key, kj = m_lastQuads[j].key;
            if (ki < kj) { ++i; continue; }
            if (kj < ki) { ++j; continue; }
            size_t ie = i, je = j;
            while (ie < m_curQuads.size() && m_curQuads[ie].key == ki) ++ie;
            while (je < m_lastQuads.size() && m_lastQuads[je].key == ki) ++je;
            if (ie - i == 1 && je - j == 1) m_motionPairs.push_back({&m_lastQuads[j], &m_curQuads[i]});
            i = ie; j = je;
        }

        // Camera motion prev -> cur. The raw positions are (zoom*x + view*z, zoom*y + view*z, z): a rigid move of the
        // game camera is an AFFINE map in that space, fitted exactly by least squares (one 4x4 system per output
        // coordinate), trimmed at 3x the median residual.
        float A3[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        auto apply = [&](const float* p, float* o) {
            for (int r = 0; r < 3; ++r) o[r] = A3[r * 4] * p[0] + A3[r * 4 + 1] * p[1] + A3[r * 4 + 2] * p[2] + A3[r * 4 + 3];
        };
        float trim = 1e30f, medRes = -1.0f;
        std::vector<float> res;
        for (int it = 0; it < 4 && m_motionPairs.size() >= 8; ++it) {
            double N[4][4] = {}, B[4][3] = {};
            res.clear();
            int used = 0;
            for (const auto& pr : m_motionPairs) {
                float p[3]; apply(pr.first->c, p);
                const float* q = pr.second->c;
                const float e = std::sqrt((q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1]) + (q[2] - p[2]) * (q[2] - p[2]));
                res.push_back(e);
                if (e > trim) continue;
                const double x[4] = {pr.first->c[0], pr.first->c[1], pr.first->c[2], 1.0};
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) N[r][c] += x[r] * x[c];
                    for (int c = 0; c < 3; ++c) B[r][c] += x[r] * q[c];
                }
                ++used;
            }
            std::vector<float> sr = res;
            std::nth_element(sr.begin(), sr.begin() + sr.size() / 2, sr.end());
            medRes = sr[sr.size() / 2];
            trim = std::max(3.0f * medRes, 1e-3f);
            if (used < 8) break;
            bool ok = true;   // Gauss-Jordan with partial pivoting, 3 right-hand sides
            for (int c = 0; c < 4 && ok; ++c) {
                int piv = c;
                for (int r = c + 1; r < 4; ++r) if (std::fabs(N[r][c]) > std::fabs(N[piv][c])) piv = r;
                if (std::fabs(N[piv][c]) < 1e-9) { ok = false; break; }
                for (int k = 0; k < 4; ++k) std::swap(N[c][k], N[piv][k]);
                for (int k = 0; k < 3; ++k) std::swap(B[c][k], B[piv][k]);
                for (int r = 0; r < 4; ++r) {
                    if (r == c) continue;
                    const double f = N[r][c] / N[c][c];
                    for (int k = 0; k < 4; ++k) N[r][k] -= f * N[c][k];
                    for (int k = 0; k < 3; ++k) B[r][k] -= f * B[c][k];
                }
            }
            if (!ok) break;
            for (int o = 0; o < 3; ++o)
                for (int c = 0; c < 4; ++c) A3[o * 4 + c] = float(B[c][o] / N[c][c]);
        }
        const float T[3] = {A3[3], A3[7], A3[11]};
        // Inverse map: where a static piece of the world was in the previous frame (fallback for unmatched quads).
        const bool fitOk = medRes >= 0.0f && m_motionPairs.size() >= 8;
        float Ai[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        if (fitOk) {
            const float a = A3[0], b = A3[1], c = A3[2], d = A3[4], e = A3[5], f = A3[6], g = A3[8], h = A3[9], k = A3[10];
            const float det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
            if (std::fabs(det) > 1e-6f) {
                const float inv[9] = {(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det,
                                      (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det,
                                      (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
                for (int r = 0; r < 3; ++r) {
                    for (int cc = 0; cc < 3; ++cc) Ai[r * 4 + cc] = inv[r * 3 + cc];
                    Ai[r * 4 + 3] = -(inv[r * 3] * A3[3] + inv[r * 3 + 1] * A3[7] + inv[r * 3 + 2] * A3[11]);
                }
            }
        }

        // Match whole OBJECTS, then their quads. Per-quad matching let a wheel of car A pair with the wheel of car B
        // on the starting grid (same model): ghost cars stretched between two places (Guillaume, 24/09). Each object
        // scores every previous object sharing a quad key by the mean distance of those quads once moved; the
        // pairs are taken greedily, nearest first, one previous object for one current object.
        m_prevPosCpu.assign(nv * 4, 0.0f);
        size_t matchedQuads = 0, staticQuads = 0;
        std::vector<float> rel, fitErr;
        const bool diag = arcadexr::config::GetInt("m2.motionDiag", 0) != 0;
        // Distance in arcade screen pixels (raw x, y carry the zoom: x/z is the screen offset) plus the relative depth
        // gap, weighted so 1 % of depth counts as one pixel. Moved with the world, or fixed to the screen (HUD).
        auto dist1 = [](const float* p, const float* c) {
            const float pz = std::max(std::fabs(p[2]), 1e-3f), cz = std::max(std::fabs(c[2]), 1e-3f);
            const float sx = p[0] / pz - c[0] / cz, sy = p[1] / pz - c[1] / cz;
            return std::sqrt(sx * sx + sy * sy) + 100.0f * std::fabs(p[2] - c[2]) / cz;
        };
        auto dist = [&](const MotionQuad& prev, const MotionQuad& cur) {
            float p[3]; apply(prev.c, p);
            return std::min(dist1(p, cur.c), dist1(prev.c, cur.c));
        };
        auto indexByObj = [](const std::vector<MotionQuad>& qs, std::vector<uint32_t>& order, std::vector<std::pair<uint32_t, uint32_t>>& ranges) {
            order.resize(qs.size());
            for (uint32_t i = 0; i < qs.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return qs[x].obj < qs[y].obj; });
            ranges.clear();
            for (uint32_t i = 0; i < order.size();) {
                uint32_t e = i + 1;
                while (e < order.size() && qs[order[e]].obj == qs[order[i]].obj) ++e;
                ranges.push_back({i, e});
                i = e;
            }
        };
        std::vector<uint32_t> curOrder, lastOrder;
        std::vector<std::pair<uint32_t, uint32_t>> curObjs, lastObjs;
        indexByObj(m_curQuads, curOrder, curObjs);
        indexByObj(m_lastQuads, lastOrder, lastObjs);
        std::vector<uint32_t> lastObjOfQuad(m_lastQuads.size());
        for (uint32_t o = 0; o < lastObjs.size(); ++o)
            for (uint32_t i = lastObjs[o].first; i < lastObjs[o].second; ++i) lastObjOfQuad[lastOrder[i]] = o;
        auto keyRange = [&](uint32_t key) {
            auto lo = std::lower_bound(m_lastQuads.begin(), m_lastQuads.end(), key, [](const MotionQuad& q, uint32_t k) { return q.key < k; });
            auto hi = lo;
            while (hi != m_lastQuads.end() && hi->key == key) ++hi;
            return std::make_pair(size_t(lo - m_lastQuads.begin()), size_t(hi - m_lastQuads.begin()));
        };
        struct Cand { float score; uint32_t cur, last; };
        std::vector<Cand> cands;
        std::vector<std::pair<uint32_t, std::pair<float, uint32_t>>> acc;   // last object -> (sum, count)
        for (uint32_t o = 0; o < curObjs.size(); ++o) {
            acc.clear();
            const uint32_t n = curObjs[o].second - curObjs[o].first;
            const uint32_t step = std::max(1u, n / 6u);   // a few quads spread over the object are enough to score it
            for (uint32_t i = curObjs[o].first; i < curObjs[o].second; i += step) {
                const MotionQuad& cq = m_curQuads[curOrder[i]];
                const auto r = keyRange(cq.key);
                for (size_t b2 = r.first; b2 < r.second; ++b2) {
                    const uint32_t lo = lastObjOfQuad[b2];
                    const float d = dist(m_lastQuads[b2], cq);
                    auto it = std::find_if(acc.begin(), acc.end(), [&](const auto& x) { return x.first == lo; });
                    if (it == acc.end()) acc.push_back({lo, {d, 1u}});
                    else { it->second.first += d; it->second.second += 1u; }
                }
            }
            for (const auto& x : acc) {
                const float sc = x.second.first / float(x.second.second);
                if (sc < 16.0f) cands.push_back({sc, o, x.first});
            }
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.score < y.score; });
        std::vector<int32_t> curTaken(curObjs.size(), -1);
        std::vector<char> lastTaken(lastObjs.size(), 0);
        for (const Cand& c : cands) {
            if (curTaken[c.cur] >= 0 || lastTaken[c.last]) continue;
            curTaken[c.cur] = int32_t(c.last); lastTaken[c.last] = 1;
        }
        // Inside a matched object, quads that do not follow the object's own motion (a spinning wheel, whose facets
        // jump by a large angle every step; particles) would twist when blended in a straight line: they take the
        // object's MEAN motion instead (Guillaume, 24/09: "effet bizarre sur nos roues"). The motion is measured after
        // removing the camera's, in arcade screen pixels, so static scenery never trips it.
        auto scr = [](const float* p, float* o2) { const float z = std::max(std::fabs(p[2]), 1e-3f); o2[0] = p[0] / z; o2[1] = p[1] / z; };
        std::vector<const MotionQuad*> bqs;
        std::vector<float> own;   // per quad: own motion on screen (x, y)
        size_t rigidQuads = 0;
        // Pass 1: each matched object's own mean motion (raw space, camera removed) and every object's centroid.
        // An UNMATCHED object borrows the motion of the nearest matched one: the game swaps wheel models every frame
        // to spin them, so a wheel never finds itself and, taken as static scenery, slid away from its car.
        std::vector<float> objOwn(curObjs.size() * 3, 0.0f), objC(curObjs.size() * 3, 0.0f);
        std::vector<char> objHasOwn(curObjs.size(), 0);
        for (uint32_t o = 0; o < curObjs.size(); ++o) {
            const uint32_t n = curObjs[o].second - curObjs[o].first;
            for (uint32_t i = 0; i < n; ++i)
                for (int c = 0; c < 3; ++c) objC[o * 3 + c] += m_curQuads[curOrder[curObjs[o].first + i]].c[c] / float(n);
            if (curTaken[o] < 0) continue;
            const auto& lr = lastObjs[uint32_t(curTaken[o])];
            uint32_t nm = 0;
            for (uint32_t i = 0; i < n; ++i) {
                const MotionQuad& cq = m_curQuads[curOrder[curObjs[o].first + i]];
                for (uint32_t k = lr.first; k < lr.second; ++k)
                    if (m_lastQuads[lastOrder[k]].key == cq.key) {
                        float moved[3]; apply(m_lastQuads[lastOrder[k]].c, moved);
                        for (int c = 0; c < 3; ++c) objOwn[o * 3 + c] += cq.c[c] - moved[c];
                        ++nm;
                        break;
                    }
            }
            if (nm) { for (int c = 0; c < 3; ++c) objOwn[o * 3 + c] /= float(nm); objHasOwn[o] = 1; }
        }
        size_t borrowed = 0;
        for (uint32_t o = 0; o < curObjs.size(); ++o) {
            const uint32_t n = curObjs[o].second - curObjs[o].first;
            bqs.assign(n, nullptr);
            own.assign(n * 2, 0.0f);
            float meanOwn[2] = {0, 0}, meanRaw[3] = {0, 0, 0};
            uint32_t nm = 0;
            if (curTaken[o] >= 0) {
                const auto& lr = lastObjs[uint32_t(curTaken[o])];
                for (uint32_t i = 0; i < n; ++i) {
                    const MotionQuad& cq = m_curQuads[curOrder[curObjs[o].first + i]];
                    for (uint32_t k = lr.first; k < lr.second; ++k)
                        if (m_lastQuads[lastOrder[k]].key == cq.key) { bqs[i] = &m_lastQuads[lastOrder[k]]; break; }
                    if (!bqs[i]) continue;
                    float moved[3]; apply(bqs[i]->c, moved);
                    float a2[2], b2[2]; scr(cq.c, a2); scr(moved, b2);
                    own[i * 2] = a2[0] - b2[0]; own[i * 2 + 1] = a2[1] - b2[1];
                    meanOwn[0] += own[i * 2]; meanOwn[1] += own[i * 2 + 1];
                    for (int c = 0; c < 3; ++c) meanRaw[c] += cq.c[c] - moved[c];
                    ++nm;
                }
            }
            if (nm) { meanOwn[0] /= nm; meanOwn[1] /= nm; for (int c = 0; c < 3; ++c) meanRaw[c] /= nm; }
            bool borrow = false;
            if (!nm) {   // nearest matched object, on screen and in depth, within 48 (pixels + depth %)
                float bestD = 48.0f; int32_t bo = -1;
                for (uint32_t o2 = 0; o2 < curObjs.size(); ++o2) {
                    if (!objHasOwn[o2]) continue;
                    const float d = dist1(&objC[o2 * 3], &objC[o * 3]);
                    if (d < bestD) { bestD = d; bo = int32_t(o2); }
                }
                if (bo >= 0) { for (int c = 0; c < 3; ++c) meanRaw[c] = objOwn[size_t(bo) * 3 + c]; borrow = true; ++borrowed; }
                if (diag && (m_motionDiagTick % 8u) == 7u)
                    __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION", "unmatched obj n=%u c=(%.0f %.0f %.1f) nearest=%.1f",
                                        curObjs[o].second - curObjs[o].first, objC[o * 3], objC[o * 3 + 1], objC[o * 3 + 2], bestD);
            }
            const float ownLen = std::sqrt(meanOwn[0] * meanOwn[0] + meanOwn[1] * meanOwn[1]);
            if (m_m2Smooth && fitOk) {
                // Model 2 (25/09): one motion per OBJECT, never per polygon -- and no deforming fit. Still objects
                // (the road pieces, the scenery) follow the camera motion, identical for the whole scene; moving
                // objects (cars) follow the camera plus one translation (their mean own motion), Model 1's proven
                // rigid rule. The class keeps a margin and remembers itself from frame to frame (by object identity):
                // an object flipping between classes jumped, and two neighbours in different classes opened a crack
                // where the sky showed (the thin blue line across at wheel level).
                const MotionQuad& first = m_curQuads[curOrder[curObjs[o].first]];
                const uint32_t objId = first.obj;
                bool moving = false;
                auto itc = m_objMoving.find(objId);
                const bool was = itc != m_objMoving.end() && itc->second;
                if (nm > 0) moving = was ? (ownLen > 3.0f) : (ownLen > 8.0f);
                else if (borrow) moving = true;
                if (moving != was) ++m_classFlips;
                m_objMovingNext[objId] = moving;
                for (uint32_t i = 0; i < n; ++i) {
                    const MotionQuad& cq = m_curQuads[curOrder[curObjs[o].first + i]];
                    for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) {
                        const float* d = &m_rawVerts[size_t(v) * 5];
                        const float src[3] = {d[0] - (moving ? meanRaw[0] : 0.0f), d[1] - (moving ? meanRaw[1] : 0.0f), d[2] - (moving ? meanRaw[2] : 0.0f)};
                        float* o4 = &m_prevPosCpu[size_t(v) * 4];
                        for (int r = 0; r < 3; ++r) o4[r] = Ai[r * 4] * src[0] + Ai[r * 4 + 1] * src[1] + Ai[r * 4 + 2] * src[2] + Ai[r * 4 + 3];
                        o4[3] = 1.0f;
                    }
                    if (moving) { ++rigidQuads; ++matchedQuads; } else { ++staticQuads; if (nm > 0) ++matchedQuads; }
                }
                continue;
            }
            for (uint32_t i = 0; i < n; ++i) {
                const MotionQuad& cq = m_curQuads[curOrder[curObjs[o].first + i]];
                const MotionQuad* bq = bqs[i];
                bool rigid = false;
                if (bq) {
                    const float dx = own[i * 2] - meanOwn[0], dy = own[i * 2 + 1] - meanOwn[1];
                    rigid = std::sqrt(dx * dx + dy * dy) > 2.0f + 0.25f * ownLen;
                }
                if (!bq && (borrow || nm)) rigid = true;   // a facet with no partner follows its object
                if (bq && !rigid) {
                    ++matchedQuads;
                    for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) {
                        // Model 1 carries the corner in v; a Model 2 quad's corners are its vertices in order (its v is
                        // a texture coordinate -- read as a corner it smeared polygons into streaks, 25/09).
                        const uint32_t corner = m_directColour ? (uint32_t(m_rawVerts[size_t(v) * 5 + 4]) & 3u)
                                                               : std::min<uint32_t>(v - cq.v0, 3u);
                        float* o4 = &m_prevPosCpu[size_t(v) * 4];
                        o4[0] = bq->p[corner][0]; o4[1] = bq->p[corner][1]; o4[2] = bq->p[corner][2]; o4[3] = 1.0f;
                    }
                    if (diag) { rel.push_back(dist1(bq->c, cq.c)); fitErr.push_back(dist(*bq, cq)); }
                } else if (fitOk) {
                    // rigid: camera motion + the object's mean own motion; unmatched: static in the world
                    if (rigid) { ++rigidQuads; ++matchedQuads; } else ++staticQuads;
                    for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) {
                        const float* d = &m_rawVerts[size_t(v) * 5];
                        const float src[3] = {d[0] - (rigid ? meanRaw[0] : 0.0f), d[1] - (rigid ? meanRaw[1] : 0.0f), d[2] - (rigid ? meanRaw[2] : 0.0f)};
                        float* o4 = &m_prevPosCpu[size_t(v) * 4];
                        for (int r = 0; r < 3; ++r) o4[r] = Ai[r * 4] * src[0] + Ai[r * 4 + 1] * src[1] + Ai[r * 4 + 2] * src[2] + Ai[r * 4 + 3];
                        o4[3] = 1.0f;
                    }
                }
            }
        }
        // Whole quads only: a quad blended at some corners and not at others tears into slivers (the grass under
        // the car in green shards, 24/09 -- a per-vertex guard did that). Crossing z = 0 is fine: the GPU clips in
        // homogeneous space. A matched quad whose own motion (camera removed) exceeds 64 screen pixels is a bad
        // match: it follows the camera instead, as static scenery.
        if (m_m2Smooth) {
            m_objMoving.swap(m_objMovingNext); m_objMovingNext.clear();
            if (diag && (m_motionDiagTick % 8u) == 7u)
                __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION", "class flips since last log=%u objects=%zu", m_classFlips, m_objMoving.size());
            if (diag && (m_motionDiagTick % 8u) == 7u) m_classFlips = 0;
        }
        size_t guarded = 0;
        for (const MotionQuad& cq : m_curQuads) {
            bool all = true;
            for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) all = all && m_prevPosCpu[size_t(v) * 4 + 3] > 0.5f;
            if (!all) { for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) m_prevPosCpu[size_t(v) * 4 + 3] = 0.0f; continue; }
            if (!fitOk) continue;
            float pc[3] = {0, 0, 0}, sc[3];
            for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v)
                for (int c = 0; c < 3; ++c) pc[c] += m_prevPosCpu[size_t(v) * 4 + c] / float(cq.n);
            for (int r = 0; r < 3; ++r) sc[r] = Ai[r * 4] * cq.c[0] + Ai[r * 4 + 1] * cq.c[1] + Ai[r * 4 + 2] * cq.c[2] + Ai[r * 4 + 3];
            if (pc[2] <= 1.0f || sc[2] <= 1.0f) continue;
            const float dx = pc[0] / pc[2] - sc[0] / sc[2], dy = pc[1] / pc[2] - sc[1] / sc[2];
            if (dx * dx + dy * dy <= 64.0f * 64.0f) continue;
            ++guarded;
            for (uint32_t v = cq.v0; v < cq.v0 + cq.n; ++v) {
                const float* d = &m_rawVerts[size_t(v) * 5];
                float* o4 = &m_prevPosCpu[size_t(v) * 4];
                for (int r = 0; r < 3; ++r) o4[r] = Ai[r * 4] * d[0] + Ai[r * 4 + 1] * d[1] + Ai[r * 4 + 2] * d[2] + Ai[r * 4 + 3];
            }
        }
        if (diag && (m_motionDiagTick % 8u) == 7u) __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION", "guarded quads=%zu", guarded);
        // A camera cut (replay angles, attract mode): almost nothing matches -> no blend for this step at all,
        // or the whole scene would sweep from the old shot to the new one.
        if (matchedQuads * 10 < m_curQuads.size() * 3) {
            for (size_t v = 0; v < nv; ++v) m_prevPosCpu[v * 4 + 3] = 0.0f;
            staticQuads = 0;
        }
        if (diag && (++m_motionDiagTick % 8u) == 0u) {
            std::sort(rel.begin(), rel.end());
            std::sort(fitErr.begin(), fitErr.end());
            if (!fitErr.empty()) __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION", "match residual p50=%.2f p90=%.2f p99=%.2f",
                fitErr[fitErr.size() / 2], fitErr[fitErr.size() * 9 / 10], fitErr[fitErr.size() * 99 / 100]);
            auto pc = [&](float f) { return rel.empty() ? -1.0f : rel[std::min(rel.size() - 1, size_t(f * rel.size()))]; };
            __android_log_print(ANDROID_LOG_INFO, "TCVR_MOTION",
                                "quads=%zu last=%zu pairs=%zu fitMedRes=%.3f T=(%.2f %.2f %.2f) matched=%zu static=%zu rigid=%zu borrowedObjs=%zu | screen motion px p50=%.4f p90=%.4f p99=%.4f max=%.4f | period=%.1fms",
                                m_curQuads.size(), m_lastQuads.size(), m_motionPairs.size(), medRes, T[0], T[1], T[2], matchedQuads, staticQuads, rigidQuads, borrowed,
                                pc(0.5f), pc(0.9f), pc(0.99f), rel.empty() ? -1.0f : rel.back(), m_stepPeriod * 1000.0f);
        }
        m_lastQuads.swap(m_curQuads);
    }
    bool m_smooth = false;
    std::chrono::steady_clock::time_point m_stepTime{};
    float m_stepPeriod = 2.0f / 57.5f;
public:
    // Smooth motion: blend of the previous and current arcade frames for this display refresh (1 = current).
    void SetSmooth(bool on) { m_smooth = on; }
    bool HasMotionIds() const { return m_directColour || m_m2Smooth; }
    std::unordered_map<uint32_t, bool> m_objMoving, m_objMovingNext;   // Model 2 smooth motion: class per object
    uint32_t m_classFlips = 0;
    bool m_m2Smooth = false;
private:
    float* m_vboMappedF[kFrames] = {};
    size_t m_vboSize = 0;

    BufferAndMemory m_primIndexBufferF[kFrames];
    uint32_t* m_primIndexMappedF[kFrames] = {};
    size_t m_primIndexSize = 0;

    BufferAndMemory m_iboBufferF[kFrames];
    uint32_t* m_iboMappedF[kFrames] = {};
    size_t m_iboSize = 0;

    // Descriptor Sets
    VkDescriptorSet m_m2DescSetF[kFrames][2][2] = {};
    VkDescriptorSet m_layerDescSet[2] = {};

    // Textures
    // [0],[1]: sheets as R8 (t*17). [2],[3]: the same sheets as RG8 for cut-outs:
    // R = t*17 premultiplied by opacity, G = opacity (texel != 15). Filtering those two in
    // hardware and dividing gives MAME's "a transparent texel borrows its neighbour" for free.
    VkImage m_sheetImage[4] = {};
    VkDeviceMemory m_sheetMem[4] = {};
    VkImageView m_sheetView[4] = {};
    VkImageLayout m_sheetLayout[4] = {};
    BufferAndMemory m_sheetStagingBuffer[4];
    uint32_t* m_sheetStagingMapped[4] = {nullptr, nullptr, nullptr, nullptr};
    std::vector<uint8_t> m_sheetCpu[2];
    M2RegionTextures m_regions;
    std::vector<uint8_t> m_lutSrcCx, m_lutSrcGamma;
    bool m_lutValid = false;
    bool m_useRegions = false;
    std::vector<uint32_t> m_primSlot;
    std::vector<uint32_t> m_primLayer;                       // unpacked t per texel, for the region scan
    std::unordered_map<uint64_t, bool> m_regionHasHoles;      // region key -> contains texel 15
    uint64_t m_sheetGeneration = 0;
    uint64_t m_texHash = 0;
    std::vector<uint32_t> m_texShadow[2];   // texture RAM as last uploaded (partial updates)
    bool m_regionsStale = false;
    int m_lastTexReupload = 0;            // a full upload happened: regions to rebuild on the next built frame
    uint32_t m_partialTexLog = 0;
    bool m_texturesUploaded = false;

    VkImage m_layerImage[2] = {};
    VkDeviceMemory m_layerMem[2] = {};
    VkImageView m_layerView[2] = {};
    VkImageLayout m_layerLayout[2] = {};
    BufferAndMemory m_layerStagingBufferF[kFrames][2];
    uint32_t* m_layerStagingMappedF[kFrames][2] = {};
    bool m_haveLayer[2] = {false, false};
    bool m_frontFullscreen = false;
    bool m_backNoTile = false;
    bool m_directColour = false;
    bool m_newGeometry = false;
    float m_zMaxSmooth = 0.0f;
    float m_fogColour[3] = {0.6f, 0.75f, 0.9f};
    unsigned m_pitchLogTick = 0;
    int m_menuFrames = 0;
    bool m_isMenuM1 = false;
    float m_lastInterp = 1.0f;
  public:
    float LastInterp() const { return m_lastInterp; }
  private:
    static bool HudIsoProfile() {
        return arcadexr::profiles::GetInt("immersive.hudIso", arcadexr::profiles::CurrentGame() == "srallyc" ? 1 : 0) != 0;
    }
    // A menu in ISO (26/09): its 2D page on the game camera's rays at the depth of its 3D, and its 3D (the cars in their
    // boxes) left in 3D at its true depth -- on the page's plane within 0.05 degree of disparity, with its depth test.
    // Flattened, the cars were painted face by face with no depth test: 17 ms of GPU in the car select.
    bool MenuIso() const { return m_isMenuM1 && m_mainZMax > 0.0f && HudIsoProfile(); }
    // Otherwise a menu is drawn flat on the screen plane: the vertex shader (uMenuFlat) and the pass routing (isMain)
    // both read THIS.
    bool MenuFlat() const { return m_isMenuM1 && !MenuIso(); }
    float m_lastPitchTarget = 0.0f;
    float m_layerUvScaleX[2] = {496.0f / 512.0f, 496.0f / 512.0f};

    // Geometry unpack state
    std::vector<std::uint64_t> m_rawKeys;
    bool m_primDiagOn = false; uint32_t m_primDiagTick = 0; float m_primDiagX = 0, m_primDiagY = 0;
    uint32_t m_clN = 0, m_clNotMain = 0, m_clBk[2] = {}, m_clBand[24] = {}, m_clPass[5] = {}; float m_clBox[4] = {};
    std::vector<tcvr_m2_prim> m_rawPrims;
    std::vector<float> m_rawVerts;
    std::vector<std::uint32_t> m_rawPrimOfVertex;
    std::vector<std::uint32_t> m_rawIdx;

    uint64_t m_preparedSeq = 0;
    uint32_t m_opaqueIndexCount = 0;
    uint32_t m_glassIndexCount = 0;
    int32_t m_mainClip[4] = {-1, -1, -1, -1};
    int32_t m_mainCenter[2] = {-100000, -100000};
    bool m_haveMainView = false;
    float m_horizonGeo = -1.0f;
    bool m_haveHorizon = false;
    float m_mainZMax = 0.0f;   // deepest main-view vertex (menu or scene?)
    float m_crtc[2] = {0.0f, 0.0f};
    struct AimTransform { bool valid; arcadexr::gun::Vec3 cam, R, U, N; float s, nearM; };
    AimTransform m_aimXf{false, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 1.0f, 0.25f};
    unsigned m_aimTestTick = 0;
    float m_aimFocus[2] = {512.0f, 512.0f};
    struct AimHud { bool valid; arcadexr::gun::Vec3 C, R, U, N; float w, h; };
    AimHud m_aimHud{false, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 1.0f, 1.0f};
    mutable bool m_aimOnPlane = false;
public:
    bool AimOnPlane() const { return m_aimOnPlane; }
private:
    bool m_flatMode = false;
    XrMatrix4x4f m_flatMvp{};
    struct FlatTarget { VkImage image = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
                        uint32_t w = 0, h = 0, levels = 0; } m_flat;
    float m_groundColor[3] = {0.18f, 0.16f, 0.14f};
    float m_voidColor[3] = {0.16f, 0.36f, 0.78f};
    float m_m2Pitch = 0.0f;
    float m_m2FocusX = 512.0f;
    float m_m2FocusY = 512.0f;
};

}  // namespace arcadexr::vulkan
