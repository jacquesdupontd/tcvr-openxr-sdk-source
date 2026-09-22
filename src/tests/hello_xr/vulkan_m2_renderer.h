#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
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
#include "m2_frag_spv.h"
#include "m2_frag_nd_spv.h"
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
    bool Initialize(VkDevice device, const MemoryAllocator* allocator, VkFormat colorFormat, uint32_t samples) {
        if (m_initialized) return true;
        m_vkDevice = device;
        m_memAllocator = allocator;
        m_colorFormat = colorFormat;
        m_samples = (samples >= 4) ? VK_SAMPLE_COUNT_4_BIT : (samples >= 2 ? VK_SAMPLE_COUNT_2_BIT : VK_SAMPLE_COUNT_1_BIT);
        CreateRenderPass();
        VkRenderPass renderPass = m_pass;

        Log::Write(Log::Level::Info, "TCVR_M2VK: Initializing Model 2 native Vulkan immersive renderer");

        // 1. Create Descriptor Pool
        std::array<VkDescriptorPoolSize, 5> poolSizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4 * M2RegionTextures::kMaxSlots},
            {VK_DESCRIPTOR_TYPE_SAMPLER, 16}
        }};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 16;
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
            ssboBind.stageFlags = (b == 1) ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                                          : VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(ssboBind);
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
            sb.descriptorCount = 4; sb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            m2Bindings.push_back(sb);
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

    void PrepareFrame(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        if (!m_initialized) return;

        if (frame.geometry_unchanged != 0u) {
            UploadLayers(frame, cmd);
            return;
        }

        const bool haveClipped = frame.prim_count != 0 && frame.vertex_count != 0;
        const bool haveRaw = frame.raw_prim_count != 0 && frame.raw_vertex_count != 0;
        if (!haveClipped && !haveRaw) return;

        // Upload colour chain & textures
        UploadColourChain(frame);
        const bool texChanged = UploadTextures(frame, cmd);
        if (texChanged && m_regions.Count() > 0) {
            vkDeviceWaitIdle(m_vkDevice);   // rare: the sheets changed (course / menu load)
            m_regions.Clear();
        }

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
            const std::int64_t halfScreen = std::int64_t(496) * std::int64_t(384) / 2;
            for (unsigned k = 0; k < n; ++k) {
                const std::int64_t area = std::int64_t(keys[k].r - keys[k].l + 1) * std::int64_t(keys[k].b - keys[k].t + 1);
                const int sx = frame.crtc_xoffset + keys[k].cx;
                const int sy = (384 - keys[k].cy) + frame.crtc_yoffset;
                const bool centred = std::abs(sx - 496 / 2) <= 496 / 8 && std::abs(sy - 384 / 2) <= 384 / 8;
                if (area < halfScreen || !centred) continue;
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
        if (m_haveMainView) {
            std::vector<float> farRows;
            for (std::uint32_t i = 0; i < kn; i++) {
                const tcvr_m2_prim& p = kp[i];
                if (p.clip_l != mainL || p.clip_t != mainT || p.clip_r != mainR || p.clip_b != mainB ||
                    p.center_x != mainCx || p.center_y != mainCy) continue;
                for (std::uint32_t v = 0; v < p.vertex_count; v++) {
                    const std::uint32_t index = p.first_vertex + v;
                    if (frame.raw_prim_count) {
                        if (index >= frame.raw_vertex_count) break;
                        const tcvr_m2_raw_vertex& rv = frame.raw_vertices[index];
                        if (rv.z >= 1000.0f) farRows.push_back((384.0f - float(p.center_y)) + float(frame.crtc_yoffset) - rv.y / rv.z);
                    }
                }
            }
            if (farRows.size() >= 32) {
                std::nth_element(farRows.begin(), farRows.begin() + farRows.size() / 2, farRows.end());
                m_horizonGeo = farRows[farRows.size() / 2];
            }
        }

        m_mainClip[0] = mainL; m_mainClip[1] = mainT; m_mainClip[2] = mainR; m_mainClip[3] = mainB;
        m_mainCenter[0] = mainCx; m_mainCenter[1] = mainCy;

        // Build raw pre-clip geometry
        if (frame.raw_prim_count > 0 && frame.raw_vertex_count > 0) {
            const std::uint32_t n = std::min<std::uint32_t>(frame.raw_prim_count, 0xffffu);
            m_rawKeys.resize(n);
            for (std::uint32_t i = 0; i < n; i++)
                m_rawKeys[i] = (std::uint32_t(frame.raw_prims[i].zsort & 0xffffu) << 16) | (0xffffu - i);
            std::sort(m_rawKeys.begin(), m_rawKeys.end());

            m_rawPrims.resize(n);
            m_primSlot.assign(n, 0xffffffffu);
            std::size_t vcount = 0, icount = 0;
            for (std::uint32_t k = 0; k < n; k++) {
                const tcvr_m2_prim& p = frame.raw_prims[0xffffu - (m_rawKeys[k] & 0xffffu)];
                const std::uint32_t vc = std::min<std::uint32_t>(p.vertex_count, frame.raw_vertex_count - std::min(p.first_vertex, frame.raw_vertex_count));
                vcount += vc; if (vc >= 3) icount += (vc - 2) * 3;
            }

            m_rawVerts.resize(vcount * 5);
            m_rawPrimOfVertex.resize(vcount);
            m_rawIdx.resize(icount);
            std::vector<std::uint32_t> rawGlassIdx, rawCutIdx;
            rawGlassIdx.reserve(icount / 8);
            rawCutIdx.reserve(icount / 4);

            std::size_t vo = 0, io = 0;
            std::uint32_t last_zsort = 0xffffffffu;
            std::uint32_t intra_bucket_rank = 0;

            for (std::uint32_t k = 0; k < n; k++) {
                const std::uint32_t cur_zsort = m_rawKeys[k] >> 16;
                if (cur_zsort == last_zsort) {
                    intra_bucket_rank++;
                } else {
                    last_zsort = cur_zsort;
                    intra_bucket_rank = 0;
                }
                const tcvr_m2_prim& p = frame.raw_prims[0xffffu - (m_rawKeys[k] & 0xffffu)];
                tcvr_m2_prim q = p;
                q.first_vertex = std::uint32_t(vo);
                const std::uint32_t vc = std::min<std::uint32_t>(p.vertex_count, frame.raw_vertex_count - std::min(p.first_vertex, frame.raw_vertex_count));
                q.vertex_count = vc;
                q.zsort = intra_bucket_rank;
                m_rawPrims[k] = q;
                {
                    uint32_t slot = M2RegionTextures::kNone, micro = M2RegionTextures::kNone;
                    if (m_useRegions && q.textured != 0u) {
                        slot = m_regions.Slot(m_sheetCpu, q.texsheet & 1u, (q.texx - 2048u) & 2047u, (q.texy - 1024u) & 1023u,
                                              q.texwidth, q.texheight);
                        if (q.utex != 0u) micro = m_regions.Slot(m_sheetCpu, (1u - q.texsheet) & 1u, q.utexx, q.utexy, 128, 128);
                    }
                    m_primSlot[k] = slot | (micro << 16);
                }

                for (std::uint32_t v = 0; v < vc; v++) {
                    const tcvr_m2_raw_vertex& rv = frame.raw_vertices[p.first_vertex + v];
                    float* d = &m_rawVerts[(vo + v) * 5];
                    d[0] = rv.x; d[1] = rv.y; d[2] = rv.z; d[3] = rv.u; d[4] = rv.v;
                    m_rawPrimOfVertex[vo + v] = k;
                }

                const bool isGlass = (q.checker != 0u);
                const bool isMain = q.center_x == mainCx && q.center_y == mainCy &&
                                    std::abs(q.clip_l - mainL) <= 2 && std::abs(q.clip_t - mainT) <= 2 &&
                                    std::abs(q.clip_r - mainR) <= 2 && std::abs(q.clip_b - mainB) <= 2;
                // Untextured + translucent draws NOTHING on the board (draw_scanline_solid returns).
                const bool invisible = q.textured == 0u && q.translucent != 0u;
                // Discard-free: opaque, no stipple, main camera (secondary views need the clip test).
                const bool fast = !isGlass && isMain && (q.translucent == 0u || !RegionHasHoles(q));
                if (invisible) {
                } else if (!isGlass && !fast) {
                    for (std::uint32_t t = 1; t + 1 < vc; t++) {
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
                    for (std::uint32_t t = 1; t + 1 < vc; t++) {
                        m_rawIdx[io++] = q.first_vertex;
                        m_rawIdx[io++] = q.first_vertex + t;
                        m_rawIdx[io++] = q.first_vertex + t + 1;
                    }
                }
                vo += vc;
            }

            m_fastIndexCount = unsigned(io);
            for (std::uint32_t idx : rawCutIdx) m_rawIdx[io++] = idx;
            m_opaqueIndexCount = unsigned(io);
            m_glassIndexCount = unsigned(rawGlassIdx.size());
            for (std::uint32_t idx : rawGlassIdx) {
                m_rawIdx[io++] = idx;
            }
            m_rawIdx.resize(io);

            // Copy to host-visible buffers
            if (!m_rawVerts.empty()) {
                size_t vertBytes = std::min(m_rawVerts.size() * sizeof(float), m_vboSize);
                memcpy(m_vboMapped, m_rawVerts.data(), vertBytes);
            }
            if (!m_rawPrimOfVertex.empty()) {
                size_t primOfVertBytes = std::min(m_rawPrimOfVertex.size() * sizeof(uint32_t), m_primIndexSize);
                memcpy(m_primIndexMapped, m_rawPrimOfVertex.data(), primOfVertBytes);
            }
            if (!m_rawIdx.empty()) {
                size_t idxBytes = std::min(m_rawIdx.size() * sizeof(uint32_t), m_iboSize);
                memcpy(m_iboMapped, m_rawIdx.data(), idxBytes);
            }
            // first_vertex is not read by any shader: it now carries the region slots
            // (main | microtexture << 16) to the vertex stage, which hands them on flat.
            for (std::uint32_t k = 0; k < n; k++) m_rawPrims[k].first_vertex = m_primSlot[k];
            {   // always: the descriptor array must be fully valid (dummy image) before any draw
                m_regions.Flush(cmd);
                const bool full = m_regions.NeedsFullDescriptorWrite();
                if (full || m_regions.HasDirty())
                    for (int e = 0; e < 2; ++e)
                        for (int ps = 0; ps < 2; ++ps) m_regions.WriteDescriptors(m_m2DescSet[e][ps], 12, 13, full);
                m_regions.DescriptorsDone();
                static unsigned s_regLog = 0;
                if ((s_regLog++ % 300u) == 0u)
                    Log::Write(Log::Level::Info, Fmt("TCVR_M2VK regions=%u created=%u", m_regions.Count(), m_regions.Created()));
            }
            if (!m_rawPrims.empty()) {
                size_t primBytes = std::min(m_rawPrims.size() * sizeof(tcvr_m2_prim), m_primsSize);
                memcpy(m_primsMapped, m_rawPrims.data(), primBytes);
            }
        }

        UploadLayers(frame, cmd);
        m_preparedSeq = frame.sequence;
    }

    bool RenderImmersive(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView,
                         VkCommandBuffer cmd, VkExtent2D renderAreaExtent) {
        if (!m_initialized || m_opaqueIndexCount == 0) return false;

        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) return false;

        const float distance = std::max(0.25f, arcadexr::config::GetFloat("screen.distance", 2.0f));
        const float depthUnits = std::max(0.5f, arcadexr::config::GetFloat("m2.immersiveDepth", 9.4f));
        const float worldScale = distance / depthUnits;
        const arcadexr::gun::Vec3 camera{screen.center.x + screen.normal.x * distance,
                                         screen.center.y + screen.normal.y * distance,
                                         screen.center.z + screen.normal.z * distance};

        const float focusY = (m_m2FocusY > 1.0f) ? m_m2FocusY : 512.0f;
        const float focusX = (m_m2FocusX > 1.0f) ? m_m2FocusX : 512.0f;

        float pitchTarget = 0.0f;
        if (arcadexr::config::GetInt("m2.immersivePitch", 1) != 0 && m_haveMainView && m_horizonGeo >= 0.0f) {
            const float t = ((384.0f - float(m_mainCenter[1])) - m_horizonGeo) / std::max(focusY, 1.0f);
            pitchTarget = std::max(-0.35f, std::min(0.35f, ::atanf(t)));
        }
        pitchTarget += arcadexr::config::GetFloat("m2.immersivePitchOffset", 0.0f);
        if (viewIndex == 0) m_m2Pitch += (pitchTarget - m_m2Pitch) * 0.1f;

        const float cp = ::cosf(m_m2Pitch), sp = ::sinf(m_m2Pitch);
        const arcadexr::gun::Vec3 upP{cp * screen.up.x - sp * screen.normal.x, cp * screen.up.y - sp * screen.normal.y, cp * screen.up.z - sp * screen.normal.z};
        const arcadexr::gun::Vec3 normalP{cp * screen.normal.x + sp * screen.up.x, cp * screen.normal.y + sp * screen.up.y, cp * screen.normal.z + sp * screen.up.z};

        XrMatrix4x4f arcadeToWorld{};
        arcadeToWorld.m[0] = screen.right.x * worldScale;  arcadeToWorld.m[1] = screen.right.y * worldScale;  arcadeToWorld.m[2] = screen.right.z * worldScale;
        arcadeToWorld.m[4] = upP.x * worldScale;          arcadeToWorld.m[5] = upP.y * worldScale;          arcadeToWorld.m[6] = upP.z * worldScale;
        arcadeToWorld.m[8] = -normalP.x * worldScale;     arcadeToWorld.m[9] = -normalP.y * worldScale;     arcadeToWorld.m[10] = -normalP.z * worldScale;
        arcadeToWorld.m[12] = camera.x; arcadeToWorld.m[13] = camera.y; arcadeToWorld.m[14] = camera.z; arcadeToWorld.m[15] = 1.0f;

        const float farMetres = std::max(200.0f, arcadexr::config::GetFloat("immersive.far", 20000.0f));
        const float nearMetres = std::max(0.002f, std::min(1.0f, arcadexr::config::GetFloat("m2.immersiveNear", 0.25f)));

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

        const float hudDistance = std::max(0.5f, arcadexr::config::GetFloat("m2.hudDistance", 10.0f));
        const float hudK = hudDistance / distance;
        const float hudLift = arcadexr::config::GetFloat("m2.hudLift", 1.8f);
        const arcadexr::gun::Vec3 hudCenter{camera.x - screen.normal.x * hudDistance + screen.up.x * hudLift,
                                            camera.y - screen.normal.y * hudDistance + screen.up.y * hudLift,
                                            camera.z - screen.normal.z * hudDistance + screen.up.z * hudLift};

        XrMatrix4x4f hudToWorld{};
        hudToWorld.m[0] = screen.right.x * screen.width * hudK; hudToWorld.m[1] = screen.right.y * screen.width * hudK; hudToWorld.m[2] = screen.right.z * screen.width * hudK;
        hudToWorld.m[4] = screen.up.x * screen.height * hudK;   hudToWorld.m[5] = screen.up.y * screen.height * hudK;   hudToWorld.m[6] = screen.up.z * screen.height * hudK;
        hudToWorld.m[8] = screen.normal.x; hudToWorld.m[9] = screen.normal.y; hudToWorld.m[10] = screen.normal.z;
        hudToWorld.m[12] = hudCenter.x; hudToWorld.m[13] = hudCenter.y; hudToWorld.m[14] = hudCenter.z; hudToWorld.m[15] = 1.0f;
        XrMatrix4x4f hudMvp;
        XrMatrix4x4f_Multiply(&hudMvp, &viewProjection, &hudToWorld);

        XrMatrix4x4f hudToWorldBack{};
        const arcadexr::gun::Vec3 backCenter{camera.x - normalP.x * hudDistance + screen.up.x * hudLift,
                                             camera.y - normalP.y * hudDistance + screen.up.y * hudLift,
                                             camera.z - normalP.z * hudDistance + screen.up.z * hudLift};
        hudToWorldBack.m[0] = screen.right.x * screen.width * hudK; hudToWorldBack.m[1] = screen.right.y * screen.width * hudK; hudToWorldBack.m[2] = screen.right.z * screen.width * hudK;
        hudToWorldBack.m[4] = upP.x * screen.height * hudK;         hudToWorldBack.m[5] = upP.y * screen.height * hudK;         hudToWorldBack.m[6] = upP.z * screen.height * hudK;
        hudToWorldBack.m[8] = normalP.x; hudToWorldBack.m[9] = normalP.y; hudToWorldBack.m[10] = normalP.z;
        hudToWorldBack.m[12] = backCenter.x; hudToWorldBack.m[13] = backCenter.y; hudToWorldBack.m[14] = backCenter.z; hudToWorldBack.m[15] = 1.0f;
        XrMatrix4x4f hudMvpBack;
        XrMatrix4x4f_Multiply(&hudMvpBack, &viewProjection, &hudToWorldBack);

        // Update UBO slices for this eye
        const uint32_t eye = (viewIndex < 2) ? viewIndex : 0;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            M2UniformBufferObject& ubo = *m_uboMapped[eye][pass];
            memcpy(ubo.uMvp, mvp.m, sizeof(mvp.m));
            memcpy(ubo.uHudMvp, hudMvp.m, sizeof(hudMvp.m));
            ubo.uViewport[0] = 496.0f; ubo.uViewport[1] = 384.0f;
            ubo.uFocus[0] = focusX; ubo.uFocus[1] = focusY;
            ubo.uCrtc[0] = 0.0f; ubo.uCrtc[1] = 0.0f;
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
            ubo.uPrepassClass = 0;
            ubo.uEdgeFade = 0.0f;
            ubo.uHorizonRow = m_horizonGeo;
            ubo.uSky[0] = m_voidColor[0]; ubo.uSky[1] = m_voidColor[1]; ubo.uSky[2] = m_voidColor[2];
            ubo.uGround[0] = m_groundColor[0]; ubo.uGround[1] = m_groundColor[1]; ubo.uGround[2] = m_groundColor[2];
            ubo.uAniso = std::max(1, std::min(8, arcadexr::config::GetInt("m2.aniso", 1)));
            ubo.uFilterMode = std::max(0, std::min(5, arcadexr::config::GetInt("m2.filter", 5)));
            ubo.uMipBias = std::max(0, std::min(512, arcadexr::config::GetInt("m2.mipBias", 0)));
            ubo.uAlphaCoverage = IsMsaa() ? arcadexr::config::GetInt("m2.alphaCoverage", 1) : 0;
            ubo.uContrast = arcadexr::config::GetFloat("contrast", 1.2f);
            ubo.uBright = arcadexr::config::GetFloat("bright", -0.02f);
            ubo.uTestStage = arcadexr::config::GetInt("m2.stage", 0);
            ubo.uCountOverdraw = (m_useRegions && arcadexr::config::GetInt("m2.regions", 1) != 0) ? 1 : 0;  // = uUseRegions
        }

        const float outW = float(renderAreaExtent.width);
        const float outH = float(renderAreaExtent.height);

        // Debug, live: debug.tcvr.m2.vkskip bitmask to time each pass on the headset.
        // 1 void, 2 back plane, 4 discard-free opaque, 8 cut-outs/secondary, 16 glass, 32 front HUD.
        const int skip = arcadexr::config::GetInt("m2.vkskip", 0);
        const bool bgFar = true;  // background is drawn after the geometry: far-plane, depth-tested
        // 3. Draw Model 2 Geometry
        if (m_opaqueIndexCount > 0) {
            VkBuffer vtxBufs[2] = { m_vboBuffer.buf, m_primIndexBuffer.buf };
            VkDeviceSize offsets[2] = { 0, 0 };
            vkCmdBindVertexBuffers(cmd, 0, 2, vtxBufs, offsets);
            vkCmdBindIndexBuffer(cmd, m_iboBuffer.buf, 0, VK_INDEX_TYPE_UINT32);

            // Pass 1: Opaque
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSet[eye][0], 0, nullptr);
            if (m_fastIndexCount > 0 && !(skip & 4)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineFast);
                vkCmdDrawIndexed(cmd, m_fastIndexCount, 1, 0, 0, 0);
            }
            if (m_opaqueIndexCount > m_fastIndexCount && !(skip & 8)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineOpaque);
                vkCmdDrawIndexed(cmd, m_opaqueIndexCount - m_fastIndexCount, 1, m_fastIndexCount, 0, 0);
            }

            // Background AFTER the opaque geometry, depth-tested on the far plane (m2.bgAfter=1):
            // void + back 2D layer now shade only the pixels no polygon covered.
        // 1. DrawVoid
            float invMvp[16];
            if (!(skip & 1) && InvertMatrix4x4(mvp.m, invMvp)) {
                VoidPushConstants voidPc{};
                memcpy(voidPc.uInvMvp, invMvp, sizeof(invMvp));
                voidPc.uSky[0] = m_voidColor[0]; voidPc.uSky[1] = m_voidColor[1]; voidPc.uSky[2] = m_voidColor[2]; voidPc.uSky[3] = 1.0f;
                voidPc.uGround[0] = m_groundColor[0]; voidPc.uGround[1] = m_groundColor[1]; voidPc.uGround[2] = m_groundColor[2]; voidPc.uGround[3] = 1.0f;
                voidPc.uOutSize[0] = outW; voidPc.uOutSize[1] = outH;
    
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (bgFar ? m_voidPipelineFar : m_voidPipeline));
                vkCmdPushConstants(cmd, m_voidPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(voidPc), &voidPc);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
    
            // 2. DrawPlaneLayer (Back 2D)
            if (m_haveLayer[1] && !(skip & 2)) {
                PlanePushConstants backPc{};
                memcpy(backPc.uHudMvp, hudMvpBack.m, sizeof(hudMvpBack.m));
                backPc.uOutSize[0] = outW; backPc.uOutSize[1] = outH;
                backPc.uKeyZero = 0;
    
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, (bgFar ? m_planePipelineFar : m_planePipeline));
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipelineLayout, 0, 1, &m_layerDescSet[1], 0, nullptr);
                vkCmdPushConstants(cmd, m_planePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(backPc), &backPc);
                vkCmdDraw(cmd, 3, 1, 0, 0);
            }
    
    
            // Pass 2: Glass (after the background: it blends over what is really behind it)
            if (m_glassIndexCount > 0 && !(skip & 16)) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineGlass);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_m2PipelineLayout, 0, 1, &m_m2DescSet[eye][1], 0, nullptr);
                vkCmdDrawIndexed(cmd, m_glassIndexCount, 1, m_opaqueIndexCount, 0, 0);
            }
        }

        // 4. DrawPlaneLayer (Front 2D HUD)
        if (m_haveLayer[0] && arcadexr::config::GetInt("m2.hideHud", 0) == 0 && !(skip & 32)) {
            PlanePushConstants frontPc{};
            memcpy(frontPc.uHudMvp, hudMvp.m, sizeof(hudMvp.m));
            frontPc.uOutSize[0] = outW; frontPc.uOutSize[1] = outH;
            frontPc.uKeyZero = 1;

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_planePipelineLayout, 0, 1, &m_layerDescSet[0], 0, nullptr);
            vkCmdPushConstants(cmd, m_planePipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(frontPc), &frontPc);
            vkCmdDraw(cmd, 3, 1, 0, 0);
        }

        static unsigned s_immFrames = 0;
        if ((s_immFrames++ % 120u) == 0u) {
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK rendered eye=%u prims=%zu fast=%u opq=%u gls=%u res=%ux%u",
                                             eye, m_rawPrims.size(), m_fastIndexCount, m_opaqueIndexCount, m_glassIndexCount,
                                             renderAreaExtent.width, renderAreaExtent.height));
        }

        return true;
    }

    bool HaveMainView() const { return m_haveMainView; }

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
    bool HasGeometry() const { return m_initialized && m_opaqueIndexCount > 0; }
    bool IsMsaa() const { return m_samples != VK_SAMPLE_COUNT_1_BIT; }

    // Begin the immersive render pass on swapchain image `target` (eye `eye`).
    void BeginPass(VkCommandBuffer cmd, uint32_t eye, VkImage target, VkExtent2D ext, const VkRect2D& area, const float clear[4]) {
        eye = eye < 2 ? eye : 0;
        EyeTargets& et = m_eyeTargets[eye];
        if (et.ext.width != ext.width || et.ext.height != ext.height) DestroyEyeTargets(et), CreateEyeTargets(et, ext);
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_fbs) if (f.image == target && f.eye == eye) fb = f.fb;
        if (fb == VK_NULL_HANDLE) {
            FbEntry e{};
            e.image = target; e.eye = eye;
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = target; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = m_colorFormat;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &vi, nullptr, &e.view));
            std::array<VkImageView, 3> att{};
            uint32_t n = 0;
            if (IsMsaa()) { att[n++] = et.colorView; att[n++] = et.depthView; att[n++] = e.view; }
            else { att[n++] = e.view; att[n++] = et.depthView; }
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

    void CreateRenderPass() {
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
        ri.attachmentCount = ms ? 3u : 2u; ri.pAttachments = at.data();
        ri.subpassCount = 1; ri.pSubpasses = &sp;
        ri.dependencyCount = 1; ri.pDependencies = &dep;
        XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_vkDevice, &ri, nullptr, &m_pass));
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK render pass: MSAA x%d, transient colour/depth, on-tile resolve", int(m_samples)));
    }

    struct EyeTargets {
        VkExtent2D ext{0, 0};
        VkImage color = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
        VkDeviceMemory colorMem = VK_NULL_HANDLE, depthMem = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE, depthView = VK_NULL_HANDLE;
    };
    struct FbEntry { VkImage image; uint32_t eye; VkImageView view; VkFramebuffer fb; };
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
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); }
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

        vkDeviceWaitIdle(m_vkDevice);
        m_regions.Destroy();
        DestroyEyeTargets(m_eyeTargets[0]);
        DestroyEyeTargets(m_eyeTargets[1]);
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_vkDevice, f.fb, nullptr); vkDestroyImageView(m_vkDevice, f.view, nullptr); }
        m_fbs.clear();
        if (m_pass != VK_NULL_HANDLE) { vkDestroyRenderPass(m_vkDevice, m_pass, nullptr); m_pass = VK_NULL_HANDLE; }

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_vkDevice, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }

        auto destroyPipe = [&](VkPipeline& p) { if (p != VK_NULL_HANDLE) { vkDestroyPipeline(m_vkDevice, p, nullptr); p = VK_NULL_HANDLE; } };
        destroyPipe(m_m2PipelineOpaque);
        destroyPipe(m_m2PipelineFast);
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

        m_vboBuffer.Reset(m_vkDevice);
        m_primIndexBuffer.Reset(m_vkDevice);
        m_iboBuffer.Reset(m_vkDevice);
        m_primsBuffer.Reset(m_vkDevice);
        m_palramBuffer.Reset(m_vkDevice);
        m_colorxlatBuffer.Reset(m_vkDevice);
        m_lumaramBuffer.Reset(m_vkDevice);
        m_gammaBuffer.Reset(m_vkDevice);
        m_dummyBuffer.Reset(m_vkDevice);
        for (int i = 0; i < 4; ++i) {
            m_sheetStagingBuffer[i].Reset(m_vkDevice);
            m_sheetStagingMapped[i] = nullptr;
        }
        for (int i = 0; i < 2; ++i) m_layerStagingBuffer[i].Reset(m_vkDevice);
        m_texturesUploaded = false;
        m_texHash = 0;

        for (int e = 0; e < 2; ++e) {
            for (int p = 0; p < 2; ++p) {
                m_uboBuffer[e][p].Reset(m_vkDevice);
            }
        }
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
            pipeInfo.pMultisampleState = &msState;
            // Same state, fragment shader compiled WITHOUT any discard: the Adreno keeps its
            // early depth rejection (LRZ) on for these, so hidden fragments are never shaded.
            stages[1].module = m_m2FragNdModule;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_m2PipelineFast));
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

        // 1. UBO buffers: 4 slices (2 eyes x 2 passes)
        for (int e = 0; e < 2; ++e) {
            for (int p = 0; p < 2; ++p) {
                createMappedBuffer(m_uboBuffer[e][p], sizeof(M2UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                   reinterpret_cast<void**>(&m_uboMapped[e][p]));
            }
        }

        // 2. SSBOs
        m_primsSize = 4096 * sizeof(tcvr_m2_prim);
        createMappedBuffer(m_primsBuffer, m_primsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_primsMapped));

        m_palramSize = 65536 * sizeof(uint32_t);
        createMappedBuffer(m_palramBuffer, m_palramSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_palramMapped));

        m_colorxlatSize = 16384 * sizeof(uint32_t);
        createMappedBuffer(m_colorxlatBuffer, m_colorxlatSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_colorxlatMapped));

        m_lumaramSize = 4096 * sizeof(uint32_t);
        createMappedBuffer(m_lumaramBuffer, m_lumaramSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_lumaramMapped));

        m_gammaSize = 256 * sizeof(uint32_t);
        createMappedBuffer(m_gammaBuffer, m_gammaSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_gammaMapped));

        void* dummyMapped = nullptr;
        createMappedBuffer(m_dummyBuffer, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &dummyMapped);

        // 3. Geometry Buffers
        m_vboSize = 65536 * 5 * sizeof(float); // 32k verts
        createMappedBuffer(m_vboBuffer, m_vboSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_vboMapped));

        m_primIndexSize = 65536 * sizeof(uint32_t);
        createMappedBuffer(m_primIndexBuffer, m_primIndexSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_primIndexMapped));

        m_iboSize = 131072 * sizeof(uint32_t); // 65k indices
        createMappedBuffer(m_iboBuffer, m_iboSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                           reinterpret_cast<void**>(&m_iboMapped));

        // 4. Staging Buffers
        for (int i = 0; i < 4; ++i) {
            createMappedBuffer(m_sheetStagingBuffer[i], (i < 2 ? 1u : 2u) * 1024 * 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               reinterpret_cast<void**>(&m_sheetStagingMapped[i]));
        }

        for (int i = 0; i < 2; ++i) {
            createMappedBuffer(m_layerStagingBuffer[i], 512 * 384 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               reinterpret_cast<void**>(&m_layerStagingMapped[i]));
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

        // Allocate Model 2 descriptor sets (2 eyes x 2 passes = 4 sets)
        for (int e = 0; e < 2; ++e) {
            for (int p = 0; p < 2; ++p) {
                VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocInfo.descriptorPool = m_descriptorPool;
                allocInfo.descriptorSetCount = 1;
                allocInfo.pSetLayouts = &m_m2DescLayout;
                XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_m2DescSet[e][p]));

                std::vector<VkWriteDescriptorSet> writes;

                // Binding 0: UBO
                VkDescriptorBufferInfo uboInfo{m_uboBuffer[e][p].buf, 0, sizeof(M2UniformBufferObject)};
                VkWriteDescriptorSet uboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                uboWrite.dstSet = m_m2DescSet[e][p];
                uboWrite.dstBinding = 0;
                uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                uboWrite.descriptorCount = 1;
                uboWrite.pBufferInfo = &uboInfo;
                writes.push_back(uboWrite);

                // Binding 1: Prims
                VkDescriptorBufferInfo primsInfo{m_primsBuffer.buf, 0, m_primsSize};
                VkWriteDescriptorSet primsWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                primsWrite.dstSet = m_m2DescSet[e][p];
                primsWrite.dstBinding = 1;
                primsWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                primsWrite.descriptorCount = 1;
                primsWrite.pBufferInfo = &primsInfo;
                writes.push_back(primsWrite);

                // Binding 2: Palram
                VkDescriptorBufferInfo palramInfo{m_palramBuffer.buf, 0, m_palramSize};
                VkWriteDescriptorSet palramWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                palramWrite.dstSet = m_m2DescSet[e][p];
                palramWrite.dstBinding = 2;
                palramWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                palramWrite.descriptorCount = 1;
                palramWrite.pBufferInfo = &palramInfo;
                writes.push_back(palramWrite);

                // Binding 3: Colorxlat
                VkDescriptorBufferInfo colInfo{m_colorxlatBuffer.buf, 0, m_colorxlatSize};
                VkWriteDescriptorSet colWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                colWrite.dstSet = m_m2DescSet[e][p];
                colWrite.dstBinding = 3;
                colWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                colWrite.descriptorCount = 1;
                colWrite.pBufferInfo = &colInfo;
                writes.push_back(colWrite);

                // Binding 4: Lumaram
                VkDescriptorBufferInfo lumInfo{m_lumaramBuffer.buf, 0, m_lumaramSize};
                VkWriteDescriptorSet lumWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                lumWrite.dstSet = m_m2DescSet[e][p];
                lumWrite.dstBinding = 4;
                lumWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                lumWrite.descriptorCount = 1;
                lumWrite.pBufferInfo = &lumInfo;
                writes.push_back(lumWrite);

                // Binding 5: Gamma
                VkDescriptorBufferInfo gamInfo{m_gammaBuffer.buf, 0, m_gammaSize};
                VkWriteDescriptorSet gamWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                gamWrite.dstSet = m_m2DescSet[e][p];
                gamWrite.dstBinding = 5;
                gamWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                gamWrite.descriptorCount = 1;
                gamWrite.pBufferInfo = &gamInfo;
                writes.push_back(gamWrite);

                // Binding 6, 7: Dummy SSBOs
                VkDescriptorBufferInfo dummyInfo{m_dummyBuffer.buf, 0, 256};
                VkWriteDescriptorSet dum6Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                dum6Write.dstSet = m_m2DescSet[e][p];
                dum6Write.dstBinding = 6;
                dum6Write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dum6Write.descriptorCount = 1;
                dum6Write.pBufferInfo = &dummyInfo;
                writes.push_back(dum6Write);

                VkWriteDescriptorSet dum7Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                dum7Write.dstSet = m_m2DescSet[e][p];
                dum7Write.dstBinding = 7;
                dum7Write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                dum7Write.descriptorCount = 1;
                dum7Write.pBufferInfo = &dummyInfo;
                writes.push_back(dum7Write);

                // Binding 8, 9: Sheet texture samplers
                VkDescriptorImageInfo sheet0Info{m_sheetSampler, m_sheetView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet s0Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                s0Write.dstSet = m_m2DescSet[e][p];
                s0Write.dstBinding = 8;
                s0Write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                s0Write.descriptorCount = 1;
                s0Write.pImageInfo = &sheet0Info;
                writes.push_back(s0Write);

                VkDescriptorImageInfo sheet1Info{m_sheetSampler, m_sheetView[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                VkWriteDescriptorSet s1Write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                s1Write.dstSet = m_m2DescSet[e][p];
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
    }

    void UploadColourChain(const tcvr_m2_frame& frame) {
        if (frame.palram && frame.palram_entries) {
            size_t bytes = std::min(size_t(frame.palram_entries) * sizeof(uint16_t), m_palramSize);
            memcpy(m_palramMapped, frame.palram, bytes);
        }
        if (frame.colorxlat && frame.colorxlat_entries) {
            size_t bytes = std::min(size_t(frame.colorxlat_entries) * sizeof(uint16_t), m_colorxlatSize);
            memcpy(m_colorxlatMapped, frame.colorxlat, bytes);
        }
        if (frame.lumaram && frame.lumaram_entries) {
            size_t bytes = std::min(size_t(frame.lumaram_entries) * sizeof(uint8_t), m_lumaramSize);
            memcpy(m_lumaramMapped, frame.lumaram, bytes);
        }
        if (frame.gamma && frame.gamma_entries) {
            size_t bytes = std::min(size_t(frame.gamma_entries) * sizeof(uint8_t), m_gammaSize);
            memcpy(m_gammaMapped, frame.gamma, bytes);
        }
    }

    bool UploadTextures(const tcvr_m2_frame& frame, VkCommandBuffer cmd) {
        if (!frame.textureram[0] || !frame.textureram[1] || frame.textureram_words == 0) return false;

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

        const bool needsUpload = !m_texturesUploaded || hashChanged;
        if (!needsUpload) return false;

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
        m_texturesUploaded = true;
        m_sheetGeneration = frame.dirty_generation;
        m_regionHasHoles.clear();
        return true;
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
            for (uint32_t y = 0; y < copyH; ++y) {
                memcpy(&m_layerStagingMapped[idx][y * 512], &pixels[y * stride], copyW * sizeof(uint32_t));
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
            vkCmdCopyBufferToImage(cmd, m_layerStagingBuffer[idx].buf, m_layerImage[idx],
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

            m_layerLayout[idx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            m_haveLayer[idx] = true;
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
    VkFormat m_colorFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_samples = VK_SAMPLE_COUNT_1_BIT;
    VkRenderPass m_pass = VK_NULL_HANDLE;
    EyeTargets m_eyeTargets[2];
    std::vector<FbEntry> m_fbs;
    VkShaderModule m_quadFarVertModule = VK_NULL_HANDLE;
    VkPipeline m_voidPipelineFar = VK_NULL_HANDLE;
    VkPipeline m_planePipelineFar = VK_NULL_HANDLE;
    uint32_t m_fastIndexCount = 0;

    VkPipeline m_voidPipeline = VK_NULL_HANDLE;
    VkPipeline m_planePipeline = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineOpaque = VK_NULL_HANDLE;
    VkPipeline m_m2PipelineGlass = VK_NULL_HANDLE;

    VkSampler m_sheetSampler = VK_NULL_HANDLE;
    VkSampler m_layerSampler[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Host-visible mapped buffers
    BufferAndMemory m_uboBuffer[2][2]; // [eye][pass]
    M2UniformBufferObject* m_uboMapped[2][2] = {};

    BufferAndMemory m_primsBuffer;
    tcvr_m2_prim* m_primsMapped = nullptr;
    size_t m_primsSize = 0;

    BufferAndMemory m_palramBuffer;
    uint32_t* m_palramMapped = nullptr;
    size_t m_palramSize = 0;

    BufferAndMemory m_colorxlatBuffer;
    uint32_t* m_colorxlatMapped = nullptr;
    size_t m_colorxlatSize = 0;

    BufferAndMemory m_lumaramBuffer;
    uint32_t* m_lumaramMapped = nullptr;
    size_t m_lumaramSize = 0;

    BufferAndMemory m_gammaBuffer;
    uint32_t* m_gammaMapped = nullptr;
    size_t m_gammaSize = 0;

    BufferAndMemory m_dummyBuffer;

    BufferAndMemory m_vboBuffer;
    float* m_vboMapped = nullptr;
    size_t m_vboSize = 0;

    BufferAndMemory m_primIndexBuffer;
    uint32_t* m_primIndexMapped = nullptr;
    size_t m_primIndexSize = 0;

    BufferAndMemory m_iboBuffer;
    uint32_t* m_iboMapped = nullptr;
    size_t m_iboSize = 0;

    // Descriptor Sets
    VkDescriptorSet m_m2DescSet[2][2] = {};
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
    bool m_useRegions = false;
    std::vector<uint32_t> m_primSlot;                       // unpacked t per texel, for the region scan
    std::unordered_map<uint64_t, bool> m_regionHasHoles;      // region key -> contains texel 15
    uint64_t m_sheetGeneration = 0;
    uint64_t m_texHash = 0;
    bool m_texturesUploaded = false;

    VkImage m_layerImage[2] = {};
    VkDeviceMemory m_layerMem[2] = {};
    VkImageView m_layerView[2] = {};
    VkImageLayout m_layerLayout[2] = {};
    BufferAndMemory m_layerStagingBuffer[2];
    uint32_t* m_layerStagingMapped[2] = {};
    bool m_haveLayer[2] = {false, false};

    // Geometry unpack state
    std::vector<std::uint32_t> m_rawKeys;
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
    float m_groundColor[3] = {0.18f, 0.16f, 0.14f};
    float m_voidColor[3] = {0.16f, 0.36f, 0.78f};
    float m_m2Pitch = 0.0f;
    float m_m2FocusX = 512.0f;
    float m_m2FocusY = 512.0f;
};

}  // namespace arcadexr::vulkan
