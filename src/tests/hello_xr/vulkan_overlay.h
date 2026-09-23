#pragma once

// Overlay pass drawn on top of every eye image, whatever produced it (Model 2 immersive, screen mode,
// System 22 later): the game selector menu and the Time Crisis pistol. Ports of RenderMenu() and of the
// lit gun of graphicsplugin_opengles.cpp, same geometry, same lighting.
//
// One small render pass: the swapchain image LOADed and STOREd (1x), plus a transient depth buffer so
// the pistol hides its own back faces. Only recorded when there is something to draw.

#include <vulkan/vulkan.h>
#include <vector>
#include <cstring>
#include <algorithm>

#include "vulkan_utils.h"
#include "gun_mesh.h"
#include "ov_quad_vert_spv.h"
#include "ov_quad_frag_spv.h"
#include "ov_gun_vert_spv.h"
#include "ov_gun_frag_spv.h"

namespace arcadexr::vulkan {

class VulkanOverlay {
public:
    struct QuadPC { float mvp[16]; float tint[4]; };
    struct GunPC { float mvp[16]; float model0[4], model1[4], model2[4]; float eye[4]; };

    void Init(VkDevice dev, const MemoryAllocator* alloc, VkFormat colorFormat) {
        if (m_dev) return;
        m_dev = dev; m_alloc = alloc; m_format = colorFormat;
        // Render pass: colour LOAD/STORE, transient depth CLEAR.
        VkAttachmentDescription at[2]{};
        at[0].format = colorFormat; at[0].samples = VK_SAMPLE_COUNT_1_BIT;
        at[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; at[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        at[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; at[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        at[1].format = VK_FORMAT_D32_SFLOAT; at[1].samples = VK_SAMPLE_COUNT_1_BIT;
        at[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; at[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; at[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        at[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; at[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount = 1; sp.pColorAttachments = &cref; sp.pDepthStencilAttachment = &dref;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ri.attachmentCount = 2; ri.pAttachments = at; ri.subpassCount = 1; ri.pSubpasses = &sp;
        ri.dependencyCount = 1; ri.pDependencies = &dep;
        XRC_CHECK_THROW_VKCMD(vkCreateRenderPass(m_dev, &ri, nullptr, &m_pass));

        // Descriptor: one combined sampler (the menu texture).
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = 1; li.pBindings = &b;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_dev, &li, nullptr, &m_setLayout));
        VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &m_setLayout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_dev, &pli, nullptr, &m_layout));
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 4; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorPool(m_dev, &pi, nullptr, &m_pool));
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = m_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &m_setLayout;
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_dev, &ai, &m_menuSet));
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_dev, &si, nullptr, &m_sampler));

        CreatePipelines();
        CreateGunBuffers();
    }

    void Destroy() {
        if (!m_dev) return;
        vkDeviceWaitIdle(m_dev);
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_dev, f.fb, nullptr); vkDestroyImageView(m_dev, f.view, nullptr); }
        m_fbs.clear();
        DestroyDepth();
        auto dp = [&](VkPipeline& p) { if (p) vkDestroyPipeline(m_dev, p, nullptr); p = VK_NULL_HANDLE; };
        dp(m_quadPipe); dp(m_gunPipe);
        if (m_layout) vkDestroyPipelineLayout(m_dev, m_layout, nullptr);
        if (m_setLayout) vkDestroyDescriptorSetLayout(m_dev, m_setLayout, nullptr);
        if (m_pool) vkDestroyDescriptorPool(m_dev, m_pool, nullptr);
        if (m_sampler) vkDestroySampler(m_dev, m_sampler, nullptr);
        if (m_pass) vkDestroyRenderPass(m_dev, m_pass, nullptr);
        DestroyMenuImage();
        m_gunVbo.Reset(m_dev); m_gunIbo.Reset(m_dev); m_menuStaging.Reset(m_dev);
        m_dev = VK_NULL_HANDLE;
    }

    // Upload a new menu picture (RGBA8, w x h). Recorded into cmd (outside any render pass).
    void UploadMenu(VkCommandBuffer cmd, const std::vector<unsigned char>& rgba, int w, int h) {
        if (w <= 0 || h <= 0 || rgba.size() < size_t(w) * h * 4) return;
        if (w != m_menuW || h != m_menuH) {
            vkDeviceWaitIdle(m_dev);   // rare (menu size change)
            DestroyMenuImage();
            m_menuW = w; m_menuH = h;
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_R8G8B8A8_SRGB;   // menu bytes are display values (see screen texture) ii.extent = {uint32_t(w), uint32_t(h), 1};
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT; ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            XRC_CHECK_THROW_VKCMD(vkCreateImage(m_dev, &ii, nullptr, &m_menuImage));
            VkMemoryRequirements req{}; vkGetImageMemoryRequirements(m_dev, m_menuImage, &req);
            m_alloc->Allocate(req, &m_menuMem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_dev, m_menuImage, m_menuMem, 0));
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = m_menuImage; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_R8G8B8A8_SRGB;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &vi, nullptr, &m_menuView));
            m_menuStaging.Reset(m_dev);
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = VkDeviceSize(w) * h * 4 * 2; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // two halves: frames in flight
            m_menuStaging.Create(m_dev, *m_alloc, bi);
            XRC_CHECK_THROW_VKCMD(vkMapMemory(m_dev, m_menuStaging.mem, 0, bi.size, 0, reinterpret_cast<void**>(&m_menuMapped)));
            VkDescriptorImageInfo di{m_sampler, m_menuView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wd.dstSet = m_menuSet; wd.dstBinding = 0; wd.descriptorCount = 1;
            wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wd.pImageInfo = &di;
            vkUpdateDescriptorSets(m_dev, 1, &wd, 0, nullptr);
            m_menuLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
        const size_t bytes = size_t(w) * h * 4;
        m_menuHalf ^= 1u;
        std::memcpy(m_menuMapped + m_menuHalf * bytes, rgba.data(), bytes);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = m_menuLayout; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = m_menuImage; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy r{};
        r.bufferOffset = m_menuHalf * bytes;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyBufferToImage(cmd, m_menuStaging.buf, m_menuImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        m_menuLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    bool HaveMenuImage() const { return m_menuLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }
    float MenuAspect() const { return m_menuH > 0 ? float(m_menuW) / float(m_menuH) : 1.0f; }

    void Begin(VkCommandBuffer cmd, uint32_t eye, VkImage target, VkExtent2D ext, const VkRect2D& area) {
        if (m_depthExt.width != ext.width || m_depthExt.height != ext.height) { DestroyDepth(); CreateDepth(ext); }
        VkFramebuffer fb = VK_NULL_HANDLE;
        for (auto& f : m_fbs) if (f.image == target) fb = f.fb;
        if (!fb) {
            Fb f{}; f.image = target;
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = target; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = m_format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &vi, nullptr, &f.view));
            VkImageView att[2] = {f.view, m_depthView};
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = m_pass; fi.attachmentCount = 2; fi.pAttachments = att;
            fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
            XRC_CHECK_THROW_VKCMD(vkCreateFramebuffer(m_dev, &fi, nullptr, &f.fb));
            m_fbs.push_back(f); fb = f.fb;
        }
        (void)eye;
        VkClearValue cv[2]{}; cv[1].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        bi.renderPass = m_pass; bi.framebuffer = fb; bi.renderArea = area; bi.clearValueCount = 2; bi.pClearValues = cv;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp{float(area.offset.x), float(area.offset.y), float(area.extent.width), float(area.extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &area);
    }
    void End(VkCommandBuffer cmd) { vkCmdEndRenderPass(cmd); }

    void DrawMenu(VkCommandBuffer cmd, const float mvp[16]) {
        if (!HaveMenuImage()) return;
        QuadPC pc{}; std::memcpy(pc.mvp, mvp, 64); pc.tint[0] = pc.tint[1] = pc.tint[2] = pc.tint[3] = 1.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_quadPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &m_menuSet, 0, nullptr);
        vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(cmd, 6, 1, 0, 0);
    }

    // model: column-major XrMatrix4x4f of the gun pose; eye: eye position (world).
    void DrawGun(VkCommandBuffer cmd, const float mvp[16], const float model[16], const float eye[3]) {
        if (!m_gunIndexCount) return;
        GunPC pc{};
        std::memcpy(pc.mvp, mvp, 64);
        for (int r = 0; r < 3; ++r) {   // rows of the model matrix (column-major input)
            float* row = r == 0 ? pc.model0 : (r == 1 ? pc.model1 : pc.model2);
            for (int c = 0; c < 4; ++c) row[c] = model[c * 4 + r];
        }
        pc.eye[0] = eye[0]; pc.eye[1] = eye[1]; pc.eye[2] = eye[2]; pc.eye[3] = 1.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gunPipe);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_gunVbo.buf, &off);
        vkCmdBindIndexBuffer(cmd, m_gunIbo.buf, 0, VK_INDEX_TYPE_UINT16);
        vkCmdPushConstants(cmd, m_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        vkCmdDrawIndexed(cmd, m_gunIndexCount, 1, 0, 0, 0);
    }

private:
    struct Fb { VkImage image; VkImageView view; VkFramebuffer fb; };

    VkShaderModule Mod(const uint8_t* code, size_t n) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = n; ci.pCode = reinterpret_cast<const uint32_t*>(code);
        VkShaderModule m = VK_NULL_HANDLE;
        XRC_CHECK_THROW_VKCMD(vkCreateShaderModule(m_dev, &ci, nullptr, &m));
        return m;
    }

    void CreatePipelines() {
        VkShaderModule qv = Mod(c_ovQuadVertSpv, sizeof(c_ovQuadVertSpv)), qf = Mod(c_ovQuadFragSpv, sizeof(c_ovQuadFragSpv));
        VkShaderModule gv = Mod(c_ovGunVertSpv, sizeof(c_ovGunVertSpv)), gf = Mod(c_ovGunFragSpv, sizeof(c_ovGunFragSpv));
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2; ds.pDynamicStates = dyn;
        VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vps.viewportCount = 1; vps.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[0].pName = st[1].pName = "main";
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 2; pi.pStages = st; pi.pInputAssemblyState = &ia; pi.pViewportState = &vps;
        pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDynamicState = &ds;
        pi.layout = m_layout; pi.renderPass = m_pass;
        // Menu: alpha blended, no depth.
        {
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            VkPipelineColorBlendAttachmentState cba{};
            cba.blendEnable = VK_TRUE;
            cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.colorBlendOp = VK_BLEND_OP_ADD;
            cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            cba.alphaBlendOp = VK_BLEND_OP_ADD;
            cba.colorWriteMask = 0xf;
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = 1; cb.pAttachments = &cba;
            st[0].module = qv; st[1].module = qf;
            pi.pVertexInputState = &vi; pi.pDepthStencilState = &dss; pi.pColorBlendState = &cb;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pi, nullptr, &m_quadPipe));
        }
        // Gun: opaque, depth tested and written (its own transient depth).
        {
            VkVertexInputBindingDescription bd{0, sizeof(arcadexr::gun::MeshVertex), VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription ad[4] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(arcadexr::gun::MeshVertex, position)},
                {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(arcadexr::gun::MeshVertex, normal)},
                {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(arcadexr::gun::MeshVertex, color)},
                {3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(arcadexr::gun::MeshVertex, material)}};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bd;
            vi.vertexAttributeDescriptionCount = 4; vi.pVertexAttributeDescriptions = ad;
            VkPipelineDepthStencilStateCreateInfo dss{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            dss.depthTestEnable = VK_TRUE; dss.depthWriteEnable = VK_TRUE; dss.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
            VkPipelineColorBlendAttachmentState cba{}; cba.colorWriteMask = 0xf;
            VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb.attachmentCount = 1; cb.pAttachments = &cba;
            st[0].module = gv; st[1].module = gf;
            pi.pVertexInputState = &vi; pi.pDepthStencilState = &dss; pi.pColorBlendState = &cb;
            XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pi, nullptr, &m_gunPipe));
        }
        vkDestroyShaderModule(m_dev, qv, nullptr); vkDestroyShaderModule(m_dev, qf, nullptr);
        vkDestroyShaderModule(m_dev, gv, nullptr); vkDestroyShaderModule(m_dev, gf, nullptr);
    }

    void CreateGunBuffers() {
        const auto& mesh = arcadexr::gun::GunMesh();
        if (mesh.vertices.empty() || mesh.indices.empty()) return;
        auto mk = [&](BufferAndMemory& bm, const void* data, size_t n, VkBufferUsageFlags usage) {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = n; bi.usage = usage;
            bm.Create(m_dev, *m_alloc, bi);
            void* p = nullptr;
            XRC_CHECK_THROW_VKCMD(vkMapMemory(m_dev, bm.mem, 0, n, 0, &p));
            std::memcpy(p, data, n);
            vkUnmapMemory(m_dev, bm.mem);
        };
        mk(m_gunVbo, mesh.vertices.data(), mesh.vertices.size() * sizeof(arcadexr::gun::MeshVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        mk(m_gunIbo, mesh.indices.data(), mesh.indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        m_gunIndexCount = uint32_t(mesh.indices.size());
    }

    void CreateDepth(VkExtent2D ext) {
        m_depthExt = ext;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = VK_FORMAT_D32_SFLOAT; ii.extent = {ext.width, ext.height, 1};
        ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        XRC_CHECK_THROW_VKCMD(vkCreateImage(m_dev, &ii, nullptr, &m_depth));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(m_dev, m_depth, &req);
        try { m_alloc->Allocate(req, &m_depthMem, VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT); }
        catch (...) { m_alloc->Allocate(req, &m_depthMem, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); }
        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_dev, m_depth, m_depthMem, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = m_depth; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = VK_FORMAT_D32_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_dev, &vi, nullptr, &m_depthView));
    }
    void DestroyDepth() {
        if (!m_depth) return;
        vkDeviceWaitIdle(m_dev);
        for (auto& f : m_fbs) { vkDestroyFramebuffer(m_dev, f.fb, nullptr); vkDestroyImageView(m_dev, f.view, nullptr); }
        m_fbs.clear();
        vkDestroyImageView(m_dev, m_depthView, nullptr); vkDestroyImage(m_dev, m_depth, nullptr); vkFreeMemory(m_dev, m_depthMem, nullptr);
        m_depth = VK_NULL_HANDLE; m_depthView = VK_NULL_HANDLE; m_depthMem = VK_NULL_HANDLE; m_depthExt = {0, 0};
    }
    void DestroyMenuImage() {
        if (m_menuView) vkDestroyImageView(m_dev, m_menuView, nullptr);
        if (m_menuImage) vkDestroyImage(m_dev, m_menuImage, nullptr);
        if (m_menuMem) vkFreeMemory(m_dev, m_menuMem, nullptr);
        m_menuView = VK_NULL_HANDLE; m_menuImage = VK_NULL_HANDLE; m_menuMem = VK_NULL_HANDLE;
        m_menuW = m_menuH = 0; m_menuLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    VkDevice m_dev = VK_NULL_HANDLE;
    const MemoryAllocator* m_alloc = nullptr;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkRenderPass m_pass = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_menuSet = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkPipeline m_quadPipe = VK_NULL_HANDLE, m_gunPipe = VK_NULL_HANDLE;
    BufferAndMemory m_gunVbo, m_gunIbo, m_menuStaging;
    uint32_t m_gunIndexCount = 0;
    VkImage m_menuImage = VK_NULL_HANDLE;
    VkDeviceMemory m_menuMem = VK_NULL_HANDLE;
    VkImageView m_menuView = VK_NULL_HANDLE;
    VkImageLayout m_menuLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    unsigned char* m_menuMapped = nullptr;
    uint32_t m_menuHalf = 0;
    int m_menuW = 0, m_menuH = 0;
    VkImage m_depth = VK_NULL_HANDLE;
    VkDeviceMemory m_depthMem = VK_NULL_HANDLE;
    VkImageView m_depthView = VK_NULL_HANDLE;
    VkExtent2D m_depthExt{0, 0};
    std::vector<Fb> m_fbs;
};

}  // namespace arcadexr::vulkan
