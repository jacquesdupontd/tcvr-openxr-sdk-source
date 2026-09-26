// Copyright (c) 2017-2026 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.h"
#include "common.h"
#include "geometry.h"
#include "graphicsplugin.h"
#include "graphics_plugin_impl_helpers.h"
#include <nonstd/span.hpp>
#include "check.h"

#ifdef XR_USE_GRAPHICS_API_VULKAN
#include <common/vulkan_debug_object_namer.hpp>
#include <common/xr_linear.h>
#include "vulkan_utils.h"

#ifdef USE_ONLINE_VULKAN_SHADERC
#include <shaderc/shaderc.hpp>
#endif

#if defined(VK_USE_PLATFORM_WIN32_KHR)
// Define USE_MIRROR_WINDOW to open a otherwise-unused window for e.g. RenderDoc
#define USE_MIRROR_WINDOW
#endif

// glslangValidator doesn't wrap its output in brackets if you don't have it define the whole array.
#if defined(USE_GLSLANGVALIDATOR)
#define SPV_PREFIX {
#define SPV_SUFFIX }
#else
#define SPV_PREFIX
#define SPV_SUFFIX
#endif

#include "framebuffer_bridge.h"
#include "virtual_screen.h"
#include "scene_bridge.h"
#include "settings.h"
#include "game_profile.h"
#include <new>
#include "m2_pipeline_types.h"
#include "vulkan_m2_renderer.h"
#include "vulkan_overlay.h"
#include "vulkan_s22_renderer.h"
#include "menu.h"
#include "aim_state.h"
#include "display_refresh.h"
#include <chrono>
#include <map>
#include <algorithm>
#include <algorithm>
#include <cmath>

struct ScreenVertex {
    XrVector3f Position;
    XrVector2f TexCoord;
};

static constexpr ScreenVertex c_screenVertices[] = {
    {{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f}},
    {{ 0.5f, -0.5f, 0.0f}, {1.0f, 1.0f}},
    {{ 0.5f,  0.5f, 0.0f}, {1.0f, 0.0f}},
    {{-0.5f,  0.5f, 0.0f}, {0.0f, 0.0f}},
};

static constexpr uint16_t c_screenIndices[] = {0, 1, 2, 0, 2, 3};

// Push constants of vulkan_shaders_ov/screen_*.glsl (std430 push block layout).
struct ScreenPC {
    float mvp[16];
    float aim[2];
    float srcSize[2];
    int32_t aimVisible, calibrating, filter;
    float sharpen;
};
static_assert(sizeof(ScreenPC) == 96, "ScreenPC layout");
#include "ov_screen_vert_spv.h"
#include "ov_screen_frag_spv.h"

namespace {

using nonstd::span;

#ifdef USE_ONLINE_VULKAN_SHADERC
constexpr char VertexShaderGlsl[] =
    R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (std140, push_constant) uniform buf
    {
        mat4 mvp;
    } ubuf;

    layout (location = 0) in vec3 Position;
    layout (location = 1) in vec3 Color;

    layout (location = 0) out vec4 oColor;
    out gl_PerVertex
    {
        vec4 gl_Position;
    };

    void main()
    {
        oColor.rgba  = Color.rgba;
        gl_Position = ubuf.mvp * Position;
    }
)_";

constexpr char FragmentShaderGlsl[] =
    R"_(
    #version 430
    #extension GL_ARB_separate_shader_objects : enable

    layout (location = 0) in vec4 oColor;

    layout (location = 0) out vec4 FragColor;

    void main()
    {
        FragColor = oColor;
    }
)_";
#endif  // USE_ONLINE_VULKAN_SHADERC

#if defined(USE_MIRROR_WINDOW)
// Swapchain
struct Swapchain {
    VkFormat format{VK_FORMAT_B8G8R8A8_SRGB};
    VkSurfaceKHR surface{VK_NULL_HANDLE};
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    VkFence readyFence{VK_NULL_HANDLE};
    VkFence presentFence{VK_NULL_HANDLE};
    static const uint32_t maxImages = 4;
    uint32_t swapchainCount = 0;
    uint32_t renderImageIdx = 0;
    VkImage image[maxImages]{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

    Swapchain() {}
    ~Swapchain() { Release(); }

    void Create(VkInstance instance, VkPhysicalDevice physDevice, VkDevice device, uint32_t queueFamilyIndex);
    void Prepare(VkCommandBuffer buf);
    void Wait();
    void Acquire(VkSemaphore readySemaphore = VK_NULL_HANDLE);
    void Present(VkQueue queue, VkSemaphore drawComplete = VK_NULL_HANDLE);
    void Release() {
        if (m_vkDevice) {
            // Flush any pending Present() calls which are using the fence
            Wait();
            if (swapchain) vkDestroySwapchainKHR(m_vkDevice, swapchain, nullptr);
            if (readyFence) vkDestroyFence(m_vkDevice, readyFence, nullptr);
        }

        if (m_vkInstance && surface) vkDestroySurfaceKHR(m_vkInstance, surface, nullptr);

        readyFence = VK_NULL_HANDLE;
        presentFence = VK_NULL_HANDLE;
        swapchain = VK_NULL_HANDLE;
        surface = VK_NULL_HANDLE;
        for (uint32_t i = 0; i < swapchainCount; ++i) {
            image[i] = VK_NULL_HANDLE;
        }
        swapchainCount = 0;

#if defined(VK_USE_PLATFORM_WIN32_KHR)
        if (hWnd) {
            DestroyWindow(hWnd);
            hWnd = nullptr;
            UnregisterClassW(L"hello_xr", hInst);
        }
        if (hUser32Dll != NULL) {
            ::FreeLibrary(hUser32Dll);
            hUser32Dll = NULL;
        }
#endif

        m_vkDevice = nullptr;
    }
    void Recreate() {
        Release();
        Create(m_vkInstance, m_vkPhysicalDevice, m_vkDevice, m_queueFamilyIndex);
    }

   private:
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    HINSTANCE hInst{NULL};
    HWND hWnd{NULL};
    HINSTANCE hUser32Dll{NULL};
#endif
    const VkExtent2D size{640, 480};
    VkInstance m_vkInstance{VK_NULL_HANDLE};
    VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
    VkDevice m_vkDevice{VK_NULL_HANDLE};
    uint32_t m_queueFamilyIndex = 0;
};

void Swapchain::Create(VkInstance instance, VkPhysicalDevice physDevice, VkDevice device, uint32_t queueFamilyIndex) {
    m_vkInstance = instance;
    m_vkPhysicalDevice = physDevice;
    m_vkDevice = device;
    m_queueFamilyIndex = queueFamilyIndex;

// Create a WSI surface for the window:
#if defined(VK_USE_PLATFORM_WIN32_KHR)
    hInst = GetModuleHandle(NULL);

    WNDCLASSW wc{};
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = DefWindowProcW;
    wc.cbWndExtra = sizeof(this);
    wc.hInstance = hInst;
    wc.lpszClassName = L"hello_xr";
    RegisterClassW(&wc);

// adjust the window size and show at InitDevice time
#if defined(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    typedef DPI_AWARENESS_CONTEXT(WINAPI * PFN_SetThreadDpiAwarenessContext)(DPI_AWARENESS_CONTEXT);
    hUser32Dll = ::LoadLibraryA("user32.dll");
    if (PFN_SetThreadDpiAwarenessContext SetThreadDpiAwarenessContextFn =
            reinterpret_cast<PFN_SetThreadDpiAwarenessContext>(::GetProcAddress(hUser32Dll, "SetThreadDpiAwarenessContext"))) {
        // Make sure we're 1:1 for HMD pixels
        SetThreadDpiAwarenessContextFn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
#endif
    RECT rect{0, 0, (LONG)size.width, (LONG)size.height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, false);
    hWnd = CreateWindowW(wc.lpszClassName, L"hello_xr (Vulkan)", WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
                         rect.right - rect.left, rect.bottom - rect.top, 0, 0, hInst, 0);
    assert(hWnd != NULL);

    SetWindowLongPtr(hWnd, 0, LONG_PTR(this));

    VkWin32SurfaceCreateInfoKHR surfCreateInfo{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    surfCreateInfo.flags = 0;
    surfCreateInfo.hinstance = hInst;
    surfCreateInfo.hwnd = hWnd;
    XRC_CHECK_THROW_VKCMD(vkCreateWin32SurfaceKHR(m_vkInstance, &surfCreateInfo, nullptr, &surface));
#else
#error CreateSurface not supported on this OS
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)

    VkSurfaceCapabilitiesKHR surfCaps;
    XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_vkPhysicalDevice, surface, &surfCaps));
    CHECK(surfCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    uint32_t surfFmtCount = 0;
    XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, surface, &surfFmtCount, nullptr));
    std::vector<VkSurfaceFormatKHR> surfFmts(surfFmtCount);
    XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceFormatsKHR(m_vkPhysicalDevice, surface, &surfFmtCount, &surfFmts[0]));
    uint32_t foundFmt;
    for (foundFmt = 0; foundFmt < surfFmtCount; ++foundFmt) {
        if (surfFmts[foundFmt].format == format) break;
    }

    CHECK(foundFmt < surfFmtCount);

    uint32_t presentModeCount = 0;
    XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, surface, &presentModeCount, nullptr));
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    XRC_CHECK_THROW_VKCMD(
        vkGetPhysicalDeviceSurfacePresentModesKHR(m_vkPhysicalDevice, surface, &presentModeCount, &presentModes[0]));

    // Do not use VSYNC for the mirror window, but Nvidia doesn't support IMMEDIATE so fall back to MAILBOX
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    for (uint32_t i = 0; i < presentModeCount; ++i) {
        if ((presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) || (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)) {
            presentMode = presentModes[i];
            break;
        }
    }

    VkBool32 presentable = false;
    XRC_CHECK_THROW_VKCMD(vkGetPhysicalDeviceSurfaceSupportKHR(m_vkPhysicalDevice, m_queueFamilyIndex, surface, &presentable));
    CHECK(presentable);

    VkSwapchainCreateInfoKHR swapchainInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchainInfo.flags = 0;
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = surfCaps.minImageCount;
    swapchainInfo.imageFormat = format;
    swapchainInfo.imageColorSpace = surfFmts[foundFmt].colorSpace;
    swapchainInfo.imageExtent = size;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainInfo.queueFamilyIndexCount = 0;
    swapchainInfo.pQueueFamilyIndices = nullptr;
    swapchainInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = presentMode;
    swapchainInfo.clipped = true;
    swapchainInfo.oldSwapchain = VK_NULL_HANDLE;
    XRC_CHECK_THROW_VKCMD(vkCreateSwapchainKHR(m_vkDevice, &swapchainInfo, nullptr, &swapchain));

    // Fence to throttle host on Acquire
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    XRC_CHECK_THROW_VKCMD(vkCreateFence(m_vkDevice, &fenceInfo, nullptr, &readyFence));

    swapchainCount = 0;
    XRC_CHECK_THROW_VKCMD(vkGetSwapchainImagesKHR(m_vkDevice, swapchain, &swapchainCount, nullptr));
    assert(swapchainCount < maxImages);
    XRC_CHECK_THROW_VKCMD(vkGetSwapchainImagesKHR(m_vkDevice, swapchain, &swapchainCount, image));
    if (swapchainCount > maxImages) {
        Log::Write(Log::Level::Info,
                   "Reducing swapchain length from " + std::to_string(swapchainCount) + " to " + std::to_string(maxImages));
        swapchainCount = maxImages;
    }

    Log::Write(Log::Level::Info, "Swapchain length " + std::to_string(swapchainCount));
}

void Swapchain::Prepare(VkCommandBuffer buf) {
    // Convert swapchain images to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    for (uint32_t i = 0; i < swapchainCount; ++i) {
        VkImageMemoryBarrier imgBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        imgBarrier.srcAccessMask = 0;  // XXX was VK_ACCESS_TRANSFER_READ_BIT wrong?
        imgBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imgBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imgBarrier.image = image[i];
        imgBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &imgBarrier);
    }
}

void Swapchain::Wait() {
    if (presentFence) {
        // Wait for the fence...
        XRC_CHECK_THROW_VKCMD(vkWaitForFences(m_vkDevice, 1, &presentFence, VK_TRUE, UINT64_MAX));
        // ...then reset the fence for future Acquire calls
        XRC_CHECK_THROW_VKCMD(vkResetFences(m_vkDevice, 1, &presentFence));
        presentFence = VK_NULL_HANDLE;
    }
}

void Swapchain::Acquire(VkSemaphore readySemaphore) {
    // If we're not using a semaphore to rate-limit the GPU, rate limit the host with a fence instead
    if (readySemaphore == VK_NULL_HANDLE) {
        Wait();
        presentFence = readyFence;
    }

    XRC_CHECK_THROW_VKCMD(vkAcquireNextImageKHR(m_vkDevice, swapchain, UINT64_MAX, readySemaphore, presentFence, &renderImageIdx));
}

void Swapchain::Present(VkQueue queue, VkSemaphore drawComplete) {
    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    if (drawComplete) {
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &drawComplete;
    }
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &renderImageIdx;
    auto res = vkQueuePresentKHR(queue, &presentInfo);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        Recreate();
        return;
    }
    XRC_CHECK_THROW_VKRESULT(res, "vkQueuePresentKHR");
}
#endif  // defined(USE_MIRROR_WINDOW)

struct VulkanArraySliceState {
    VulkanArraySliceState() = default;
    VulkanArraySliceState(const VulkanArraySliceState&) = delete;
    std::vector<RenderTarget> m_renderTarget;  // per swapchain index
    RenderPass m_rp{};
    Pipeline m_pipe{};
    Pipeline m_pipeCompute{};

    void init(const VulkanDebugObjectNamer& namer, VkDevice device, uint32_t capacity, const VkExtent2D size, VkFormat colorFormat,
              VkFormat depthFormat, VkSampleCountFlagBits sampleCount, const PipelineLayout& layout,
              const PipelineLayout& computeLayout, const ShaderProgram& sp, const ShaderProgram& spCompute,
              const VkVertexInputBindingDescription& bindDesc, span<const VkVertexInputAttributeDescription> attrDesc) {
        m_renderTarget.resize(capacity);
        m_rp.Create(namer, device, colorFormat, depthFormat, sampleCount);
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_VIEWPORT};
        m_pipe.Create(device, size, layout, m_rp, sp, bindDesc, attrDesc, dynamicStates);

        // m_pipeCompute not created because hello_xr doesn't need compute shaders
        (void)computeLayout;
        (void)spCompute;
    }

    void Reset() {
        m_pipe.Reset();
        m_pipeCompute.Reset();
        m_rp.Reset();
        m_renderTarget.clear();
    }
};

/// Vulkan data used per swapchain. One per XrSwapchain handle.
class VulkanSwapchainImageData : public SwapchainImageDataBase<XrSwapchainImageVulkanKHR> {
    void init(uint32_t capacity, VkFormat colorFormat, const PipelineLayout& layout, const PipelineLayout& computeLayout,
              const ShaderProgram& sp, const ShaderProgram& spCompute, const VkVertexInputBindingDescription& bindDesc,
              span<const VkVertexInputAttributeDescription> attrDesc) {
        m_depthBuffer.resize(capacity);
        for (auto& slice : m_slices) {
            slice.init(m_namer, m_vkDevice, capacity, m_size, colorFormat, m_depthFormat, m_sampleCount, layout, computeLayout, sp,
                       spCompute, bindDesc, attrDesc);
        }
    }

   public:
    VulkanSwapchainImageData(const VulkanDebugObjectNamer& namer, uint32_t capacity,
                             const XrSwapchainCreateInfo& swapchainCreateInfo, VkDevice device, MemoryAllocator* memAllocator,
                             const PipelineLayout& layout, const PipelineLayout& computeLayout, const ShaderProgram& sp,
                             const ShaderProgram& spCompute, const VkVertexInputBindingDescription& bindDesc,
                             span<const VkVertexInputAttributeDescription> attrDesc)
        : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, capacity, swapchainCreateInfo),
          m_namer(namer),
          m_vkDevice(device),
          m_memAllocator(memAllocator),
          m_size{swapchainCreateInfo.width, swapchainCreateInfo.height},
          m_sampleCount{(VkSampleCountFlagBits)swapchainCreateInfo.sampleCount},
          m_slices(swapchainCreateInfo.arraySize) {
        init(capacity, (VkFormat)swapchainCreateInfo.format, layout, computeLayout, sp, spCompute, bindDesc, attrDesc);
    }

    VulkanSwapchainImageData(const VulkanDebugObjectNamer& namer, uint32_t capacity,
                             const XrSwapchainCreateInfo& swapchainCreateInfo, XrSwapchain depthSwapchain,
                             const XrSwapchainCreateInfo& depthSwapchainCreateInfo, VkDevice device, MemoryAllocator* memAllocator,
                             const PipelineLayout& layout, const PipelineLayout& computeLayout, const ShaderProgram& sp,
                             const ShaderProgram& spCompute, const VkVertexInputBindingDescription& bindDesc,
                             span<const VkVertexInputAttributeDescription> attrDesc)
        : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, capacity, swapchainCreateInfo, depthSwapchain,
                                 depthSwapchainCreateInfo),
          m_namer(namer),
          m_vkDevice(device),
          m_memAllocator(memAllocator),
          m_size{swapchainCreateInfo.width, swapchainCreateInfo.height},
          m_sampleCount{(VkSampleCountFlagBits)swapchainCreateInfo.sampleCount},
          m_depthFormat((VkFormat)depthSwapchainCreateInfo.format),
          m_slices(swapchainCreateInfo.arraySize) {
        init(capacity, (VkFormat)swapchainCreateInfo.format, layout, computeLayout, sp, spCompute, bindDesc, attrDesc);
    }

    ~VulkanSwapchainImageData() override {
        // Calling a virtual function from a destructor doesn't work the way you'd expect, so we do this here.
        VulkanSwapchainImageData::Reset();
    }

    void BindRenderTarget(uint32_t index, uint32_t arraySlice, const VkRect2D& renderArea,
                          VkImageAspectFlags secondAttachmentAspect, VkRenderPassBeginInfo* renderPassBeginInfo) {
        RenderTarget& rt = m_slices[arraySlice].m_renderTarget[index];
        RenderPass& rp = m_slices[arraySlice].m_rp;
        if (rt.fb == VK_NULL_HANDLE) {
            rt.Create(m_namer, m_vkDevice, GetTypedImage(index).image, GetDepthImageForColorIndex(index).image,
                      secondAttachmentAspect, arraySlice, m_size, rp);
        }
        renderPassBeginInfo->renderPass = rp.pass;
        renderPassBeginInfo->framebuffer = rt.fb;
        renderPassBeginInfo->renderArea = renderArea;
    }

    void BindPipeline(VkCommandBuffer buf, uint32_t arraySlice, enum ShaderProgramType programType = SHADER_PROGRAM_TYPE_GRAPHICS) {
        switch (programType) {
            case SHADER_PROGRAM_TYPE_GRAPHICS:
                vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_slices[arraySlice].m_pipe.pipe);
                break;
            case SHADER_PROGRAM_TYPE_COMPUTE:
                vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_slices[arraySlice].m_pipeCompute.pipe);
                break;
            default:
                Throw("unknown programType");
        }
    }

    void TransitionLayout(uint32_t imageIndex, CmdBuffer* cmdBuffer, VkImageLayout newLayout) {
        m_depthBuffer[imageIndex].TransitionLayout(cmdBuffer, newLayout);
    }

    void Reset() override {
        for (auto& slice : m_slices) {
            slice.Reset();
        }
        m_depthBuffer.clear();
        SwapchainImageDataBase::Reset();
    }

    int64_t GetDepthFormat() const { return m_depthFormat; }

    const std::vector<VulkanArraySliceState>& GetSlices() const { return m_slices; }

   protected:
    const XrSwapchainImageVulkanKHR& GetFallbackDepthSwapchainImage(uint32_t i) override {
        if (!m_depthBuffer[i].Allocated()) {
            m_depthBuffer[i].Allocate(m_namer, m_vkDevice, m_memAllocator, m_depthFormat, this->Width(), this->Height(),
                                      this->ArraySize(), this->SampleCount());
        }

        return m_depthBuffer[i].GetTexture();
    }

   private:
    VulkanDebugObjectNamer m_namer;
    VkDevice m_vkDevice{VK_NULL_HANDLE};
    MemoryAllocator* m_memAllocator{nullptr};
    VkExtent2D m_size{};
    VkSampleCountFlagBits m_sampleCount;
    std::vector<DepthBuffer> m_depthBuffer;  // per swapchain index
    VkFormat m_depthFormat{VK_FORMAT_D32_SFLOAT};

    std::vector<VulkanArraySliceState> m_slices;
};

struct VulkanGraphicsPlugin : public IGraphicsPlugin {
    VulkanGraphicsPlugin() { m_graphicsBinding.type = GetGraphicsBindingType(); };

    std::vector<std::string> GetInstanceExtensions() const override { return {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME}; }

    // Note: The output must not outlive the input - this modifies the input and returns a collection of views into that modified
    // input!
    std::vector<const char*> ParseExtensionString(char* names) {
        std::vector<const char*> list;
        while (*names != 0) {
            list.push_back(names);
            while (*(++names) != 0) {
                if (*names == ' ') {
                    *names++ = '\0';
                    break;
                }
            }
        }
        return list;
    }

    const char* GetValidationLayerName() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        std::vector<const char*> validationLayerNames;
        validationLayerNames.push_back("VK_LAYER_KHRONOS_validation");
        validationLayerNames.push_back("VK_LAYER_LUNARG_standard_validation");

        // Enable only one validation layer from the list above. Prefer KHRONOS.
        for (auto& validationLayerName : validationLayerNames) {
            for (const auto& layerProperties : availableLayers) {
                if (0 == strcmp(validationLayerName, layerProperties.layerName)) {
                    return validationLayerName;
                }
            }
        }

        return nullptr;
    }

    void InitializeDevice(XrInstance instance, XrSystemId systemId) override {
        // Create the Vulkan device for the adapter associated with the system.
        // Extension function must be loaded by name
        XrGraphicsRequirementsVulkan2KHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
        CHECK_XRCMD(GetVulkanGraphicsRequirements2KHR(instance, systemId, &graphicsRequirements));

        VkResult err;

        std::vector<const char*> layers;
#if !defined(NDEBUG)
        const char* const validationLayerName = GetValidationLayerName();
        if (validationLayerName) {
            layers.push_back(validationLayerName);
        } else {
            Log::Write(Log::Level::Warning, "No validation layers found in the system, skipping");
        }
#endif

        std::vector<const char*> extensions;
        {
            uint32_t extensionCount = 0;
            XRC_CHECK_THROW_VKCMD(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr));

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            XRC_CHECK_THROW_VKCMD(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()));
            const auto b = availableExtensions.begin();
            const auto e = availableExtensions.end();

            auto isExtSupported = [&](const char* extName) -> bool {
                auto it = std::find_if(b, e, [&](const VkExtensionProperties& properties) {
                    return (0 == strcmp(extName, properties.extensionName));
                });
                return (it != e);
            };

            // Debug utils is optional and not always available
            if (isExtSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
            // TODO add back VK_EXT_debug_report code for compatibility with older systems? (Android)
        }
#if defined(USE_MIRROR_WINDOW)
        extensions.push_back("VK_KHR_surface");
#if defined(VK_USE_PLATFORM_WIN32_KHR)
        extensions.push_back("VK_KHR_win32_surface");
#else
#error CreateSurface not supported on this OS
#endif  // defined(VK_USE_PLATFORM_WIN32_KHR)
#endif  // defined(USE_MIRROR_WINDOW)

        VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
#if !defined(NDEBUG)
        debugInfo.messageSeverity |= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
#endif
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = debugMessageThunk;
        debugInfo.pUserData = this;

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = "hello_xr";
        appInfo.applicationVersion = 1;
        appInfo.pEngineName = "hello_xr";
        appInfo.engineVersion = 1;
        appInfo.apiVersion = VK_API_VERSION_1_1;  // vkGetPhysicalDeviceFeatures2 (descriptor indexing probe)

        VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instInfo.pNext = &debugInfo;
        instInfo.pApplicationInfo = &appInfo;
        instInfo.enabledLayerCount = (uint32_t)layers.size();
        instInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        instInfo.enabledExtensionCount = (uint32_t)extensions.size();
        instInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

        XrVulkanInstanceCreateInfoKHR createInfo{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
        createInfo.systemId = systemId;
        createInfo.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        createInfo.vulkanCreateInfo = &instInfo;
        createInfo.vulkanAllocator = nullptr;
        CHECK_XRCMD(CreateVulkanInstanceKHR(instance, &createInfo, &m_vkInstance, &err));
        XRC_CHECK_THROW_VKCMD(err);

        vkCreateDebugUtilsMessengerEXT =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_vkInstance, "vkCreateDebugUtilsMessengerEXT");

        if (vkCreateDebugUtilsMessengerEXT != nullptr) {
            XRC_CHECK_THROW_VKCMD(vkCreateDebugUtilsMessengerEXT(m_vkInstance, &debugInfo, nullptr, &m_vkDebugUtilsMessenger));
        }

        XrVulkanGraphicsDeviceGetInfoKHR deviceGetInfo{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        deviceGetInfo.systemId = systemId;
        deviceGetInfo.vulkanInstance = m_vkInstance;
        CHECK_XRCMD(GetVulkanGraphicsDevice2KHR(instance, &deviceGetInfo, &m_vkPhysicalDevice));

        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        float queuePriorities = 0;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriorities;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilyProps(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, &queueFamilyProps[0]);

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            // Only need graphics (not presentation) for draw queue
            if ((queueFamilyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0u) {
                m_queueFamilyIndex = queueInfo.queueFamilyIndex = i;
                break;
            }
        }

        std::vector<const char*> deviceExtensions;

        VkPhysicalDeviceFeatures features{};
        // Model 2 renderer: every texture region becomes its own GPU image, all indexed from one
        // descriptor array by a per-polygon (non-uniform) index -> VK_EXT_descriptor_indexing.
        // Hardware anisotropy for those images -> samplerAnisotropy.
        VkPhysicalDeviceDescriptorIndexingFeaturesEXT indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
        {
            uint32_t n = 0;
            vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> exts(n);
            vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &n, exts.data());
            bool haveIndexing = false, haveFdm = false, haveRp2 = false, haveDsResolve = false;
            for (auto& e : exts) {
                if (strcmp(e.extensionName, "VK_KHR_create_renderpass2") == 0) haveRp2 = true;
                if (strcmp(e.extensionName, "VK_KHR_depth_stencil_resolve") == 0) haveDsResolve = true;
                if (strcmp(e.extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) == 0) haveIndexing = true;
                if (strcmp(e.extensionName, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME) == 0) haveFdm = true;
            }
            VkPhysicalDeviceFragmentDensityMapFeaturesEXT qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT};
            VkPhysicalDeviceDescriptorIndexingFeaturesEXT q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &q;
            q.pNext = &qf;
            // Android API 26 libvulkan does not export the 1.1 symbol: fetch it from the instance.
            auto getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceFeatures2"));
            if (!getFeatures2)
                getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                    vkGetInstanceProcAddr(m_vkInstance, "vkGetPhysicalDeviceFeatures2KHR"));
            if (getFeatures2) getFeatures2(m_vkPhysicalDevice, &f2);
            else vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice, &f2.features);
            features.samplerAnisotropy = f2.features.samplerAnisotropy;
            features.fragmentStoresAndAtomics = f2.features.fragmentStoresAndAtomics;  // debug overdraw counters
            features.shaderClipDistance = f2.features.shaderClipDistance;  // secondary views cut at their window (m2_vert)
            Log::Write(Log::Level::Info, Fmt("TCVR_VKFEAT shaderClipDistance=%d", int(f2.features.shaderClipDistance)));
            if (haveIndexing && q.shaderSampledImageArrayNonUniformIndexing) {
                deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
                indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
                arcadexr::vulkan::g_m2DescriptorIndexing = true;
            }
            arcadexr::vulkan::g_m2SamplerAnisotropy = f2.features.samplerAnisotropy == VK_TRUE;
            // Fixed foveation (fragment density map), used by the Model 2 immersive render pass.
            if (haveFdm && qf.fragmentDensityMap) {
                deviceExtensions.push_back(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
                m_fdmFeature.fragmentDensityMap = VK_TRUE;
                m_fdmEnabled = true;
            }
            // AppSW needs the depth in the headset's depth image: the Model 2 pass resolves its MSAA depth into it (26/09).
            if (haveRp2 && haveDsResolve) {
                deviceExtensions.push_back("VK_KHR_create_renderpass2");
                deviceExtensions.push_back("VK_KHR_depth_stencil_resolve");
                m_haveDepthResolve = true;
            }
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK device: fragment density map ext=%d feature=%d | depth resolve rp2=%d dsr=%d",
                                             int(haveFdm), int(qf.fragmentDensityMap), int(haveRp2), int(haveDsResolve)));
            Log::Write(Log::Level::Info, Fmt("TCVR_M2VK device: descriptor indexing ext=%d nonUniformSampled=%d samplerAnisotropy=%d",
                                             int(haveIndexing), int(q.shaderSampledImageArrayNonUniformIndexing),
                                             int(f2.features.samplerAnisotropy)));
        }

#if defined(USE_MIRROR_WINDOW)
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#endif

        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledLayerCount = 0;
        deviceInfo.ppEnabledLayerNames = nullptr;
        deviceInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
        deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
        deviceInfo.pEnabledFeatures = &features;
        {   // pNext chain: descriptor indexing, fragment density map
            const void* chain = nullptr;
            if (m_fdmEnabled) { m_fdmFeature.pNext = const_cast<void*>(chain); chain = &m_fdmFeature; }
            if (arcadexr::vulkan::g_m2DescriptorIndexing) { indexing.pNext = const_cast<void*>(chain); chain = &indexing; }
            deviceInfo.pNext = chain;
        }

        XrVulkanDeviceCreateInfoKHR deviceCreateInfo{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        deviceCreateInfo.systemId = systemId;
        deviceCreateInfo.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
        deviceCreateInfo.vulkanCreateInfo = &deviceInfo;
        deviceCreateInfo.vulkanPhysicalDevice = m_vkPhysicalDevice;
        deviceCreateInfo.vulkanAllocator = nullptr;
        CHECK_XRCMD(CreateVulkanDeviceKHR(instance, &deviceCreateInfo, &m_vkDevice, &err));
        XRC_CHECK_THROW_VKCMD(err);

        m_namer.Init(m_vkInstance, m_vkDevice);

        vkGetDeviceQueue(m_vkDevice, queueInfo.queueFamilyIndex, 0, &m_vkQueue);

        m_memAllocator.Init(m_vkPhysicalDevice, m_vkDevice);

        InitializeResources();

        m_graphicsBinding.instance = m_vkInstance;
        m_graphicsBinding.physicalDevice = m_vkPhysicalDevice;
        m_graphicsBinding.device = m_vkDevice;
        m_graphicsBinding.queueFamilyIndex = queueInfo.queueFamilyIndex;
        m_graphicsBinding.queueIndex = 0;
    }

#ifdef USE_ONLINE_VULKAN_SHADERC
    // Compile a shader to a SPIR-V binary.
    std::vector<uint32_t> CompileGlslShader(const std::string& name, shaderc_shader_kind kind, const std::string& source) {
        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        options.SetOptimizationLevel(shaderc_optimization_level_size);

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, kind, name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
            Log::Write(Log::Level::Error, Fmt("Shader %s compilation failed: %s", name.c_str(), module.GetErrorMessage().c_str()));
            return std::vector<uint32_t>();
        }

        return {module.cbegin(), module.cend()};
    }
#endif

    void InitializeResources() {
#ifdef USE_ONLINE_VULKAN_SHADERC
        auto vertexSPIRV = CompileGlslShader("vertex", shaderc_glsl_default_vertex_shader, VertexShaderGlsl);
        auto fragmentSPIRV = CompileGlslShader("fragment", shaderc_glsl_default_fragment_shader, FragmentShaderGlsl);
#else
        std::vector<uint32_t> vertexSPIRV = SPV_PREFIX
#include "vert.spv"
            SPV_SUFFIX;
        std::vector<uint32_t> fragmentSPIRV = SPV_PREFIX
#include "frag.spv"
            SPV_SUFFIX;
#endif
        if (vertexSPIRV.empty()) THROW("Failed to compile vertex shader");
        if (fragmentSPIRV.empty()) THROW("Failed to compile fragment shader");

        m_shaderProgram.Init(m_vkDevice);
        m_shaderProgram.LoadVertexShader(vertexSPIRV);
        m_shaderProgram.LoadFragmentShader(fragmentSPIRV);

        // Semaphore to block on draw complete
        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        XRC_CHECK_THROW_VKCMD(vkCreateSemaphore(m_vkDevice, &semInfo, nullptr, &m_vkDrawDone));
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_vkDrawDone, "hello_xr draw done semaphore"));

        for (int i = 0; i < 4; ++i) {
            if (!m_cmdBuffer[i].Init(m_namer, m_vkDevice, m_queueFamilyIndex)) THROW("Failed to create command buffer");
        }

        m_pipelineLayout.Create(m_vkDevice);

        // hello_xr: doesn't need compute shader support
#if 0
        XRC_CHECK_THROW_VKCMD(
            m_namer.SetName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)m_pipelineLayout.layout, "hello_xr graphics pipeline layout"));

        m_computePipelineLayout.Create(m_vkDevice, SHADER_PROGRAM_TYPE_COMPUTE);
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)m_computePipelineLayout.layout,
                                    "hello_xr compute pipeline layout"));
        XRC_CHECK_THROW_VKCMD(m_namer.SetName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)m_computePipelineLayout.descriptorSetLayout,
                                    "hello_xr compute descriptor set layout"));

        m_computeDescriptorPool.adopt(CreateDescriptorPool(m_vkDevice, 1, 1), m_vkDevice);

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = m_computeDescriptorPool.get();
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_computePipelineLayout.descriptorSetLayout;
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_ComputeDescriptorSet));
#endif

        static_assert(sizeof(Geometry::Vertex) == 24, "Unexpected Vertex size");
        m_drawBuffer.Init(m_vkDevice, &m_memAllocator,
                          {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Geometry::Vertex, Position)},
                           {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Geometry::Vertex, Color)}});
        uint32_t numCubeIdicies = sizeof(Geometry::c_cubeIndices) / sizeof(Geometry::c_cubeIndices[0]);
        uint32_t numCubeVerticies = sizeof(Geometry::c_cubeVertices) / sizeof(Geometry::c_cubeVertices[0]);
        m_drawBuffer.Create(numCubeIdicies, numCubeVerticies);

        m_drawBuffer.UpdateIndices(span<const uint16_t>(Geometry::c_cubeIndices, numCubeIdicies), 0);
        m_drawBuffer.UpdateVertices(span<const Geometry::Vertex>(Geometry::c_cubeVertices, numCubeVerticies), 0);

        // Screen shader program & pipeline layout
        // Screen shaders: port of the GLES screen path (Catmull-Rom, crosshair, calibration), vulkan_shaders_ov/.
        std::vector<uint32_t> screenVertexSPIRV(sizeof(c_ovScreenVertSpv) / 4), screenFragmentSPIRV(sizeof(c_ovScreenFragSpv) / 4);
        std::memcpy(screenVertexSPIRV.data(), c_ovScreenVertSpv, sizeof(c_ovScreenVertSpv));
        std::memcpy(screenFragmentSPIRV.data(), c_ovScreenFragSpv, sizeof(c_ovScreenFragSpv));
        m_screenShaderProgram.Init(m_vkDevice);
        m_screenShaderProgram.LoadVertexShader(screenVertexSPIRV);
        m_screenShaderProgram.LoadFragmentShader(screenFragmentSPIRV);

        VkDescriptorSetLayoutBinding screenBinding{};
        screenBinding.binding = 0;
        screenBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        screenBinding.descriptorCount = 1;
        screenBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo descLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descLayoutInfo.bindingCount = 1;
        descLayoutInfo.pBindings = &screenBinding;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorSetLayout(m_vkDevice, &descLayoutInfo, nullptr, &m_screenDescriptorSetLayout));

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(ScreenPC);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &m_screenDescriptorSetLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pcr;
        XRC_CHECK_THROW_VKCMD(vkCreatePipelineLayout(m_vkDevice, &pipeLayoutInfo, nullptr, &m_screenPipelineLayout));

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 2;   // [0] MAME framebuffer, [1] the GPU-drawn flat Model 2 image

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        XRC_CHECK_THROW_VKCMD(vkCreateDescriptorPool(m_vkDevice, &poolInfo, nullptr, &m_screenDescriptorPool));

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = m_screenDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_screenDescriptorSetLayout;
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_screenDescriptorSet));
        XRC_CHECK_THROW_VKCMD(vkAllocateDescriptorSets(m_vkDevice, &allocInfo, &m_flatDescriptorSet));
        {   // trilinear: the 4x flat image is minified on the virtual screen
            VkSamplerCreateInfo fs{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            fs.magFilter = fs.minFilter = VK_FILTER_LINEAR; fs.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            fs.addressModeU = fs.addressModeV = fs.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            fs.maxLod = 16.0f;
            XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_vkDevice, &fs, nullptr, &m_flatSampler));
        }

        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        XRC_CHECK_THROW_VKCMD(vkCreateSampler(m_vkDevice, &samplerInfo, nullptr, &m_screenSampler));

        m_screenDrawBuffer.Init(m_vkDevice, &m_memAllocator,
                                {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ScreenVertex, Position)},
                                 {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ScreenVertex, TexCoord)}});
        m_screenDrawBuffer.Create(6, 4);
        m_screenDrawBuffer.UpdateIndices(span<const uint16_t>(c_screenIndices, 6), 0);
        m_screenDrawBuffer.UpdateVertices(span<const ScreenVertex>(c_screenVertices, 4), 0);
#if defined(USE_MIRROR_WINDOW)
        m_swapchain.Create(m_vkInstance, m_vkPhysicalDevice, m_vkDevice, m_graphicsBinding.queueFamilyIndex);

        m_cmdBuffer[0].Reset();
        if (!m_cmdBuffer[0].Init(m_namer, m_vkDevice, m_queueFamilyIndex)) THROW("Failed to create command buffer");

        m_cmdBuffer[0].Begin();
        m_swapchain.Prepare(m_cmdBuffer[0].buf);
        m_cmdBuffer[0].End();
        m_cmdBuffer[0].Exec(m_vkQueue);
        m_cmdBuffer[0].Wait();
#endif
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t SelectColorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override {
        // List of supported color swapchain formats.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_FORMAT_B8G8R8A8_SRGB,
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_FORMAT_B8G8R8A8_UNORM,
            });
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t SelectDepthSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override {
        // AppSW (26/09): the Model 2 pass resolves its D32F depth into the headset's depth image, which needs the same
        // format -- the runtime lists D16 first. Only when AppSW is requested: otherwise the runtime's order, as before.
        if (arcadexr::config::GetInt("appsw", 0) != 0)
            for (int64_t f : imageFormatArray)
                if (f == int64_t(VK_FORMAT_D32_SFLOAT)) return f;
        // List of supported depth swapchain formats.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D24_UNORM_S8_UINT,
                VK_FORMAT_D16_UNORM,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
            });
    }

    const XrBaseInStructure* GetGraphicsBinding() const override {
        return reinterpret_cast<const XrBaseInStructure*>(&m_graphicsBinding);
    }

    // Arcade colours are display values (CRT-referred, like sRGB-encoded bytes). Rendered through an _SRGB view
    // they get encoded AGAIN: lighter, washed out (measured by Guillaume's eyes 23/09; the GL build writes raw
    // with GL_FRAMEBUFFER_SRGB off and looks right, and Sega Rally had a contrast 1.2 compensating it). The
    // swapchain stays _SRGB for the compositor (created MUTABLE_FORMAT); every view and render pass we make on
    // it is the _UNORM twin, so what the shaders write is the byte the compositor reads. xr.rawColor=0 = old.
    static XrSwapchainCreateInfo RawColorCreateInfo(const XrSwapchainCreateInfo& in) {
        XrSwapchainCreateInfo out = in;
        if (arcadexr::config::GetInt("xr.rawColor", 0) == 0) return out;   // off: shaders decode instead (no compression loss)
        if (in.format == VK_FORMAT_R8G8B8A8_SRGB) out.format = VK_FORMAT_R8G8B8A8_UNORM;
        else if (in.format == VK_FORMAT_B8G8R8A8_SRGB) out.format = VK_FORMAT_B8G8R8A8_UNORM;
        if (out.format != in.format) Log::Write(Log::Level::Info, Fmt("TCVR_VK raw colour views: swapchain %d viewed as %d", int(in.format), int(out.format)));
        return out;
    }

    ISwapchainImageData* AllocateSwapchainImageData(size_t size, const XrSwapchainCreateInfo& swapchainCreateInfo) override {
        auto typedResult = std::make_unique<VulkanSwapchainImageData>(
            m_namer, uint32_t(size), RawColorCreateInfo(swapchainCreateInfo), m_vkDevice, &m_memAllocator, m_pipelineLayout, m_computePipelineLayout,
            m_shaderProgram, m_computeShaderProgram, m_drawBuffer.bindDesc, m_drawBuffer.attrDesc);

        // Cast our derived type to the caller-expected type.
        auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

        m_swapchainImageDataMap.Adopt(std::move(typedResult));

        return ret;
    }

    inline ISwapchainImageData* AllocateSwapchainImageDataWithDepthSwapchain(
        size_t size, const XrSwapchainCreateInfo& colorSwapchainCreateInfo, XrSwapchain depthSwapchain,
        const XrSwapchainCreateInfo& depthSwapchainCreateInfo) override {
        m_xrDepthFormat = int64_t(depthSwapchainCreateInfo.format);
        auto typedResult = std::make_unique<VulkanSwapchainImageData>(
            m_namer, uint32_t(size), RawColorCreateInfo(colorSwapchainCreateInfo), depthSwapchain, depthSwapchainCreateInfo, m_vkDevice,
            &m_memAllocator, m_pipelineLayout, m_computePipelineLayout, m_shaderProgram, m_computeShaderProgram,
            m_drawBuffer.bindDesc, m_drawBuffer.attrDesc);

        // Cast our derived type to the caller-expected type.
        auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

        m_swapchainImageDataMap.Adopt(std::move(typedResult));

        return ret;
    }

    void SetViewportAndScissor(VkCommandBuffer cmd, const VkRect2D& rect) {
        VkViewport viewport{
            float(rect.offset.x), float(rect.offset.y), float(rect.extent.width), float(rect.extent.height), 0.0f, 1.0f};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &rect);
    }

    void CreateScreenPipeline(VkRenderPass renderPass) {
        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &m_screenDrawBuffer.bindDesc;
        vi.vertexAttributeDescriptionCount = (uint32_t)m_screenDrawBuffer.attrDesc.size();
        vi.pVertexAttributeDescriptions = m_screenDrawBuffer.attrDesc.data();

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.primitiveRestartEnable = VK_FALSE;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineColorBlendAttachmentState attachState{};
        attachState.blendEnable = VK_FALSE;
        attachState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &attachState;

        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeInfo.stageCount = (uint32_t)m_screenShaderProgram.shaderInfo.size();
        pipeInfo.pStages = m_screenShaderProgram.shaderInfo.data();
        pipeInfo.pVertexInputState = &vi;
        pipeInfo.pInputAssemblyState = &ia;
        pipeInfo.pViewportState = &vp;
        pipeInfo.pRasterizationState = &rs;
        pipeInfo.pMultisampleState = &ms;
        pipeInfo.pDepthStencilState = &ds;
        pipeInfo.pColorBlendState = &cb;
        pipeInfo.pDynamicState = &dynamicState;
        pipeInfo.layout = m_screenPipelineLayout;
        pipeInfo.renderPass = renderPass;
        pipeInfo.subpass = 0;

        XRC_CHECK_THROW_VKCMD(vkCreateGraphicsPipelines(m_vkDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_screenPipeline));
    }

    void CreateOrUpdateScreenTexture(uint32_t width, uint32_t height) {
        if (m_screenImage != VK_NULL_HANDLE && m_screenWidth == width && m_screenHeight == height) {
            return;
        }
        if (m_screenImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_vkDevice, m_screenImageView, nullptr);
            m_screenImageView = VK_NULL_HANDLE;
        }
        if (m_screenImage != VK_NULL_HANDLE) {
            vkDestroyImage(m_vkDevice, m_screenImage, nullptr);
            m_screenImage = VK_NULL_HANDLE;
        }
        if (m_screenImageMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_vkDevice, m_screenImageMemory, nullptr);
            m_screenImageMemory = VK_NULL_HANDLE;
        }
        if (m_stagingBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_vkDevice, m_stagingBuffer, nullptr);
            m_stagingBuffer = VK_NULL_HANDLE;
        }
        if (m_stagingBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_vkDevice, m_stagingBufferMemory, nullptr);
            m_stagingBufferMemory = VK_NULL_HANDLE;
        }

        m_screenWidth = width;
        m_screenHeight = height;

        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_B8G8R8A8_SRGB;   // MAME's display bytes: decoded on sampling, re-encoded on write = true colours
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        XRC_CHECK_THROW_VKCMD(vkCreateImage(m_vkDevice, &imageInfo, nullptr, &m_screenImage));

        VkMemoryRequirements memReq{};
        vkGetImageMemoryRequirements(m_vkDevice, m_screenImage, &memReq);
        m_memAllocator.Allocate(memReq, &m_screenImageMemory, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        XRC_CHECK_THROW_VKCMD(vkBindImageMemory(m_vkDevice, m_screenImage, m_screenImageMemory, 0));

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = m_screenImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_B8G8R8A8_SRGB;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        XRC_CHECK_THROW_VKCMD(vkCreateImageView(m_vkDevice, &viewInfo, nullptr, &m_screenImageView));

        VkDeviceSize stagingSize = (VkDeviceSize)std::max(width, 1024u) * height * 4;
        VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo.size = stagingSize;
        bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        XRC_CHECK_THROW_VKCMD(vkCreateBuffer(m_vkDevice, &bufInfo, nullptr, &m_stagingBuffer));

        vkGetBufferMemoryRequirements(m_vkDevice, m_stagingBuffer, &memReq);
        m_memAllocator.Allocate(memReq, &m_stagingBufferMemory,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        XRC_CHECK_THROW_VKCMD(vkBindBufferMemory(m_vkDevice, m_stagingBuffer, m_stagingBufferMemory, 0));
        m_stagingBufferSize = stagingSize;

        VkDescriptorImageInfo descImageInfo{};
        descImageInfo.sampler = m_screenSampler;
        descImageInfo.imageView = m_screenImageView;
        descImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet writeDesc{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writeDesc.dstSet = m_screenDescriptorSet;
        writeDesc.dstBinding = 0;
        writeDesc.dstArrayElement = 0;
        writeDesc.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDesc.descriptorCount = 1;
        writeDesc.pImageInfo = &descImageInfo;
        vkUpdateDescriptorSets(m_vkDevice, 1, &writeDesc, 0, nullptr);

        m_screenImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void RenderView(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
                    int64_t /*swapchainFormat*/, const std::vector<Cube>& cubes) override {
        CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.

        VulkanSwapchainImageData* swapchainData;
        uint32_t imageIndex;

        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(swapchainImage);

        if (viewIndex == 0) m_frameSlot ^= 1u;
        const size_t v = m_frameSlot * 2 + (viewIndex % 2);   // command buffer of (frame slot, eye)

        // Double-buffered command execution:
        // Wait for previous frame's GPU completion at the start of view 0,
        // allowing the CPU to record without blocking at the end of each eye.
        using clk = std::chrono::steady_clock;
        const auto tView0 = clk::now();
        // Left eye: take the emulator's newest scene and BUILD it on the CPU first, while the GPU
        // is still drawing the previous frame; only then wait for that frame's fence.
        const tcvr_m2_frame* m2Frame = nullptr;
        if (viewIndex == 0) {
            // Wait only for the frame that used THIS slot (two frames ago); the previous frame keeps
            // running on the GPU while we prepare this one. Outside the immersive pass (screen
            // fallback, whose staging buffer is single) keep the old full serialization.
            const auto tw = clk::now();
            const uint32_t s0 = m_frameSlot * 2;
            // Wait() throws unless the buffer is Executing or Initialized: a buffer already waited
            // (Executable) must not be waited again -- that exception killed the render thread.
            auto waitIfRunning = [](CmdBuffer& cb) {
                if (cb.state == CmdBuffer::CmdBufferState::Executing) cb.Wait();
            };
            waitIfRunning(m_cmdBuffer[s0]);
            waitIfRunning(m_cmdBuffer[s0 + 1]);
            if (!m_lastM2Drawn) {
                waitIfRunning(m_cmdBuffer[(s0 + 2) & 3]);
                waitIfRunning(m_cmdBuffer[(s0 + 3) & 3]);
            }
            m_cpuWaitMs += std::chrono::duration<float, std::milli>(clk::now() - tw).count();
            m_cmdBuffer[s0].Clear();
            m_cmdBuffer[s0 + 1].Clear();
            ReadGpuTimestamps();
            if (m_dumpState == 1 && m_cmdBuffer[m_dumpCb].state != CmdBuffer::CmdBufferState::Executing) WriteDump();
            FlushOracle();
            // Game switched in the same process (headset selector, debug.tcvr.switch_game): the Model 2 module
            // still held the previous game's geometry, textures and last frame, and HaveSceneSource() stays true
            // for every game once the Model 2 library is loaded -> Time Crisis II showed Virtua Racing frozen.
            {
                const std::string game = arcadexr::profiles::CurrentGame();
                if (game != m_m2Game) { m_m2Game = game; ResetM2ForGameChange(); }
            }
            m_m2Renderer.SetFrameSlot(int(m_frameSlot));
            ++m_paceHold;
            {
                const auto now = std::chrono::steady_clock::now();
                if (now - m_paceAt >= std::chrono::seconds(1)) {
                    if (m_paceNew > 0)
                        Log::Write(Log::Level::Info, Fmt("TCVR_PACING arcade frames shown=%u never shown=%u | held 1 refresh=%u 2=%u 3=%u 4+=%u",
                                                         m_paceNew, m_paceSkipped, m_paceHist[1], m_paceHist[2], m_paceHist[3], m_paceHist[4]));
                    m_paceNew = m_paceSkipped = 0; for (auto& h : m_paceHist) h = 0; m_paceAt = now;
                }
            }
            if (!m_m2SceneRequested && M2SceneLive()) {
                m_m2SceneRequested = true;
                SetM2SceneMode(1);
            }
            // Bench: debug.tcvr.m2_freeze=1 keeps re-drawing the last scene (no new acquire): the same
            // workload every frame, so an A/B of a render setting is not drowned in scene variance.
            const bool freeze = arcadexr::config::GetInt("m2.freeze", 0) != 0 && m_lastM2Frame != nullptr;
            if (freeze) {
                m2Frame = m_lastM2Frame;
            } else if (M2SceneLive()) {
                m2Frame = arcadexr::hardware::sega_model2::AcquireScene();
                if (m2Frame) m_lastM2Frame = m2Frame;
                m_m2Renderer.SetSmooth(m_smoothOn);
                if (m2Frame) {
                    const auto tp = clk::now();
                    m_m2Renderer.BuildFrame(*m2Frame);
                    m_cpuPrepMs += std::chrono::duration<float, std::milli>(clk::now() - tp).count();
                    // Arcade frame pacing (25/09): frames the emulator published but no refresh ever showed (two
                    // published between two refreshes: a jump in the motion), and how many refreshes each shown frame
                    // stayed up. Logged once a second as TCVR_PACING.
                    if (m2Frame->sequence != m_lastM2RenderedSeq) {
                        if (m_lastM2RenderedSeq != 0 && m2Frame->sequence > m_lastM2RenderedSeq + 1)
                            m_paceSkipped += uint32_t(m2Frame->sequence - m_lastM2RenderedSeq - 1);
                        if (m_paceHold > 0) ++m_paceHist[std::min<uint32_t>(m_paceHold, 4u)];
                        m_paceHold = 0; ++m_paceNew;
                    }
                    m_lastM2RenderedSeq = m2Frame->sequence;
                }
            }
            // Arcade cadence for the Model 2 chain (23/09, Virtua Racing judder): the game makes 57.5 images a
            // second; at 90 Hz each stays up 1 then 2 refreshes, irregularly. 120 Hz + one draw per new arcade image
            // = 2 refreshes each (a 3rd about 5 times a second). Its own flag: never fights the System 22 one.
            {
                // Smooth motion (Model 1): draw every display refresh at 90 Hz, interpolated between arcade frames,
                // instead of the 120 Hz cadence that shows each arcade frame twice.
                m_smoothOn = m_lastM2Drawn && m_m2Renderer.HasMotionIds() &&
                             arcadexr::profiles::GetInt("immersive.smoothMotion", 1) != 0;
                const bool m2Cadence = m_lastM2Drawn && !m_smoothOn && arcadexr::profiles::GetInt("immersive.cadence", 0) != 0;
                // 120 Hz for the cadence, 90 Hz for smooth motion (drawn every refresh), 90 when leaving either.
                const int wantRate = m2Cadence ? 120 : (m_smoothOn ? 90 : 0);
                if (wantRate != m_m2RateRequested) {
                    if (wantRate > 0 || m_m2RateRequested > 0)
                        arcadexr::xr::RequestRateForGame(wantRate > 0 ? float(wantRate) : 90.0f,
                            m2Cadence ? "Model 2 chain: 2 refreshes per arcade frame" : (m_smoothOn ? "smooth motion: every refresh at 90 Hz" : "leaving Model 2 cadence"));
                    m_m2RateRequested = wantRate;
                }
                m_m2CadenceRequested = m2Cadence;
                m_m2CadenceActive = m2Cadence;
            }
        }

        VkCommandBuffer cmd = m_cmdBuffer[v].buf;
        m_cmdBuffer[v].Begin();
        if (m_gpuQueryPool == VK_NULL_HANDLE) CreateGpuQueryPool();
        if (m_gpuQueryPool != VK_NULL_HANDLE) {
            vkCmdResetQueryPool(cmd, m_gpuQueryPool, uint32_t(v * 2), 2);   // v = slot*2 + eye
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_gpuQueryPool, uint32_t(v * 2));
        }

        // 1. If viewIndex == 0, commit the built scene (GPU-visible copies, uploads) and upload the MAME frame
        if (viewIndex == 0) {
            if (m2Frame) m_m2Renderer.CommitFrame(*m2Frame, cmd);   // frozen: same built arrays re-committed
            {
                const auto tp = clk::now();
                PrepareSystem22(cmd, swapchainData);
                m_cpuPrepMs += std::chrono::duration<float, std::milli>(clk::now() - tp).count();
            }
            // Model 1/2 immersive: dynamic resolution (sub-rectangle of the 2x supersampled swapchain, applied
            // from the next frame; the compositor scales it for free). See UpdateM2DynamicScale.
            if (!m_s22Active && m_lastM2Drawn && M2SceneLive()) {
                const float fixed = arcadexr::config::GetFloat("m2.viewportScale", 0.0f);
                m_viewportScale = fixed > 0.0f ? std::max(0.3f, std::min(1.0f, fixed)) : m_m2DynScale;   // 1.0 unless dynres on
            }
            // SCREEN presentation of a Model 2 game: the GPU draws it in the board's projection (MAME can
            // stop rasterising, like in immersive). Falls back to MAME's framebuffer when it cannot.
            m_flatDrawn = false;
            {
                const std::string pres = arcadexr::profiles::GetString("presentation", "immersive");
                const bool wantFlat = pres != "immersive" && M2SceneLive() &&
                                      arcadexr::config::GetInt("m2.vkFlat", 1) != 0;
                {   // why flat (25/09: Top Skater showed flat with its profile saying immersive)
                    static std::string s_lastPresLog;
                    const std::string key = pres + "|" + arcadexr::profiles::CurrentGame() + "|" + (wantFlat ? "1" : "0");
                    if (key != s_lastPresLog) {
                        s_lastPresLog = key;
                        Log::Write(Log::Level::Info, Fmt("TCVR_PRES game=%s presentation=%s wantFlat=%d m2Live=%d",
                            arcadexr::profiles::CurrentGame().c_str(), pres.c_str(), int(wantFlat), int(M2SceneLive())));
                    }
                }
                if (wantFlat) {
                    EnsureM2Renderer(swapchainData);
                    if (m_m2Renderer.HasGeometry() && m_m2Renderer.RenderFlat(cmd)) {
                        m_flatDrawn = true;
                        m_flatW = m_m2Renderer.FlatWidth(); m_flatH = m_m2Renderer.FlatHeight();
                        if (m_m2Renderer.FlatView() != m_flatBoundView) {
                            VkDescriptorImageInfo di{m_flatSampler, m_m2Renderer.FlatView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                            VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                            wd.dstSet = m_flatDescriptorSet; wd.dstBinding = 0; wd.descriptorCount = 1;
                            wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wd.pImageInfo = &di;
                            vkDeviceWaitIdle(m_vkDevice);   // rare: only when the flat target is (re)created
                            vkUpdateDescriptorSets(m_vkDevice, 1, &wd, 0, nullptr);
                            m_flatBoundView = m_m2Renderer.FlatView();
                        }
                    }
                }
            }
            if (m_s22FlatWanted && m_s22Prepared) {
                arcadexr::vulkan::VulkanSystem22Renderer::Settings st;
                st.texSamples = 1;
                st.darkFlicker = arcadexr::profiles::GetInt("temporal.darkFlicker", 0) != 0;
        st.darkMode = std::max(1, arcadexr::profiles::GetInt("temporal.darkFlicker", 0));
                const float scale = std::max(1.0f, std::min(4.0f, arcadexr::config::GetFloat("s22.flatScale", 2.0f)));
                if (m_s22.RenderFlat(cmd, scale, st)) {
                    m_flatDrawn = true;
                    m_flatW = m_s22.FlatWidth(); m_flatH = m_s22.FlatHeight();
                    if (m_s22.FlatView() != m_flatBoundView) {
                        VkDescriptorImageInfo di{m_flatSampler, m_s22.FlatView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                        VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        wd.dstSet = m_flatDescriptorSet; wd.dstBinding = 0; wd.descriptorCount = 1;
                        wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wd.pImageInfo = &di;
                        vkDeviceWaitIdle(m_vkDevice);   // rare: only when the flat target is (re)created
                        vkUpdateDescriptorSets(m_vkDevice, 1, &wd, 0, nullptr);
                        m_flatBoundView = m_s22.FlatView();
                    }
                }
            }
            if (m_flatDrawn) RecordOracle(cmd, uint32_t(v));
            // Game selector menu: CPU-drawn picture, uploaded when it changed.
            if (arcadexr::ui::Menu::Get().IsOpen()) {
                if (!m_overlayInit) {
                    m_overlay.Init(m_vkDevice, &m_memAllocator, VkFormat(swapchainData->GetSlices()[0].m_rp.colorFmt));
                    m_overlayInit = true;
                }
                std::vector<unsigned char> rgba;
                int mw = 0, mh = 0;
                if (arcadexr::ui::Menu::Get().Render(rgba, mw, mh) || !m_overlay.HaveMenuImage()) {
                    if (!rgba.empty()) m_overlay.UploadMenu(cmd, rgba, mw, mh);
                }
            }
            arcadexr::video::FrameInfo info;
            // The flat framebuffer only feeds the screen fallback: skip its 760 KB copy while the
            // immersive pass is the one drawing (it resumes the first frame the fallback runs).
            if (!m_lastM2Drawn && !m_flatDrawn && arcadexr::video::PeekLatestFrameInfo(info)) {
                if (info.sequence != m_lastScreenSequence || (uint32_t)info.width != m_screenWidth || (uint32_t)info.height != m_screenHeight) {
                    arcadexr::video::FrameInfo got;
                    const std::uint32_t* pixels = arcadexr::video::AcquireLatestFrame(got);
                    if (pixels) {
                        CreateOrUpdateScreenTexture(got.width, got.height);
                        void* mapped = nullptr;
                        VkDeviceSize copySize = (VkDeviceSize)got.stride * got.height * 4;
                        if (copySize <= m_stagingBufferSize) {
                            XRC_CHECK_THROW_VKCMD(vkMapMemory(m_vkDevice, m_stagingBufferMemory, 0, copySize, 0, &mapped));
                            memcpy(mapped, pixels, copySize);
                            vkUnmapMemory(m_vkDevice, m_stagingBufferMemory);

                            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                            barrier.srcAccessMask = (m_screenImageLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? 0 : VK_ACCESS_SHADER_READ_BIT;
                            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                            barrier.oldLayout = m_screenImageLayout;
                            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                            barrier.image = m_screenImage;
                            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                            vkCmdPipelineBarrier(cmd,
                                                 (m_screenImageLayout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 0, 0, nullptr, 0, nullptr, 1, &barrier);

                            VkBufferImageCopy region{};
                            region.bufferOffset = 0;
                            region.bufferRowLength = got.stride;
                            region.bufferImageHeight = got.height;
                            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                            region.imageOffset = {0, 0, 0};
                            region.imageExtent = {(uint32_t)got.width, (uint32_t)got.height, 1};
                            vkCmdCopyBufferToImage(cmd, m_stagingBuffer, m_screenImage,
                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                            vkCmdPipelineBarrier(cmd,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                                 0, 0, nullptr, 0, nullptr, 1, &barrier);
                            m_screenImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                            m_lastScreenSequence = got.sequence;

                            if (!m_loggedScreenUpload) {
                                Log::Write(Log::Level::Info, Fmt("TCVR_VK screen texture upload seq=%llu size=%dx%d stride=%d",
                                                                 static_cast<unsigned long long>(got.sequence), got.width, got.height, got.stride));
                                m_loggedScreenUpload = true;
                            }
                        }
                    }
                }
            }
        }

        const XrRect2Di& r = layerView.subImage.imageRect;
        VkRect2D renderArea = {{r.offset.x, r.offset.y}, {uint32_t(r.extent.width), uint32_t(r.extent.height)}};

        // Model 2 Native Immersive Rendering, in the renderer's OWN render pass (MSAA resolved
        // on-tile into this swapchain image). Falls back to the plain pass below if it drew nothing.
        const std::string presentation = arcadexr::profiles::GetString("presentation", "immersive");
        const bool immersiveAllowed = (presentation == "immersive" && arcadexr::profiles::GetInt("m2.immersive", 1) != 0);
        bool m2ImmersiveDrawn = false;
        bool s22Drawn = false;
        if (m_s22Active) {
            s22Drawn = RenderSystem22Eye(cmd, viewIndex, layerView, swapchainData, imageIndex, renderArea);
            m2ImmersiveDrawn = s22Drawn;   // the eye image is written: skip the other paths
        }
        if (immersiveAllowed && !s22Drawn && M2SceneLive()) {
            EnsureM2Renderer(swapchainData);
            if (m_m2Renderer.HasGeometry()) {
                const float clear[4] = {m_clearColor[0], m_clearColor[1], m_clearColor[2], 1.0f};
                const VkExtent2D ext{uint32_t(swapchainData->Width()), uint32_t(swapchainData->Height())};
                VkImage fdmImage = VK_NULL_HANDLE;
                VkExtent2D fdmExt{0, 0};
                auto fit = m_fdmImages.find(static_cast<const ISwapchainImageData*>(swapchainData));
                if (fit != m_fdmImages.end() && imageIndex < fit->second.size()) {
                    fdmImage = fit->second[imageIndex].image;
                    fdmExt = {fit->second[imageIndex].width, fit->second[imageIndex].height};
                }
                m_m2Renderer.BeginPass(cmd, viewIndex, swapchainData->GetTypedImage(imageIndex).image, ext, renderArea, clear,
                                       fdmImage, fdmExt,
                                       m_m2Renderer.DepthResolveOn() ? swapchainData->GetDepthImageForColorIndex(imageIndex).image : VK_NULL_HANDLE);
                SetViewportAndScissor(cmd, renderArea);
                m2ImmersiveDrawn = m_m2Renderer.RenderImmersive(viewIndex, layerView, cmd, {uint32_t(r.extent.width), uint32_t(r.extent.height)});
                vkCmdEndRenderPass(cmd);
            }
        }
        // The Model 2 / System 22 modules draw with their own transient depth: the depth swapchain image of this
        // view is not written, and must not be submitted (25/09: the car-select menu slid when the head moved).
        m_viewWroteDepth = !m2ImmersiveDrawn;
        m_viewWroteSwDepth = false;
        if (!m2ImmersiveDrawn) {
        SetViewportAndScissor(cmd, renderArea);

        // may be depth, stencil, or both
        // XXX support VK_IMAGE_ASPECT_STENCIL_BIT
        VkImageAspectFlags secondAttachmentAspect = VK_IMAGE_ASPECT_DEPTH_BIT;

        VkRenderPassBeginInfo renderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};

        // aka slice
        auto imageArrayIndex = layerView.subImage.imageArrayIndex;

        swapchainData->BindRenderTarget(imageIndex, imageArrayIndex, renderArea, secondAttachmentAspect, &renderPassBeginInfo);

        if (!swapchainData->DepthSwapchainEnabled()) {
            // Ensure self-made fallback depth is in the right layout
            swapchainData->TransitionLayout(imageIndex, &m_cmdBuffer[v], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }

        // Bind and clear eye render target via render pass LOAD_OP_CLEAR
        static std::array<VkClearValue, 2> clearValues;
        clearValues[0].color.float32[0] = m_clearColor[0];
        clearValues[0].color.float32[1] = m_clearColor[1];
        clearValues[0].color.float32[2] = m_clearColor[2];
        clearValues[0].color.float32[3] = 1.0f;
        clearValues[1].depthStencil.depth = 1.0f;
        clearValues[1].depthStencil.stencil = 0;
        renderPassBeginInfo.clearValueCount = (uint32_t)clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Compute the view-projection transform.
        const auto& pose = layerView.pose;
        XrMatrix4x4f proj;
        XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_VULKAN, layerView.fov, 0.05f, 100.0f);
        XrMatrix4x4f toView;
        XrMatrix4x4f_CreateFromRigidTransform(&toView, &pose);
        XrMatrix4x4f view;
        XrMatrix4x4f_InvertRigidBody(&view, &toView);
        XrMatrix4x4f vp;
        XrMatrix4x4f_Multiply(&vp, &proj, &view);

        // Render Virtual Arcade Screen fallback
        if (!m2ImmersiveDrawn) {
            arcadexr::gun::ScreenPlane screen;
            if ((m_screenImage != VK_NULL_HANDLE || m_flatDrawn) && arcadexr::video::GetVirtualScreen(screen)) {
                if (m_screenPipeline == VK_NULL_HANDLE) {
                    CreateScreenPipeline(renderPassBeginInfo.renderPass);
                }
                if (m_screenWidth > 0 && m_screenHeight > 0) {
                    arcadexr::video::UpdateVirtualScreenAspect(static_cast<float>(m_screenWidth) / static_cast<float>(m_screenHeight));
                    arcadexr::video::GetVirtualScreen(screen);
                }

                XrMatrix4x4f model{};
                model.m[0] = screen.right.x * screen.width;
                model.m[1] = screen.right.y * screen.width;
                model.m[2] = screen.right.z * screen.width;
                model.m[4] = screen.up.x * screen.height;
                model.m[5] = screen.up.y * screen.height;
                model.m[6] = screen.up.z * screen.height;
                model.m[8] = screen.normal.x;
                model.m[9] = screen.normal.y;
                model.m[10] = screen.normal.z;
                model.m[12] = screen.center.x;
                model.m[13] = screen.center.y;
                model.m[14] = screen.center.z;
                model.m[15] = 1.0f;

                XrMatrix4x4f mvp;
                XrMatrix4x4f_Multiply(&mvp, &vp, &model);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_screenPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_screenPipelineLayout,
                                        0, 1, m_flatDrawn ? &m_flatDescriptorSet : &m_screenDescriptorSet, 0, nullptr);
                ScreenPC spc{};
                std::memcpy(spc.mvp, mvp.m, sizeof(spc.mvp));
                const auto aim = arcadexr::gun::GetAimState();
                spc.aim[0] = aim.normalized_x; spc.aim[1] = aim.normalized_y;
                spc.srcSize[0] = float(m_screenWidth); spc.srcSize[1] = float(m_screenHeight);
                if (m_flatDrawn) { spc.srcSize[0] = float(m_flatW); spc.srcSize[1] = float(m_flatH); }
                spc.aimVisible = aim.show_crosshair ? 1 : 0;
                spc.calibrating = aim.calibrating ? 1 : 0;
                // "edge" / "catmull" -> Catmull-Rom (the GLES edge pass adds FXAA on top: not ported yet)
                const std::string filt = arcadexr::profiles::GetString("filter", "edge");
                spc.filter = (m_flatDrawn || filt == "nearest" || filt == "bilinear") ? 1 : 3;   // flat GPU image: trilinear
                spc.sharpen = float(std::atof(arcadexr::profiles::GetString("sharpen", "0").c_str()));
                vkCmdPushConstants(cmd, m_screenPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(spc), &spc);

                VkDeviceSize vtxOffset = 0;
                vkCmdBindIndexBuffer(cmd, m_screenDrawBuffer.idx.buf, 0, VK_INDEX_TYPE_UINT16);
                vkCmdBindVertexBuffers(cmd, 0, 1, &m_screenDrawBuffer.vtx.buf, &vtxOffset);
                vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
            }
        }

        // Render cubes
        if (!cubes.empty()) {
            swapchainData->BindPipeline(cmd, imageArrayIndex);
            vkCmdBindIndexBuffer(cmd, m_drawBuffer.idx.buf, 0, VK_INDEX_TYPE_UINT16);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &m_drawBuffer.vtx.buf, &offset);

            for (const Cube& cube : cubes) {
                XrMatrix4x4f model;
                XrMatrix4x4f_CreateTranslationRotationScale(&model, &cube.Pose.position, &cube.Pose.orientation, &cube.Scale);
                XrMatrix4x4f mvp;
                XrMatrix4x4f_Multiply(&mvp, &vp, &model);
                vkCmdPushConstants(cmd, m_pipelineLayout.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mvp.m), &mvp.m[0]);
                vkCmdDrawIndexed(cmd, m_drawBuffer.count.idx, 1, 0, 0, 0);
            }
        }

        vkCmdEndRenderPass(cmd);
        }  // legacy pass (no immersive frame)
        RenderOverlay(cmd, viewIndex, layerView, swapchainData, imageIndex, renderArea);
        // Debug dump of the LEFT eye as rendered (debug.tcvr.dump=<tag>): copied into a host buffer,
        // written as files/dump-<tag>-vkL.ppm after the fence. The compositor screencap is black
        // while the headset is worn; this is the image the app really produced.
        // A tag ending in "_R" dumps the RIGHT eye (dump-<tag>-vkR.ppm): with the left one, the disparity between two
        // planes -- a parallax -- becomes measurable (26/09: one eye alone showed a menu "glued" that was not).
        if (m_dumpState == 0) {
            const std::string tag = arcadexr::config::GetString("dump", "0");
            const bool rightEye = tag.size() > 2 && tag.compare(tag.size() - 2, 2, "_R") == 0;
            if (viewIndex == (rightEye ? 1u : 0u) && !tag.empty() && tag != "0" && tag != m_dumpDoneTag) {
                const VkImage img = swapchainData->GetTypedImage(imageIndex).image;
                const uint32_t w = swapchainData->Width(), h = swapchainData->Height();
                const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
                if (m_dumpBuf == VK_NULL_HANDLE || m_dumpSize < bytes) {
                    if (m_dumpBuf) { vkDestroyBuffer(m_vkDevice, m_dumpBuf, nullptr); vkFreeMemory(m_vkDevice, m_dumpMem, nullptr); }
                    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                    vkCreateBuffer(m_vkDevice, &bi, nullptr, &m_dumpBuf);
                    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(m_vkDevice, m_dumpBuf, &req);
                    m_memAllocator.Allocate(req, &m_dumpMem);
                    vkBindBufferMemory(m_vkDevice, m_dumpBuf, m_dumpMem, 0);
                    m_dumpSize = bytes;
                }
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                VkBufferImageCopy r{};
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                r.imageExtent = {w, h, 1};
                vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_dumpBuf, 1, &r);
                b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                m_dumpState = 1; m_dumpTag = tag; m_dumpW = w; m_dumpH = h; m_dumpCb = uint32_t(v);
                m_dumpInterp = m_m2Renderer.LastInterp();
                // Only what the compositor shows: the sub-rectangle of a dynamic-resolution frame (25/09: the rest of
                // the swapchain holds older frames at other scales -- nested frames Guillaume never saw).
                m_dumpVisW = std::max(1u, std::min(w, uint32_t(float(renderArea.extent.width))));
                m_dumpVisH = std::max(1u, std::min(h, uint32_t(float(renderArea.extent.height))));
            }
        }
        if (m_gpuQueryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_gpuQueryPool, uint32_t(v * 2 + 1));
            m_gpuQueryWritten[v] = true;
        }
        // Scene recording mode, decided once per frame on the left eye. Mode 2 = MAME records
        // the scene INSTEAD of rasterising it (its CPU raster costs ~10 ms/frame, which is what
        // dropped the emulation to ~53 fps and pitched the sound down). Only while the immersive
        // pass really drew a centred main view (a race); menus/intro keep mode 1 so the flat
        // fallback always has a fresh picture.
        if (viewIndex == 0) m_lastM2Drawn = m2ImmersiveDrawn;
        if (viewIndex == 0) {
            // Model 2 immersive lightgun: the aim follows the scene only while that scene is what the player sees.
            m_m2AimLive = m2ImmersiveDrawn && !s22Drawn && m_m2Renderer.HaveMainView();
            if (m_m2AimLive) {
                if (s_s22Self != this) { s_s22Self = this; arcadexr::gun::SetSceneAim(&S22AimTrampoline); }
                m_m2Renderer.AimSelfTest();
            }
        }
        if (viewIndex == 0 && m_m2SceneRequested) {
            const bool allowSkip = arcadexr::config::GetInt("m2.skipCpuRaster", 1) != 0;
            SetM2SceneMode((((m2ImmersiveDrawn && m_m2Renderer.HaveMainView()) || m_flatDrawn) && allowSkip) ? 2 : 1);
        }

        // AppSW: the real distance of this eye's polygons into the headset's low-resolution AppSW depth image.
        if (viewIndex < 2 && m_swDepthTarget[viewIndex].image != VK_NULL_HANDLE) {
            const SwDepthTarget t = m_swDepthTarget[viewIndex];
            m_swDepthTarget[viewIndex] = {};
            m_viewWroteSwDepth = m_m2Renderer.RenderAppSwDepth(cmd, viewIndex, t.image, t.ext, t.format, m2ImmersiveDrawn && !s22Drawn);
        }
        m_cmdBuffer[v].End();
        const auto tSubmit = clk::now();
        m_cmdBuffer[v].Exec(m_vkQueue);
        m_cpuSubmitMs += std::chrono::duration<float, std::milli>(clk::now() - tSubmit).count();
        m_cpuViewMs += std::chrono::duration<float, std::milli>(clk::now() - tView0).count();
        if (viewIndex == 1) {
            ++m_cpuFrames;
            // Frame-to-frame period of the render loop (includes xrWaitFrame / EndFrame outside us).
            const auto now = clk::now();
            if (m_cpuLastFrame.time_since_epoch().count() != 0)
                m_cpuPeriodMs += std::chrono::duration<float, std::milli>(now - m_cpuLastFrame).count();
            m_cpuLastFrame = now;
            if (now - m_cpuLogAt >= std::chrono::seconds(2)) {
                const float n = float(m_cpuFrames);
                Log::Write(Log::Level::Info, Fmt("TCVR_VKCPU per frame (ms): period=%.2f inRenderView=%.2f waitFence=%.2f prepare=%.2f submit=%.2f frames=%u",
                                                 m_cpuPeriodMs / n, m_cpuViewMs / n, m_cpuWaitMs / n, m_cpuPrepMs / n, m_cpuSubmitMs / n, m_cpuFrames));
                m_cpuPeriodMs = m_cpuViewMs = m_cpuWaitMs = m_cpuPrepMs = m_cpuSubmitMs = 0.0f;
                m_cpuFrames = 0;
                m_cpuLogAt = now;
            }
        }
        // Fully asynchronous GPU execution:
        // Do NOT call m_cmdBuffer[v].Wait() here! Fences are waited at the start
        // of the next frame (viewIndex == 0).

#if defined(USE_MIRROR_WINDOW)
        // Cycle the window's swapchain on the last view rendered
        // XXX bit of a hack
        if (layerView.subImage.imageRect.offset.x != 0) {
            m_swapchain.Acquire();
            m_swapchain.Wait();
            m_swapchain.Present(m_vkQueue);
        }
#endif
    }

    bool WantsFoveationFdm() const override { return m_fdmEnabled; }
    float ViewportScale() const override { return m_viewportScale; }
    bool ViewWroteDepth() const override { return m_viewWroteDepth; }
    void SetAppSwWanted(bool on) override {
        m_appswWanted = on;   // kept here: the renderer is rebuilt at each game change (ResetM2ForGameChange)
        m_m2Renderer.SetCameraDeltaWanted(on);
        // resolve only into a depth image of the pass's own format (D32F); decided before the renderer's Initialize
        // Resolving the colour pass's depth is useless for AppSW: it holds the painter RANK (m2.depthOrder), not the
        // distance -- the ghosting of 26/09. The real distance comes from RenderAppSwDepth. Kept off.
        const bool resolve = false;
        m_m2Renderer.SetDepthResolveWanted(resolve);
        Log::Write(Log::Level::Info, Fmt("TCVR_APPSW wanted=%d depth resolve=%d (ext=%d xrDepthFormat=%lld)", int(on), int(resolve),
                                         int(m_haveDepthResolve), (long long)m_xrDepthFormat));
    }
    bool ViewWroteSwDepth() const override { return m_viewWroteSwDepth; }
    // The AppSW depth image paired with this view's motion-vector image (acquired by the program before RenderView).
    void SetAppSwDepthTarget(uint32_t view, const XrSwapchainImageBaseHeader* mvImage) override {
        if (view >= 2 || !mvImage) return;
        auto dataAndIndex = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(mvImage);
        VulkanSwapchainImageData* d = dataAndIndex.first;
        if (!d || !d->DepthSwapchainEnabled()) return;
        m_swDepthTarget[view] = {d->GetDepthImageForColorIndex(dataAndIndex.second).image,
                                 {uint32_t(d->Width()), uint32_t(d->Height())}, VkFormat(d->GetDepthFormat())};
    }
    void AppSwDepthRange(float* nearZ, float* farZ) override { m_m2Renderer.DepthRange(nearZ, farZ); }
    bool AppSwDelta(XrPosef* pose) override {
        *pose = XrPosef{{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
        const uint32_t seq = m_m2Renderer.CameraDeltaSeq();
        uint32_t agree = 0, pairs = 0;
        bool ok = true;
        if (seq != m_appswSeq) {   // a new arcade frame since the last submission: its camera motion
            ok = m_m2Renderer.AppSpaceDelta(pose, &agree, &pairs);
            if (!ok) *pose = XrPosef{{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
            if ((++m_appswLogTick % 60u) == 0u)
                Log::Write(Log::Level::Info, Fmt("TCVR_APPSW delta ok=%d agree=%u/%u pos=(%.3f,%.3f,%.3f) q=(%.4f,%.4f,%.4f,%.4f) skipped=%u",
                    int(ok), agree, pairs, pose->position.x, pose->position.y, pose->position.z, pose->orientation.x,
                    pose->orientation.y, pose->orientation.z, pose->orientation.w, seq - m_appswSeq - 1u));
            m_appswSeq = seq;
        }
        return ok;
    }
    // Motion vectors all zero (every pixel's motion is the camera's, given as appSpaceDeltaPose): cleared once per
    // swapchain image -- nothing ever writes them.
    void ClearMotionVectorImage(const XrSwapchainImageBaseHeader* image, int width, int height) override {
        (void)width; (void)height;
        const VkImage img = reinterpret_cast<const XrSwapchainImageVulkanKHR*>(image)->image;
        if (std::find(m_mvCleared.begin(), m_mvCleared.end(), img) != m_mvCleared.end()) return;
        if (m_mvCmd.buf == VK_NULL_HANDLE && !m_mvCmd.Init(m_namer, m_vkDevice, m_queueFamilyIndex)) return;
        m_mvCmd.Clear();
        m_mvCmd.Begin();
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = 0; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(m_mvCmd.buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(m_mvCmd.buf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        vkCmdPipelineBarrier(m_mvCmd.buf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        m_mvCmd.End();
        m_mvCmd.Exec(m_vkQueue);
        m_mvCmd.Wait();
        m_mvCleared.push_back(img);
        Log::Write(Log::Level::Info, Fmt("TCVR_APPSW motion-vector image cleared (%zu)", m_mvCleared.size()));
    }
    bool m_viewWroteDepth = true;
    // Arcade cadence (s22.cadence, default on): System 22 runs at 60 Hz. The display is set to 120 Hz and a
    // frame is drawn only when the game produced a new one, so every arcade frame is shown exactly twice
    // (no 2-1-2-1 judder on 90 Hz, and an alternating 30 Hz effect stays regular) and each drawn frame has
    // two refreshes of GPU time.
    bool RenderThisFrame() override {
        if (m_m2CadenceActive && !m_cadenceActive) {
            if (arcadexr::config::GetInt("m2.freeze", 0) != 0) return true;
            const tcvr_m2_frame* f = arcadexr::hardware::sega_model2::AcquireScene();
            ++m_framesSinceRender;
            const bool fresh = f && f->sequence != m_lastM2RenderedSeq;
            if (fresh || m_framesSinceRender >= 3) { m_framesSinceRender = 0; return true; }
            return false;
        }
        if (!m_cadenceActive) return true;
        namespace s22 = arcadexr::hardware::namco_system22;
        if (arcadexr::config::GetInt("s22.freeze", 0) != 0) return (++m_framesSinceRender & 1) == 0;   // bench: half rate
        const tcvr_scene_frame* f = s22::AcquireScene();
        if (f) m_s22Frame = f;
        ++m_framesSinceRender;
        const bool fresh = m_s22Frame && m_s22Frame->sequence != m_lastRenderedSeq;
        if (fresh || m_framesSinceRender >= 3) { m_framesSinceRender = 0; return true; }
        return false;
    }

    void ChainFoveationImages(ISwapchainImageData* images, uint32_t count) override {
        auto& v = m_fdmImages[images];
        v.assign(count, XrSwapchainImageFoveationVulkanFB{XR_TYPE_SWAPCHAIN_IMAGE_FOVEATION_VULKAN_FB});
        auto* arr = reinterpret_cast<XrSwapchainImageVulkanKHR*>(images->GetColorImageArray());
        for (uint32_t i = 0; i < count; ++i) arr[i].next = &v[i];
    }

    // Menu and pistol on top of the eye image (ports of RenderMenu / the lit gun of the GLES plugin).
    void RenderOverlay(VkCommandBuffer cmd, uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView,
                       VulkanSwapchainImageData* swapchainData, uint32_t imageIndex, const VkRect2D& renderArea) {
        auto& menu = arcadexr::ui::Menu::Get();
        const bool menuOpen = menu.IsOpen();
        auto guns = arcadexr::gun::GetGunPoses();
        const bool drawGuns = guns.count > 0 && !menuOpen && !arcadexr::profiles::IsDriving();
        const auto aim = arcadexr::gun::GetAimState();
        const bool drawReticle = aim.show_crosshair && aim.have_world && !menuOpen && !arcadexr::profiles::IsDriving();
        if (!menuOpen && !drawGuns && !drawReticle) { m_menuFramePoseValid = false; return; }
        if (!m_overlayInit) {
            m_overlay.Init(m_vkDevice, &m_memAllocator, VkFormat(swapchainData->GetSlices()[0].m_rp.colorFmt));
            m_overlayInit = true;
        }
        if (drawReticle) m_overlay.EnsureReticle(cmd);   // once, outside the render pass
        const auto& pose = layerView.pose;
        XrMatrix4x4f proj, toView, view, vp;
        XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_VULKAN, layerView.fov, 0.05f, 100.0f);
        XrMatrix4x4f_CreateFromRigidTransform(&toView, &pose);
        XrMatrix4x4f_InvertRigidBody(&view, &toView);
        XrMatrix4x4f_Multiply(&vp, &proj, &view);
        const VkExtent2D ext{uint32_t(swapchainData->Width()), uint32_t(swapchainData->Height())};
        m_overlay.Begin(cmd, viewIndex, swapchainData->GetTypedImage(imageIndex).image, ext, renderArea);
        if (drawGuns) {
            for (int g = 0; g < guns.count; ++g) {
                XrMatrix4x4f model, mvp;
                const float gs = std::max(0.5f, std::min(2.0f, arcadexr::config::GetFloat("gun.scale", 1.0f)));   // live size tuning
                const XrVector3f unit{gs, gs, gs};
                XrMatrix4x4f_CreateTranslationRotationScale(&model, &guns.pose[g].position, &guns.pose[g].orientation, &unit);
                XrMatrix4x4f_Multiply(&mvp, &vp, &model);
                const float eye[3] = {pose.position.x, pose.position.y, pose.position.z};
                m_overlay.DrawGun(cmd, mvp.m, model.m, eye);
            }
        }
        if (drawReticle) {
            // At the aimed point, facing this eye, a constant angular size (~1.6 degrees) whatever the distance.
            const XrVector3f P{aim.world[0], aim.world[1], aim.world[2]};
            XrVector3f toP{P.x - pose.position.x, P.y - pose.position.y, P.z - pose.position.z};
            float dist = std::sqrt(toP.x * toP.x + toP.y * toP.y + toP.z * toP.z);
            if (dist > 1e-3f) {
                const float useD = std::min(dist, 90.0f);   // inside the overlay's far plane
                const XrVector3f Q{pose.position.x + toP.x / dist * useD, pose.position.y + toP.y / dist * useD, pose.position.z + toP.z / dist * useD};
                const XrVector3f lr{1, 0, 0}, lu{0, 1, 0}, ln{0, 0, 1};
                XrVector3f right, up, normal;
                XrQuaternionf_RotateVector3f(&right, &pose.orientation, &lr);
                XrQuaternionf_RotateVector3f(&up, &pose.orientation, &lu);
                XrQuaternionf_RotateVector3f(&normal, &pose.orientation, &ln);
                const float size = useD * 0.028f * std::max(0.3f, std::min(3.0f, arcadexr::config::GetFloat("crosshair.size", 1.0f)));
                XrMatrix4x4f model{};
                model.m[0] = right.x * size; model.m[1] = right.y * size; model.m[2] = right.z * size;
                model.m[4] = up.x * size; model.m[5] = up.y * size; model.m[6] = up.z * size;
                model.m[8] = normal.x; model.m[9] = normal.y; model.m[10] = normal.z;
                model.m[12] = Q.x; model.m[13] = Q.y; model.m[14] = Q.z; model.m[15] = 1.0f;
                XrMatrix4x4f mvp; XrMatrix4x4f_Multiply(&mvp, &vp, &model);
                const float tint[4] = {aim.calibrating ? 0.3f : 1.0f, aim.calibrating ? 1.0f : 0.25f, aim.calibrating ? 0.3f : 0.2f, 1.0f};
                m_overlay.DrawReticle(cmd, mvp.m, tint);
            }
        }
        if (menuOpen) {
            // Anchored where the head was when the frame started (left eye), 1.4 m ahead, like the GLES path.
            if (viewIndex == 0 || !m_menuFramePoseValid) { m_menuFramePose = pose; m_menuFramePoseValid = true; }
            const XrPosef& anchor = m_menuFramePose;
            const XrVector3f lr{1, 0, 0}, lu{0, 1, 0}, ln{0, 0, 1};
            XrVector3f right, up, normal;
            XrQuaternionf_RotateVector3f(&right, &anchor.orientation, &lr);
            XrQuaternionf_RotateVector3f(&up, &anchor.orientation, &lu);
            XrQuaternionf_RotateVector3f(&normal, &anchor.orientation, &ln);
            const float width = 0.95f, height = width / m_overlay.MenuAspect();
            XrMatrix4x4f model{};
            model.m[0] = right.x * width; model.m[1] = right.y * width; model.m[2] = right.z * width;
            model.m[4] = up.x * height; model.m[5] = up.y * height; model.m[6] = up.z * height;
            model.m[8] = normal.x; model.m[9] = normal.y; model.m[10] = normal.z;
            model.m[12] = anchor.position.x - normal.x * 1.4f;
            model.m[13] = anchor.position.y - normal.y * 1.4f;
            model.m[14] = anchor.position.z - normal.z * 1.4f;
            model.m[15] = 1.0f;
            XrMatrix4x4f mvp;
            XrMatrix4x4f_Multiply(&mvp, &vp, &model);
            m_overlay.DrawMenu(cmd, mvp.m);
        }
        m_overlay.End(cmd);
    }

    // ---- Namco System 22 (Time Crisis, Dirt Dash) immersive, Vulkan module vulkan_s22_renderer.h ----
    // Port of RenderImmersiveArcadeScene of the GLES plugin: same anchor on the virtual screen, same
    // camera recovery, same settings (profiles immersive.*), same lightgun ray cast.
    void PrepareSystem22(VkCommandBuffer cmd, VulkanSwapchainImageData* swapchainData) {
        namespace s22 = arcadexr::hardware::namco_system22;
        m_s22Active = false;
        m_viewportScale = 1.0f;
        m_s22Prepared = false;
        const bool isS22 = arcadexr::profiles::IsSystem22() && s22::HaveSceneSource();
        // Vulkan: System 22 is ALWAYS drawn by the GPU module, flat or immersive. The stored "render=cpu /
        // scene.cpuRaster=1" choice is a GL-era menu option (MAME rasterising the flat screen): in Vulkan it made
        // Time Crisis lag (30+ ms of CPU raster per frame, audio underruns, 23/09). s22.cpuRaster=1 (debug,
        // the oracle) is the only way to have MAME rasterise alongside.
        const bool want = isS22 && arcadexr::profiles::GetString("presentation", "screen") == "immersive" &&
                          arcadexr::config::GetInt("s22.vk", 1) != 0;
        // SCREEN presentation drawn by the GPU too (s22.vkFlat=0 returns to MAME's CPU framebuffer).
        const bool wantFlat = isS22 && !want && arcadexr::profiles::GetString("presentation", "screen") != "immersive" &&
                              arcadexr::config::GetInt("s22.vkFlat", 1) != 0;
        m_s22FlatWanted = wantFlat;
        const int mode = (want || wantFlat) ? (arcadexr::config::GetInt("s22.cpuRaster", 0) ? 1 : 2) : 0;
        {   // Arcade cadence: 120 Hz display while a System 22 game is immersive, back to the baked rate after.
            const bool cadence = want && arcadexr::config::GetInt("s22.cadence", 1) != 0;
            if (cadence && !m_cadenceRequested) { arcadexr::xr::RequestRateForGame(120.0f, "System 22: 2 refreshes per 60 Hz frame"); m_cadenceRequested = true; }
            if (!cadence && m_cadenceRequested) { arcadexr::xr::RequestRateForGame(90.0f, "leaving System 22 immersive"); m_cadenceRequested = false; }
            m_cadenceActive = cadence;
        }
        if (isS22 && mode != m_s22SceneMode) {
            m_s22SceneMode = mode;
            s22::EnableScene(mode);
            Log::Write(Log::Level::Info, Fmt("TCVR_S22VK scene recording mode %d", mode));
        }
        if (!want && !wantFlat) return;
        if (!m_s22.Ready()) {
            const int msaa = std::max(1, std::min(4, arcadexr::config::GetInt("immersive.msaa", 4)));
            m_s22.Init(m_vkDevice, &m_memAllocator, VkFormat(swapchainData->GetSlices()[0].m_rp.colorFmt), msaa);
            arcadexr::gun::SetSceneAim(&S22AimTrampoline);   // board-agnostic: S22, then Model 2
            s_s22Self = this;
        }
        const std::string game = arcadexr::profiles::CurrentGame();
        if (game != m_s22Game) { m_s22.ResetAssets(); m_s22Game = game; }
        // Bench: s22.freeze=1 keeps redrawing the same scene (valid until the next acquire).
        if (!m_s22Frame || arcadexr::config::GetInt("s22.freeze", 0) == 0) {
            const tcvr_scene_frame* f = s22::AcquireScene();
            if (f) m_s22Frame = f;
        }
        if (!m_s22Frame || m_s22Frame->width <= 0 || m_s22Frame->height <= 0) return;
        if (!m_s22.AssetsReady()) {
            tcvr_scene_assets assets{};
            if (!s22::SceneAssets(assets) || !m_s22.UploadAssets(cmd, assets)) return;
        }
        m_s22.PumpBanks(cmd);
        m_s22.SetFilter(arcadexr::config::GetInt("s22.filter", 3));   // 0 texel-exact, 1 bilinear, 2 + index mipmaps, 3 sharp + anisotropic
        m_s22.SetHudSharp(arcadexr::config::GetInt("s22.hudSharp", 3));
        m_s22.SetLinearOut(arcadexr::config::GetInt("s22.linearOut", 1) != 0);   // true colours on an _SRGB eye image
        m_s22.SetTextDiag(arcadexr::config::GetInt("s22.textDiag", 0));
        {
            const int req = arcadexr::config::GetInt("s22.dumpPrims", 0);
            if (req != 0 && req != m_primDumpDone) { m_primDumpDone = req; m_s22.RequestPrimDump(); }
        }
        m_s22.SetDiagAlternating(arcadexr::config::GetInt("s22.diagAlt", 0) != 0);
        m_s22.SetAltFix(arcadexr::config::GetInt("s22.altFix", 0) != 0);   // off: the arcade flicker is kept (Guillaume 23/09)
        m_s22.SetAltMaxGroup(arcadexr::config::GetInt("s22.altMaxGroup", 4096));
        m_s22.SetReorder(arcadexr::config::GetInt("s22.reorder", 1) != 0);
        if (!m_s22.PrepareFrame(int(m_frameSlot), *m_s22Frame)) return;
        m_s22Prepared = true;
        if (!want) return;   // flat: drawn after the Model 2 flat block (RenderSystem22Flat)
        if ((arcadexr::config::GetInt("s22.skip", 0) & 8) == 0) m_s22.RenderDepthMap(cmd);
        m_s22Active = true;
        AimSelfTest();
        // Immersive System 22 renders its scene at immersive.scale: into a sub-rectangle of the swapchain
        // (applied from the next frame), so the composite also runs at that size and the compositor scales.
        m_lastRenderedSeq = m_s22Frame->sequence;
        if (arcadexr::config::GetInt("s22.viewportScale", 1) != 0)
            m_viewportScale = std::max(0.3f, std::min(1.0f, arcadexr::profiles::GetFloat("immersive.scale", 1.0f)));
    }

    bool RenderSystem22Eye(VkCommandBuffer cmd, uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView,
                           VulkanSwapchainImageData* swapchainData, uint32_t imageIndex, const VkRect2D& renderArea) {
        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) return false;
        const float distance = std::max(0.25f, arcadexr::config::GetFloat("screen.distance", 2.0f));
        const float depthUnits = std::max(100.0f, arcadexr::config::GetFloat("immersive.depth", 5000.0f));
        const float worldScale = distance / depthUnits;
        const arcadexr::gun::Vec3 camera{screen.center.x + screen.normal.x * distance, screen.center.y + screen.normal.y * distance,
                                         screen.center.z + screen.normal.z * distance};
        XrMatrix4x4f arcadeToWorld{};
        arcadeToWorld.m[0] = screen.right.x * worldScale; arcadeToWorld.m[1] = screen.right.y * worldScale; arcadeToWorld.m[2] = screen.right.z * worldScale;
        arcadeToWorld.m[4] = screen.up.x * worldScale; arcadeToWorld.m[5] = screen.up.y * worldScale; arcadeToWorld.m[6] = screen.up.z * worldScale;
        arcadeToWorld.m[8] = -screen.normal.x * worldScale; arcadeToWorld.m[9] = -screen.normal.y * worldScale; arcadeToWorld.m[10] = -screen.normal.z * worldScale;
        arcadeToWorld.m[12] = camera.x; arcadeToWorld.m[13] = camera.y; arcadeToWorld.m[14] = camera.z; arcadeToWorld.m[15] = 1.0f;
        m_s22AnchorCamera = camera; m_s22AnchorRight = screen.right; m_s22AnchorUp = screen.up; m_s22AnchorNormal = screen.normal;
        m_s22AnchorScale = worldScale; m_s22AnchorValid = true;
        const float farMetres = std::max(200.0f, arcadexr::config::GetFloat("immersive.far", 20000.0f));
        const float nearMetres = std::max(0.001f, std::min(0.05f, arcadexr::config::GetFloat("immersive.near", 0.005f)));
        XrMatrix4x4f projection, eyeToWorld, worldToEye, viewProjection, mvp;
        XrMatrix4x4f_CreateProjectionFov(&projection, GRAPHICS_VULKAN, layerView.fov, nearMetres, farMetres);
        XrMatrix4x4f_CreateFromRigidTransform(&eyeToWorld, &layerView.pose);
        XrMatrix4x4f_InvertRigidBody(&worldToEye, &eyeToWorld);
        XrMatrix4x4f_Multiply(&viewProjection, &projection, &worldToEye);
        XrMatrix4x4f_Multiply(&mvp, &viewProjection, &arcadeToWorld);
        XrMatrix4x4f hudToWorld{};
        hudToWorld.m[0] = screen.right.x * screen.width; hudToWorld.m[1] = screen.right.y * screen.width; hudToWorld.m[2] = screen.right.z * screen.width;
        hudToWorld.m[4] = screen.up.x * screen.height; hudToWorld.m[5] = screen.up.y * screen.height; hudToWorld.m[6] = screen.up.z * screen.height;
        hudToWorld.m[8] = screen.normal.x; hudToWorld.m[9] = screen.normal.y; hudToWorld.m[10] = screen.normal.z;
        hudToWorld.m[12] = screen.center.x; hudToWorld.m[13] = screen.center.y; hudToWorld.m[14] = screen.center.z; hudToWorld.m[15] = 1.0f;
        XrMatrix4x4f hudMvp;
        XrMatrix4x4f_Multiply(&hudMvp, &viewProjection, &hudToWorld);
        arcadexr::vulkan::VulkanSystem22Renderer::Settings st;
        st.depthTest = arcadexr::profiles::GetInt("immersive.depthTest", 1) != 0;
        st.lean = arcadexr::config::GetInt("s22.lean", 1);
        st.skip = arcadexr::config::GetInt("s22.skip", 0);
        st.darkFlicker = arcadexr::profiles::GetInt("temporal.darkFlicker", 0) != 0;
        st.darkMode = std::max(1, arcadexr::profiles::GetInt("temporal.darkFlicker", 0));
        st.merged = arcadexr::config::GetInt("s22.merged", 1) != 0;
        st.depthBias = arcadexr::config::GetFloat("s22.depthBiasRel", 8e-6f);   // validated by eye on Dirt Dash shadows (23/09)
        st.nearClip = nearMetres;
        {
            const std::string v = arcadexr::config::GetString("immersive.void", "game");
            unsigned r = 0, g = 0, b = 0;
            if (v == "game") st.voidMode = 0;
            else if (std::sscanf(v.c_str(), "%u,%u,%u", &r, &g, &b) == 3) { st.voidMode = 2; st.voidRGB[0] = r & 255; st.voidRGB[1] = g & 255; st.voidRGB[2] = b & 255; }
            else st.voidMode = 1;
        }
        const int texAa = arcadexr::profiles::GetInt("immersive.texAA", 1);
        st.texSamples = texAa <= 0 ? 1 : (texAa == 1 ? 4 : 16);
        st.spriteMinDepth = arcadexr::config::GetFloat("immersive.spriteMinDepth", 50.0f);
        const float scale = std::max(0.3f, std::min(2.0f, arcadexr::profiles::GetFloat("immersive.scale", 1.0f)));
        VkExtent2D ext{uint32_t(swapchainData->Width()), uint32_t(swapchainData->Height())};
        float renderScale = scale;
        if (renderArea.extent.width < ext.width) { ext = renderArea.extent; renderScale = 1.0f; }   // sub-rectangle: already scaled
        return m_s22.RenderEye(cmd, viewIndex, mvp.m, hudMvp.m, swapchainData->GetTypedImage(imageIndex).image, ext, renderArea, renderScale, st);
    }

    bool S22Aim(const XrVector3f& origin, const XrVector3f& direction, float& nx, float& ny, XrVector3f& hitWorld) {
        if (!m_s22Active || !m_s22AnchorValid || !m_s22Frame) return false;
        const auto dot = [](const XrVector3f& a, const arcadexr::gun::Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
        const XrVector3f rel{origin.x - m_s22AnchorCamera.x, origin.y - m_s22AnchorCamera.y, origin.z - m_s22AnchorCamera.z};
        const float o[3] = {dot(rel, m_s22AnchorRight) / m_s22AnchorScale, dot(rel, m_s22AnchorUp) / m_s22AnchorScale, -dot(rel, m_s22AnchorNormal) / m_s22AnchorScale};
        const float d[3] = {dot(direction, m_s22AnchorRight), dot(direction, m_s22AnchorUp), -dot(direction, m_s22AnchorNormal)};
        float sx = 0, sy = 0, hit[3] = {0, 0, 0};
        if (!m_s22.RayCast(o, d, m_s22Frame->width, m_s22Frame->height, sx, sy, hit)) return false;
        nx = sx / float(m_s22Frame->width);
        ny = sy / float(m_s22Frame->height);
        const float sc = m_s22AnchorScale;
        hitWorld = {m_s22AnchorCamera.x + (m_s22AnchorRight.x * hit[0] + m_s22AnchorUp.x * hit[1] - m_s22AnchorNormal.x * hit[2]) * sc,
                    m_s22AnchorCamera.y + (m_s22AnchorRight.y * hit[0] + m_s22AnchorUp.y * hit[1] - m_s22AnchorNormal.y * hit[2]) * sc,
                    m_s22AnchorCamera.z + (m_s22AnchorRight.z * hit[0] + m_s22AnchorUp.z * hit[1] - m_s22AnchorNormal.z * hit[2]) * sc};
        return true;
    }
    static inline VulkanGraphicsPlugin* s_s22Self = nullptr;
    // Aim self-test (s22.aimTest=1, logged every ~2 s as TCVR_AIMTEST): board pixels with a known depth are placed
    // in the room exactly as the renderer places them, a ray is cast at each from an origin, and the aim must
    // return the same pixel. From the eye (camera anchor) this checks the mapping; from a hand-like origin it also
    // shows occlusion (another surface nearer along that ray: then the game is RIGHT to get another pixel).
    void AimSelfTest() {
        if (!m_s22Active || !m_s22AnchorValid || !m_s22Frame || arcadexr::config::GetInt("s22.aimTest", 0) == 0) return;
        if ((++m_aimTestTick % 240u) != 0u) return;
        const int W = m_s22Frame->width, H = m_s22Frame->height;
        const float zoom = m_s22.Zoom(), sc = m_s22AnchorScale;
        const auto C = m_s22AnchorCamera, R = m_s22AnchorRight, U = m_s22AnchorUp, N = m_s22AnchorNormal;
        struct Origin { const char* name; XrVector3f o; };
        const Origin origins[2] = {
            {"oeil", {C.x, C.y, C.z}},
            {"main", {C.x + R.x * 0.15f - U.x * 0.35f - N.x * 0.30f, C.y + R.y * 0.15f - U.y * 0.35f - N.y * 0.30f,
                      C.z + R.z * 0.15f - U.z * 0.35f - N.z * 0.30f}}};
        for (const Origin& org : origins) {
            std::vector<float> errs;
            int misses = 0, occluded = 0;
            for (int gj = 0; gj < 9; ++gj)
                for (int gi = 0; gi < 12; ++gi) {
                    const float sx = (gi + 0.5f) / 12.0f * W, sy = (gj + 0.5f) / 9.0f * H;
                    const float d = m_s22.DepthAt(sx, sy, W, H);
                    if (d <= 0.0f) continue;
                    const float cx = (sx - W * 0.5f) * d / zoom, cy = (H * 0.5f - sy) * d / zoom, cz = d;
                    const XrVector3f P{C.x + (R.x * cx + U.x * cy - N.x * cz) * sc, C.y + (R.y * cx + U.y * cy - N.y * cz) * sc,
                                       C.z + (R.z * cx + U.z * cy - N.z * cz) * sc};
                    XrVector3f dir{P.x - org.o.x, P.y - org.o.y, P.z - org.o.z};
                    const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                    if (len < 1e-6f) continue;
                    dir = {dir.x / len, dir.y / len, dir.z / len};
                    float nx = 0, ny = 0; XrVector3f hw{};
                    if (!S22Aim(org.o, dir, nx, ny, hw)) { ++misses; continue; }
                    const float e = std::hypot(nx * W - sx, ny * H - sy);
                    const float hd = std::sqrt((hw.x - org.o.x) * (hw.x - org.o.x) + (hw.y - org.o.y) * (hw.y - org.o.y) + (hw.z - org.o.z) * (hw.z - org.o.z));
                    // Something nearer along this very ray: the game is RIGHT to get another pixel (not an error).
                    if (e > 1.5f && hd < 0.97f * len) { ++occluded; continue; }
                    errs.push_back(e);
                }
            std::sort(errs.begin(), errs.end());
            auto q = [&](float f) { return errs.empty() ? -1.0f : errs[std::min(errs.size() - 1, size_t(f * float(errs.size())))]; };
            Log::Write(Log::Level::Info, Fmt("TCVR_AIMTEST %s: %zu points, erreur px mediane %.2f p90 %.2f max %.2f | masques par plus pres %d, rates %d",
                                             org.name, errs.size(), q(0.5f), q(0.9f), errs.empty() ? -1.0f : errs.back(), occluded, misses));
        }
    }
    unsigned m_aimTestTick = 0;

    static bool S22AimTrampoline(const XrVector3f& o, const XrVector3f& d, float& nx, float& ny, XrVector3f& hit) {
        if (!s_s22Self) return false;
        arcadexr::gun::SetSceneAimOnPlane(false);
        if (s_s22Self->S22Aim(o, d, nx, ny, hit)) return true;
        const bool ok = s_s22Self->m_m2AimLive && s_s22Self->m_m2Renderer.Aim(o, d, nx, ny, hit);
        if (ok) arcadexr::gun::SetSceneAimOnPlane(s_s22Self->m_m2Renderer.AimOnPlane());
        return ok;
    }
    bool m_m2AimLive = false;
    bool m_m2CadenceRequested = false, m_m2CadenceActive = false;
    bool m_smoothOn = false;
    int m_m2RateRequested = 0;
    uint64_t m_lastM2RenderedSeq = 0;

    // The Model 1/2 scene path is only for a Model 1/2 game: the library's scene source outlives the game.
    static bool M2SceneLive() {
        return (arcadexr::profiles::IsModel2() || arcadexr::profiles::IsModel1()) &&
               arcadexr::hardware::sega_model2::HaveSceneSource();
    }

    // Everything the Model 2 module learnt about the previous game goes: a fresh object (built geometry,
    // texture shadows, regions, motion ids), no last frame, scene recording re-requested for the new game.
    void ResetM2ForGameChange() {
        if (m_vkDevice != VK_NULL_HANDLE) vkDeviceWaitIdle(m_vkDevice);
        m_m2Renderer.~VulkanModel2Renderer();
        new (&m_m2Renderer) arcadexr::vulkan::VulkanModel2Renderer();
        if (m_appswWanted) SetAppSwWanted(true);   // the fresh renderer must know it too (26/09)
        m_m2RendererInitialized = false;
        m_lastM2Frame = nullptr;
        m_lastM2Drawn = false;
        m_lastM2RenderedSeq = 0;
        m_m2DynScale = 1.0f;
        m_flatBoundView = VK_NULL_HANDLE;
        m_m2SceneRequested = false;
        if (m_m2SceneMode > 0 && !arcadexr::profiles::IsSystem22()) arcadexr::hardware::sega_model2::EnableScene(0);
        m_m2SceneMode = -1;
        Log::Write(Log::Level::Info, "TCVR_SWITCH Model 2 module reset for " + m_m2Game);
    }
    std::string m_m2Game;
    uint32_t m_paceNew = 0, m_paceSkipped = 0, m_paceHold = 0, m_paceHist[5] = {};
    std::chrono::steady_clock::time_point m_paceAt{};

    void EnsureM2Renderer(VulkanSwapchainImageData* swapchainData) {
        if (m_m2RendererInitialized) return;
        const VkFormat fmt = VkFormat(swapchainData->GetSlices()[0].m_rp.colorFmt);
        const int msaa = std::max(1, std::min(4, arcadexr::config::GetInt("m2.msaa", 4)));
        // Foveation on this swapchain? (it was created with a density map iff foveation > 0)
        const bool useFdm = m_fdmEnabled && arcadexr::config::GetInt("foveation", 0) > 0 && !m_fdmImages.empty();
        m_m2Renderer.Initialize(m_vkDevice, &m_memAllocator, fmt, uint32_t(msaa), useFdm);
        m_m2RendererInitialized = true;
    }

    void WriteDump() {
        m_dumpState = 0;
        m_dumpDoneTag = m_dumpTag;
        void* p = nullptr;
        if (vkMapMemory(m_vkDevice, m_dumpMem, 0, m_dumpSize, 0, &p) != VK_SUCCESS) return;
        const std::string dir = arcadexr::config::ExternalDirectory();
        const std::string path = (dir.empty() ? std::string("/sdcard/Android/data/io.tcvr2.prototype.vulkan/files") : dir) +
                                 "/dump-" + m_dumpTag + (m_dumpTag.size() > 2 && m_dumpTag.compare(m_dumpTag.size() - 2, 2, "_R") == 0 ? "-vkR.ppm" : "-vkL.ppm");
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) {
            const uint32_t outW = m_dumpVisW ? m_dumpVisW : m_dumpW, outH = m_dumpVisH ? m_dumpVisH : m_dumpH;
            std::fprintf(f, "P6\n%u %u\n255\n", outW, outH);
            const uint8_t* px = static_cast<const uint8_t*>(p);
            std::vector<uint8_t> row(size_t(outW) * 3);
            for (uint32_t y = 0; y < outH; ++y) {
                for (uint32_t x = 0; x < outW; ++x) {
                    const uint8_t* s = px + (size_t(y) * m_dumpW + x) * 4;
                    // swapchain may be BGRA or RGBA sRGB: written raw, channel order logged
                    row[x * 3 + 0] = s[0]; row[x * 3 + 1] = s[1]; row[x * 3 + 2] = s[2];
                }
                std::fwrite(row.data(), 1, row.size(), f);
            }
            std::fclose(f);
        }
        vkUnmapMemory(m_vkDevice, m_dumpMem);
        Log::Write(Log::Level::Info, Fmt("TCVR_DUMP wrote %s (%ux%u shown of %ux%u) interp=%.3f %s", path.c_str(), m_dumpVisW, m_dumpVisH, m_dumpW, m_dumpH, m_dumpInterp, f ? "ok" : "FAILED"));
    }

    // ---- Oracle capture (debug.tcvr.oracle=<tag>), for scripts/port_oracle.py ------------------------
    // With the scene recorded ALONGSIDE MAME's CPU raster (scene mode 1), the same arcade frame exists
    // twice: MAME's own rasterised image (the reference) and our GPU module's flat image. For
    // kOracleFrames consecutive frames we save both, unpaired: MAME publishes its image with a lag
    // (asynchronous raster), so the host script pairs them by content. Raw files, a descriptor listing
    // what was written and what was not; no verdict here.
    static constexpr int kOracleFrames = 16;
    struct OracleCap { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; uint32_t cb = 0, w = 0, h = 0, idx = 0; };
    std::vector<OracleCap> m_oraclePending;
    std::string m_oracleTag, m_oracleDoneTag;
    int m_oracleNext = 0;
    FILE* m_oracleDesc = nullptr;
    std::string OracleDir() const {
        const std::string dir = arcadexr::config::ExternalDirectory();
        return dir.empty() ? std::string("/sdcard/Android/data/io.tcvr2.prototype.vulkan/files") : dir;
    }
    void RecordOracle(VkCommandBuffer cmd, uint32_t cb) {
        if (m_oracleTag.empty()) {
            const std::string tag = arcadexr::config::GetString("oracle", "0");
            if (tag.empty() || tag == "0" || tag == m_oracleDoneTag) return;
            m_oracleTag = tag; m_oracleNext = 0;
            m_oracleDesc = std::fopen((OracleDir() + "/oracle-" + tag + ".txt").c_str(), "w");
        }
        const bool s22 = m_s22FlatWanted && m_s22.FlatImage() != VK_NULL_HANDLE;
        const VkImage img = s22 ? m_s22.FlatImage() : m_m2Renderer.FlatImage();
        const VkFormat fmt = s22 ? m_s22.FlatFormat() : m_m2Renderer.FlatFormat();
        if (img == VK_NULL_HANDLE || m_flatW == 0) return;
        OracleCap c; c.cb = cb; c.w = m_flatW; c.h = m_flatH; c.idx = uint32_t(m_oracleNext);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = VkDeviceSize(c.w) * c.h * 4; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(m_vkDevice, &bi, nullptr, &c.buf) != VK_SUCCESS) return;
        VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(m_vkDevice, c.buf, &req);
        m_memAllocator.Allocate(req, &c.mem);
        vkBindBufferMemory(m_vkDevice, c.buf, c.mem, 0);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; r.imageExtent = {c.w, c.h, 1};
        vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.buf, 1, &r);
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        m_oraclePending.push_back(c);
        if (m_oracleDesc) std::fprintf(m_oracleDesc, "gpu %u %u %u fmt=%d board=%s\n", c.idx, c.w, c.h, int(fmt), s22 ? "s22" : "m2");
        // MAME's latest rasterised frame at the same moment (0x00RRGGBB words).
        arcadexr::video::FrameInfo got;
        const std::uint32_t* px = arcadexr::video::AcquireLatestFrame(got);
        if (px && got.width > 0 && got.height > 0) {
            FILE* f = std::fopen((OracleDir() + Fmt("/oracle-%s-c%u.ppm", m_oracleTag.c_str(), c.idx)).c_str(), "wb");
            if (f) {
                std::fprintf(f, "P6\n%d %d\n255\n", got.width, got.height);
                std::vector<uint8_t> row(size_t(got.width) * 3);
                for (int y = 0; y < got.height; ++y) {
                    const std::uint32_t* s = px + size_t(y) * got.stride;
                    for (int x = 0; x < got.width; ++x) { row[x * 3] = uint8_t(s[x] >> 16); row[x * 3 + 1] = uint8_t(s[x] >> 8); row[x * 3 + 2] = uint8_t(s[x]); }
                    std::fwrite(row.data(), 1, row.size(), f);
                }
                std::fclose(f);
            }
            if (m_oracleDesc) std::fprintf(m_oracleDesc, "cpu %u %d %d video_seq=%llu\n", c.idx, got.width, got.height, (unsigned long long)got.sequence);
        } else if (m_oracleDesc) {
            std::fprintf(m_oracleDesc, "cpu %u missing (no MAME framebuffer: is the scene recorded in mode 1?)\n", c.idx);
        }
        if (++m_oracleNext >= kOracleFrames) { m_oracleDoneTag = m_oracleTag; m_oracleTag.clear(); }
    }
    void FlushOracle() {
        for (auto it = m_oraclePending.begin(); it != m_oraclePending.end();) {
            if (m_cmdBuffer[it->cb].state == CmdBuffer::CmdBufferState::Executing) { ++it; continue; }
            void* p = nullptr;
            if (vkMapMemory(m_vkDevice, it->mem, 0, VkDeviceSize(it->w) * it->h * 4, 0, &p) == VK_SUCCESS) {
                const std::string name = m_oracleDoneTag.empty() ? m_oracleTag : m_oracleDoneTag;
                FILE* f = std::fopen((OracleDir() + Fmt("/oracle-%s-g%u.rgba", (m_oracleTag.empty() ? m_oracleDoneTag : m_oracleTag).c_str(), it->idx)).c_str(), "wb");
                (void)name;
                if (f) { std::fwrite(p, 1, size_t(it->w) * it->h * 4, f); std::fclose(f); }
                vkUnmapMemory(m_vkDevice, it->mem);
            }
            vkDestroyBuffer(m_vkDevice, it->buf, nullptr); vkFreeMemory(m_vkDevice, it->mem, nullptr);
            it = m_oraclePending.erase(it);
        }
        if (m_oraclePending.empty() && m_oracleTag.empty() && m_oracleDesc) {
            std::fprintf(m_oracleDesc, "done\n");
            std::fclose(m_oracleDesc); m_oracleDesc = nullptr;
            Log::Write(Log::Level::Info, "TCVR_ORACLE done " + m_oracleDoneTag);
        }
    }

    void SetM2SceneMode(int mode) {
        // A System 22 game is the System 22 module's business (separate MAME switch, tcvr_mame_scene_enable);
        // the Model 2 path has nothing to record then.
        if (arcadexr::profiles::IsSystem22()) return;
        if (mode == m_m2SceneMode) return;
        m_m2SceneMode = mode;
        arcadexr::hardware::sega_model2::EnableScene(mode);
        Log::Write(Log::Level::Info, Fmt("TCVR_M2VK scene recording mode %d (%s)", mode,
                                         mode >= 2 ? "GPU draws it, MAME does not rasterise" : "alongside MAME's CPU raster"));
    }

    // GPU time per eye, from Vulkan timestamps around each eye's command buffer (not
    // inside the render pass: on a tiler that would only time the binning). Read one
    // frame late, after the fence, so it never stalls. Logged as TCVR_VKGPU once a
    // second: median/max of the per-frame sum of both eyes, in ms.
    void CreateGpuQueryPool() {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &props);
        m_timestampPeriodNs = props.limits.timestampPeriod;
        if (props.limits.timestampComputeAndGraphics == VK_FALSE || m_timestampPeriodNs <= 0.0f) {
            m_gpuQueryPool = VK_NULL_HANDLE;
            return;
        }
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 8;
        if (vkCreateQueryPool(m_vkDevice, &qi, nullptr, &m_gpuQueryPool) != VK_SUCCESS) m_gpuQueryPool = VK_NULL_HANDLE;
    }

    // Dynamic resolution for Model 1/2 immersive (25/09). The demo ran at 120 fps, a real race did not: Sega Rally
    // 58/90 with the GPU at 17 ms for both eyes, Super GT 3-34 missed frames a second at 13 ms for an 8.3 ms
    // budget. A frozen race scene measured 7 ms with NO polygon at all: the cost is the fill of the 3360x3520
    // eye (2x the runtime's recommended size, 2.6x the native pixels), not a setting. So the rendered area
    // follows the measured GPU time: down fast when a refresh would be missed, up slowly when there is room.
    // Never below m2.dynresMin (0.5 = the recommended size, i.e. no supersampling). m2.dynres=0 turns it off,
    // m2.viewportScale forces a fixed scale for A/B. Logged as TCVR_DYNRES when it moves.
    void UpdateM2DynamicScale(float gpuMs) {
        // Sega Rally is the validated GOLD (23/09, judged perfect in the headset by Guillaume): never touched.
        const bool on = arcadexr::config::GetInt("m2.dynres", 1) != 0 &&
                        arcadexr::profiles::GetInt("immersive.dynres", arcadexr::profiles::CurrentGame() == "srallyc" ? 0 : 1) != 0;
        if (!on || !m_lastM2Drawn || m_s22Active) {
            m_m2DynScale = 1.0f; m_dynSamples.clear(); return;
        }
        m_dynSamples.push_back(gpuMs);
        const auto now = std::chrono::steady_clock::now();
        if (now - m_dynAt < std::chrono::milliseconds(400) || m_dynSamples.size() < 8) return;
        m_dynAt = now;
        std::vector<float> v = m_dynSamples;
        m_dynSamples.clear();
        std::sort(v.begin(), v.end());
        const float p75 = v[(v.size() * 3) / 4];
        const float hz = arcadexr::xr::State().current > 1.0f ? arcadexr::xr::State().current : 90.0f;
        // Arcade cadence (one draw per new arcade frame, shown on two refreshes): a draw has two refresh periods, not
        // one (25/09: Top Skater at 120 Hz was held at 0.58 against a 7 ms budget it did not have to meet).
        const float period = (1000.0f / hz) * (m_m2CadenceActive ? 2.0f : 1.0f);
        const float target = period * std::max(0.5f, std::min(1.0f, arcadexr::config::GetFloat("m2.dynresBudget", 0.85f)));
        const float lo = std::max(0.3f, std::min(1.0f, arcadexr::config::GetFloat("m2.dynresMin", 0.5f)));
        // GPU time is roughly fixed + area; area goes with scale squared.
        float want = m_m2DynScale * std::sqrt(target / std::max(0.5f, p75));
        float next = m_m2DynScale;
        if (p75 > target) next = std::max(want, m_m2DynScale - 0.10f);                 // over budget: act now
        else if (p75 < target * 0.80f) next = std::min(want, m_m2DynScale + 0.03f);    // clear room: creep up
        next = std::max(lo, std::min(1.0f, next));
        if (std::fabs(next - m_m2DynScale) >= 0.01f) {
            Log::Write(Log::Level::Info, Fmt("TCVR_DYNRES gpu p75=%.2f ms target=%.2f (%.0f Hz%s) scale %.2f -> %.2f",
                                             p75, target, hz, m_m2CadenceActive ? ", cadence" : "", m_m2DynScale, next));
            m_m2DynScale = next;
        }
    }
    float m_m2DynScale{1.0f};
    std::vector<float> m_dynSamples;
    std::chrono::steady_clock::time_point m_dynAt{};

    void ReadGpuTimestamps() {
        const uint32_t s0 = m_frameSlot * 2;   // the slot just fenced: its two eyes' queries
        if (m_gpuQueryPool == VK_NULL_HANDLE || !m_gpuQueryWritten[s0] || !m_gpuQueryWritten[s0 + 1]) return;
        uint64_t ts[4] = {};
        if (vkGetQueryPoolResults(m_vkDevice, m_gpuQueryPool, s0 * 2, 4, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return;
        const double ms = (double(ts[1] - ts[0]) + double(ts[3] - ts[2])) * m_timestampPeriodNs * 1e-6;
        m_gpuSamples.push_back(float(ms));
        UpdateM2DynamicScale(float(ms));
        const auto now = std::chrono::steady_clock::now();
        if (now - m_gpuLogAt >= std::chrono::seconds(1) && !m_gpuSamples.empty()) {
            std::vector<float> s = m_gpuSamples;
            std::sort(s.begin(), s.end());
            const float med = s[s.size() / 2], p90 = s[(s.size() * 9) / 10], mx = s.back();
            Log::Write(Log::Level::Info, Fmt("TCVR_VKGPU frames=%zu gpuMs med=%.2f p90=%.2f max=%.2f (both eyes) mode=%d",
                                             s.size(), med, p90, mx, m_m2SceneMode));
            m_gpuSamples.clear();
            m_gpuLogAt = now;
        }
    }

    uint32_t GetSupportedSwapchainSampleCount(const XrViewConfigurationView&) override { return VK_SAMPLE_COUNT_1_BIT; }

    void SetClearColor(const std::array<float, 4> clearColor) override { m_clearColor = clearColor; }

    ~VulkanGraphicsPlugin() override {
        if (m_vkDevice != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_vkDevice);
            m_m2Renderer.Cleanup();
            m_overlay.Destroy();
            m_s22.Destroy();
            if (m_gpuQueryPool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(m_vkDevice, m_gpuQueryPool, nullptr);
                m_gpuQueryPool = VK_NULL_HANDLE;
            }
            if (m_screenPipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_vkDevice, m_screenPipeline, nullptr);
                m_screenPipeline = VK_NULL_HANDLE;
            }
            if (m_screenPipelineLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_vkDevice, m_screenPipelineLayout, nullptr);
                m_screenPipelineLayout = VK_NULL_HANDLE;
            }
            if (m_screenDescriptorPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(m_vkDevice, m_screenDescriptorPool, nullptr);
                m_screenDescriptorPool = VK_NULL_HANDLE;
            }
            if (m_screenDescriptorSetLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_vkDevice, m_screenDescriptorSetLayout, nullptr);
                m_screenDescriptorSetLayout = VK_NULL_HANDLE;
            }
            if (m_flatSampler != VK_NULL_HANDLE) { vkDestroySampler(m_vkDevice, m_flatSampler, nullptr); m_flatSampler = VK_NULL_HANDLE; }
            if (m_screenSampler != VK_NULL_HANDLE) {
                vkDestroySampler(m_vkDevice, m_screenSampler, nullptr);
                m_screenSampler = VK_NULL_HANDLE;
            }
            if (m_screenImageView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_vkDevice, m_screenImageView, nullptr);
                m_screenImageView = VK_NULL_HANDLE;
            }
            if (m_screenImage != VK_NULL_HANDLE) {
                vkDestroyImage(m_vkDevice, m_screenImage, nullptr);
                m_screenImage = VK_NULL_HANDLE;
            }
            if (m_screenImageMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_vkDevice, m_screenImageMemory, nullptr);
                m_screenImageMemory = VK_NULL_HANDLE;
            }
            if (m_stagingBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(m_vkDevice, m_stagingBuffer, nullptr);
                m_stagingBuffer = VK_NULL_HANDLE;
            }
            if (m_stagingBufferMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_vkDevice, m_stagingBufferMemory, nullptr);
                m_stagingBufferMemory = VK_NULL_HANDLE;
            }
        }
    }

   protected:
    XrGraphicsBindingVulkan2KHR m_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    SwapchainImageDataMap<VulkanSwapchainImageData> m_swapchainImageDataMap;

    VkInstance m_vkInstance{VK_NULL_HANDLE};
    VkPhysicalDevice m_vkPhysicalDevice{VK_NULL_HANDLE};
    VkDevice m_vkDevice{VK_NULL_HANDLE};
    VulkanDebugObjectNamer m_namer{};
    uint32_t m_queueFamilyIndex = 0;
    VkQueue m_vkQueue{VK_NULL_HANDLE};
    VkSemaphore m_vkDrawDone{VK_NULL_HANDLE};

    MemoryAllocator m_memAllocator{};
    ShaderProgram m_shaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
    ShaderProgram m_computeShaderProgram{SHADER_PROGRAM_TYPE_COMPUTE};
    // [frame slot * 2 + eye]: two frames in flight, so the CPU records frame N+1 while the GPU
    // still draws frame N (measured 22/09: waiting on the fence of the frame just submitted left
    // the GPU idle during the CPU preparation -> 12-14 ms periods at 90 Hz).
    CmdBuffer m_cmdBuffer[4]{};
    CmdBuffer m_mvCmd{};
    bool m_haveDepthResolve = false, m_viewWroteSwDepth = false, m_appswWanted = false;
    struct SwDepthTarget { VkImage image = VK_NULL_HANDLE; VkExtent2D ext{0, 0}; VkFormat format = VK_FORMAT_UNDEFINED; };
    SwDepthTarget m_swDepthTarget[2]{};
    int64_t m_xrDepthFormat = -1;
    std::vector<VkImage> m_mvCleared;
    uint32_t m_appswSeq = 0, m_appswLogTick = 0;
    uint32_t m_frameSlot{0};
    PipelineLayout m_pipelineLayout{};
    VertexBuffer<Geometry::Vertex> m_drawBuffer{};
    std::array<float, 4> m_clearColor;

    ShaderProgram m_screenShaderProgram{SHADER_PROGRAM_TYPE_GRAPHICS};
    VkDescriptorSetLayout m_screenDescriptorSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout m_screenPipelineLayout{VK_NULL_HANDLE};
    VkDescriptorPool m_screenDescriptorPool{VK_NULL_HANDLE};
    VkDescriptorSet m_screenDescriptorSet{VK_NULL_HANDLE};
    VkSampler m_screenSampler{VK_NULL_HANDLE};
    VkPipeline m_screenPipeline{VK_NULL_HANDLE};
    VertexBuffer<ScreenVertex> m_screenDrawBuffer{};

    VkImage m_screenImage{VK_NULL_HANDLE};
    VkDeviceMemory m_screenImageMemory{VK_NULL_HANDLE};
    VkImageView m_screenImageView{VK_NULL_HANDLE};
    VkImageLayout m_screenImageLayout{VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t m_screenWidth{0};
    uint32_t m_screenHeight{0};
    uint64_t m_lastScreenSequence{0};
    bool m_loggedScreenUpload{false};

    VkBuffer m_stagingBuffer{VK_NULL_HANDLE};
    VkDeviceMemory m_stagingBufferMemory{VK_NULL_HANDLE};
    VkDeviceSize m_stagingBufferSize{0};

    arcadexr::vulkan::VulkanModel2Renderer m_m2Renderer;
    bool m_m2RendererInitialized{false};
    bool m_m2SceneRequested{false};
    int m_m2SceneMode{-1};
    bool m_fdmEnabled{false};
    bool m_lastM2Drawn{false};
    arcadexr::vulkan::VulkanSystem22Renderer m_s22;
    bool m_s22Active{false};
    int m_s22SceneMode{-1};
    std::string m_s22Game;
    const tcvr_scene_frame* m_s22Frame{nullptr};
    arcadexr::gun::Vec3 m_s22AnchorCamera{}, m_s22AnchorRight{}, m_s22AnchorUp{}, m_s22AnchorNormal{};
    float m_s22AnchorScale{1.0f};
    bool m_s22AnchorValid{false};
    bool m_flatDrawn{false};
    VkDescriptorSet m_flatDescriptorSet{VK_NULL_HANDLE};
    VkSampler m_flatSampler{VK_NULL_HANDLE};
    VkImageView m_flatBoundView{VK_NULL_HANDLE};
    uint32_t m_flatW{0}, m_flatH{0};
    float m_viewportScale{1.0f};
    bool m_cadenceActive{false}, m_cadenceRequested{false};
    uint64_t m_lastRenderedSeq{~0ull};
    int m_framesSinceRender{0};
    int m_primDumpDone{0};
    bool m_s22FlatWanted{false}, m_s22Prepared{false};
    arcadexr::vulkan::VulkanOverlay m_overlay;
    bool m_overlayInit{false};
    XrPosef m_menuFramePose{};
    bool m_menuFramePoseValid{false};
    int m_dumpState{0};
    std::string m_dumpTag, m_dumpDoneTag;
    uint32_t m_dumpVisW = 0, m_dumpVisH = 0;
    float m_dumpInterp = 1.0f;   // blend position of the dumped frame (smooth motion)
    uint32_t m_dumpW{0}, m_dumpH{0}, m_dumpCb{0};
    VkBuffer m_dumpBuf{VK_NULL_HANDLE};
    VkDeviceMemory m_dumpMem{VK_NULL_HANDLE};
    VkDeviceSize m_dumpSize{0};
    const tcvr_m2_frame* m_lastM2Frame{nullptr};
    float m_cpuWaitMs{0}, m_cpuPrepMs{0}, m_cpuSubmitMs{0}, m_cpuViewMs{0}, m_cpuPeriodMs{0};
    uint32_t m_cpuFrames{0};
    std::chrono::steady_clock::time_point m_cpuLastFrame{}, m_cpuLogAt{};
    VkPhysicalDeviceFragmentDensityMapFeaturesEXT m_fdmFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT};
    std::map<const ISwapchainImageData*, std::vector<XrSwapchainImageFoveationVulkanFB>> m_fdmImages;
    VkQueryPool m_gpuQueryPool{VK_NULL_HANDLE};
    bool m_gpuQueryWritten[4]{false, false, false, false};
    float m_timestampPeriodNs{0.0f};
    std::vector<float> m_gpuSamples;
    std::chrono::steady_clock::time_point m_gpuLogAt{};

    PipelineLayout m_computePipelineLayout{};
    VkDescriptorSet m_ComputeDescriptorSet;

#if defined(USE_MIRROR_WINDOW)
    Swapchain m_swapchain{};
#endif

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT{nullptr};
    VkDebugUtilsMessengerEXT m_vkDebugUtilsMessenger{VK_NULL_HANDLE};

    static std::string vkObjectTypeToString(VkObjectType objectType) {
        std::string objName;

#define LIST_OBJECT_TYPES(_)          \
    _(UNKNOWN)                        \
    _(INSTANCE)                       \
    _(PHYSICAL_DEVICE)                \
    _(DEVICE)                         \
    _(QUEUE)                          \
    _(SEMAPHORE)                      \
    _(COMMAND_BUFFER)                 \
    _(FENCE)                          \
    _(DEVICE_MEMORY)                  \
    _(BUFFER)                         \
    _(IMAGE)                          \
    _(EVENT)                          \
    _(QUERY_POOL)                     \
    _(BUFFER_VIEW)                    \
    _(IMAGE_VIEW)                     \
    _(SHADER_MODULE)                  \
    _(PIPELINE_CACHE)                 \
    _(PIPELINE_LAYOUT)                \
    _(RENDER_PASS)                    \
    _(PIPELINE)                       \
    _(DESCRIPTOR_SET_LAYOUT)          \
    _(SAMPLER)                        \
    _(DESCRIPTOR_POOL)                \
    _(DESCRIPTOR_SET)                 \
    _(FRAMEBUFFER)                    \
    _(COMMAND_POOL)                   \
    _(SURFACE_KHR)                    \
    _(SWAPCHAIN_KHR)                  \
    _(DISPLAY_KHR)                    \
    _(DISPLAY_MODE_KHR)               \
    _(DESCRIPTOR_UPDATE_TEMPLATE_KHR) \
    _(DEBUG_UTILS_MESSENGER_EXT)

        switch (objectType) {
            default:
#define MK_OBJECT_TYPE_CASE(name) \
    case VK_OBJECT_TYPE_##name:   \
        objName = #name;          \
        break;
                LIST_OBJECT_TYPES(MK_OBJECT_TYPE_CASE)
        }

        return objName;
    }
    VkBool32 debugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData) {
        std::string flagNames;
        std::string objName;
        Log::Level level = Log::Level::Error;

        if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) != 0u) {
            flagNames += "DEBUG:";
            level = Log::Level::Verbose;
        }
        if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) != 0u) {
            flagNames += "INFO:";
            level = Log::Level::Info;
        }
        if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u) {
            flagNames += "WARN:";
            level = Log::Level::Warning;
        }
        if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u) {
            flagNames += "ERROR:";
            level = Log::Level::Error;
        }
        if ((messageTypes & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) != 0u) {
            flagNames += "PERF:";
            level = Log::Level::Warning;
        }

        uint64_t object = 0;
        // skip loader messages about device extensions
        if (pCallbackData->objectCount > 0) {
            auto objectType = pCallbackData->pObjects[0].objectType;
            if ((objectType == VK_OBJECT_TYPE_INSTANCE) && (strncmp(pCallbackData->pMessage, "Device Extension:", 17) == 0)) {
                return VK_FALSE;
            }
            objName = vkObjectTypeToString(objectType);
            object = pCallbackData->pObjects[0].objectHandle;
            if (pCallbackData->pObjects[0].pObjectName != nullptr) {
                objName += " " + std::string(pCallbackData->pObjects[0].pObjectName);
            }
        }

        // A driver hint repeated on every render pass (Adreno VKDBGUTILWARN003: 120 lines a second) drowned every
        // other log line. Each distinct message is written once, then its repetition count every 10 000.
        {
            static std::unordered_map<std::string, uint64_t> seen;
            static std::mutex seenMutex;
            std::lock_guard<std::mutex> lock(seenMutex);
            const std::string key = std::string(pCallbackData->pMessageIdName ? pCallbackData->pMessageIdName : "") + "|" +
                                    std::string(pCallbackData->pMessage ? pCallbackData->pMessage : "").substr(0, 160);
            const uint64_t n = ++seen[key];
            if (n > 1) {
                if (n % 10000 == 0) Log::Write(level, Fmt("%s (x%llu so far) %s", flagNames.c_str(), (unsigned long long)n, key.c_str()));
                return VK_FALSE;
            }
        }
        Log::Write(level, Fmt("%s (%s 0x%llx) %s", flagNames.c_str(), objName.c_str(), object, pCallbackData->pMessage));

        return VK_FALSE;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageThunk(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                            void* pUserData) {
        return static_cast<VulkanGraphicsPlugin*>(pUserData)->debugMessage(messageSeverity, messageTypes, pCallbackData);
    }

    virtual XrStructureType GetGraphicsBindingType() const { return XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR; }
    virtual XrStructureType GetSwapchainImageType() const { return XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR; }

    virtual XrResult CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo,
                                             VkInstance* vulkanInstance, VkResult* vulkanResult) {
        PFN_xrCreateVulkanInstanceKHR pfnCreateVulkanInstanceKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrCreateVulkanInstanceKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateVulkanInstanceKHR)));

        return pfnCreateVulkanInstanceKHR(instance, createInfo, vulkanInstance, vulkanResult);
    }

    virtual XrResult CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo,
                                           VkDevice* vulkanDevice, VkResult* vulkanResult) {
        PFN_xrCreateVulkanDeviceKHR pfnCreateVulkanDeviceKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrCreateVulkanDeviceKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateVulkanDeviceKHR)));

        return pfnCreateVulkanDeviceKHR(instance, createInfo, vulkanDevice, vulkanResult);
    }

    virtual XrResult GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                                 VkPhysicalDevice* vulkanPhysicalDevice) {
        PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2KHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDevice2KHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsDevice2KHR)));

        return pfnGetVulkanGraphicsDevice2KHR(instance, getInfo, vulkanPhysicalDevice);
    }

    virtual XrResult GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                       XrGraphicsRequirementsVulkan2KHR* graphicsRequirements) {
        PFN_xrGetVulkanGraphicsRequirements2KHR pfnGetVulkanGraphicsRequirements2KHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirements2KHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsRequirements2KHR)));

        return pfnGetVulkanGraphicsRequirements2KHR(instance, systemId, graphicsRequirements);
    }
};

// A compatibility class that implements the KHR_vulkan_enable2 functionality on top of KHR_vulkan_enable
struct VulkanGraphicsPluginLegacy : public VulkanGraphicsPlugin {
    VulkanGraphicsPluginLegacy() : VulkanGraphicsPlugin() { m_graphicsBinding.type = GetGraphicsBindingType(); };

    std::vector<std::string> GetInstanceExtensions() const override { return {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME}; }
    virtual XrStructureType GetGraphicsBindingType() const override { return XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR; }
    virtual XrStructureType GetSwapchainImageType() const override { return XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR; }

    static void LogVulkanExtensions(const std::string title, const std::vector<const char*>& extensions, unsigned int start = 0) {
        const std::string indentStr(1, ' ');

        Log::Write(Log::Level::Verbose, Fmt("%s: (%d)", title.c_str(), extensions.size() - start));
        for (auto ext : extensions) {
            if (start) {
                start--;
                continue;
            }
            Log::Write(Log::Level::Verbose, Fmt("%s  Name=%s", indentStr.c_str(), ext));
        }
    }

    virtual XrResult CreateVulkanInstanceKHR(XrInstance instance, const XrVulkanInstanceCreateInfoKHR* createInfo,
                                             VkInstance* vulkanInstance, VkResult* vulkanResult) override {
        PFN_xrGetVulkanInstanceExtensionsKHR pfnGetVulkanInstanceExtensionsKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanInstanceExtensionsKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanInstanceExtensionsKHR)));

        uint32_t extensionNamesSize = 0;
        CHECK_XRCMD(pfnGetVulkanInstanceExtensionsKHR(instance, createInfo->systemId, 0, &extensionNamesSize, nullptr));

        std::vector<char> extensionNames(extensionNamesSize);
        CHECK_XRCMD(pfnGetVulkanInstanceExtensionsKHR(instance, createInfo->systemId, extensionNamesSize, &extensionNamesSize,
                                                      &extensionNames[0]));
        {
            // Note: This cannot outlive the extensionNames above, since it's just a collection of views into that string!
            std::vector<const char*> extensions = ParseExtensionString(&extensionNames[0]);
            LogVulkanExtensions("Vulkan Instance Extensions, requested by runtime", extensions);

            // Merge the runtime's request with the applications requests
            for (uint32_t i = 0; i < createInfo->vulkanCreateInfo->enabledExtensionCount; ++i) {
                extensions.push_back(createInfo->vulkanCreateInfo->ppEnabledExtensionNames[i]);
            }
            LogVulkanExtensions("Vulkan Instance Extensions, requested by application", extensions,
                                (uint32_t)extensions.size() - createInfo->vulkanCreateInfo->enabledExtensionCount);

            VkInstanceCreateInfo instInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            memcpy(&instInfo, createInfo->vulkanCreateInfo, sizeof(instInfo));
            instInfo.enabledExtensionCount = (uint32_t)extensions.size();
            instInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

            auto pfnCreateInstance = (PFN_vkCreateInstance)createInfo->pfnGetInstanceProcAddr(nullptr, "vkCreateInstance");
            *vulkanResult = pfnCreateInstance(&instInfo, createInfo->vulkanAllocator, vulkanInstance);
        }

        return XR_SUCCESS;
    }

    virtual XrResult CreateVulkanDeviceKHR(XrInstance instance, const XrVulkanDeviceCreateInfoKHR* createInfo,
                                           VkDevice* vulkanDevice, VkResult* vulkanResult) override {
        PFN_xrGetVulkanDeviceExtensionsKHR pfnGetVulkanDeviceExtensionsKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanDeviceExtensionsKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanDeviceExtensionsKHR)));

        uint32_t deviceExtensionNamesSize = 0;
        CHECK_XRCMD(pfnGetVulkanDeviceExtensionsKHR(instance, createInfo->systemId, 0, &deviceExtensionNamesSize, nullptr));
        std::vector<char> deviceExtensionNames(deviceExtensionNamesSize);
        if (deviceExtensionNamesSize > 0) {
            CHECK_XRCMD(pfnGetVulkanDeviceExtensionsKHR(instance, createInfo->systemId, deviceExtensionNamesSize,
                                                        &deviceExtensionNamesSize, &deviceExtensionNames[0]));
        }
        {
            // Note: This cannot outlive the extensionNames above, since it's just a collection of views into that string!
            std::vector<const char*> extensions;

            if (deviceExtensionNamesSize > 0) {
                extensions = ParseExtensionString(&deviceExtensionNames[0]);
            }
            LogVulkanExtensions("Vulkan Device Extensions, requested by runtime", extensions);

            // Merge the runtime's request with the applications requests
            for (uint32_t i = 0; i < createInfo->vulkanCreateInfo->enabledExtensionCount; ++i) {
                extensions.push_back(createInfo->vulkanCreateInfo->ppEnabledExtensionNames[i]);
            }
            LogVulkanExtensions("Vulkan Device Extensions, requested by application", extensions,
                                (uint32_t)extensions.size() - createInfo->vulkanCreateInfo->enabledExtensionCount);

            VkPhysicalDeviceFeatures features{};
            memcpy(&features, createInfo->vulkanCreateInfo->pEnabledFeatures, sizeof(features));

#if !defined(XR_USE_PLATFORM_ANDROID)
            VkPhysicalDeviceFeatures availableFeatures{};
            vkGetPhysicalDeviceFeatures(m_vkPhysicalDevice, &availableFeatures);
            if (availableFeatures.shaderStorageImageMultisample == VK_TRUE) {
                // Setting this quiets down a validation error triggered by the Oculus runtime
                features.shaderStorageImageMultisample = VK_TRUE;
            }
#endif

            VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            memcpy(&deviceInfo, createInfo->vulkanCreateInfo, sizeof(deviceInfo));
            deviceInfo.pEnabledFeatures = &features;
            deviceInfo.enabledExtensionCount = (uint32_t)extensions.size();
            deviceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

            auto pfnCreateDevice = (PFN_vkCreateDevice)createInfo->pfnGetInstanceProcAddr(m_vkInstance, "vkCreateDevice");
            *vulkanResult = pfnCreateDevice(m_vkPhysicalDevice, &deviceInfo, createInfo->vulkanAllocator, vulkanDevice);
        }

        return XR_SUCCESS;
    }

    virtual XrResult GetVulkanGraphicsDevice2KHR(XrInstance instance, const XrVulkanGraphicsDeviceGetInfoKHR* getInfo,
                                                 VkPhysicalDevice* vulkanPhysicalDevice) override {
        PFN_xrGetVulkanGraphicsDeviceKHR pfnGetVulkanGraphicsDeviceKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsDeviceKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsDeviceKHR)));

        if (getInfo->next != nullptr) {
            return XR_ERROR_FEATURE_UNSUPPORTED;
        }

        CHECK_XRCMD(pfnGetVulkanGraphicsDeviceKHR(instance, getInfo->systemId, getInfo->vulkanInstance, vulkanPhysicalDevice));

        return XR_SUCCESS;
    }

    virtual XrResult GetVulkanGraphicsRequirements2KHR(XrInstance instance, XrSystemId systemId,
                                                       XrGraphicsRequirementsVulkan2KHR* graphicsRequirements) override {
        PFN_xrGetVulkanGraphicsRequirementsKHR pfnGetVulkanGraphicsRequirementsKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetVulkanGraphicsRequirementsKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsRequirementsKHR)));

        XrGraphicsRequirementsVulkanKHR legacyRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
        CHECK_XRCMD(pfnGetVulkanGraphicsRequirementsKHR(instance, systemId, &legacyRequirements));

        graphicsRequirements->maxApiVersionSupported = legacyRequirements.maxApiVersionSupported;
        graphicsRequirements->minApiVersionSupported = legacyRequirements.minApiVersionSupported;

        return XR_SUCCESS;
    }
};

}  // namespace

std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_Vulkan() { return std::make_shared<VulkanGraphicsPlugin>(); }

std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_VulkanLegacy() { return std::make_shared<VulkanGraphicsPluginLegacy>(); }

#endif  // XR_USE_GRAPHICS_API_VULKAN
