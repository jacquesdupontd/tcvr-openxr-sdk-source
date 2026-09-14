// Copyright (c) 2017-2026 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "openxr/openxr.h"
#include "pch.h"
#include "common.h"
#include "platformdata.h"
#include "platformplugin.h"
#include "graphicsplugin.h"
#include "swapchain_image_data.h"
#include "openxr_program.h"
#include <common/xr_linear.h>
#include "input_bridge.h"
#include "aim_state.h"
#include "xr_gun.h"
#include "ray_plane_mapping.h"
#include "virtual_screen.h"
#include "space_debug.h"
#include "display_refresh.h"
#include "audio_bridge.h"
#include "settings.h"
#include <array>
#include <cmath>
#include <set>

namespace {

#if !defined(XR_USE_PLATFORM_WIN32)
#define strcpy_s(dest, source) strncpy((dest), (source), sizeof(dest))
#endif

namespace Side {
const int LEFT = 0;
const int RIGHT = 1;
const int COUNT = 2;
}  // namespace Side

inline std::string GetXrVersionString(XrVersion ver) {
    return Fmt("%d.%d.%d", XR_VERSION_MAJOR(ver), XR_VERSION_MINOR(ver), XR_VERSION_PATCH(ver));
}

namespace Math {
namespace Pose {
static XrPosef Identity() {
    XrPosef t{};
    t.orientation.w = 1;
    return t;
}

static XrPosef Translation(const XrVector3f& translation) {
    XrPosef t = Identity();
    t.position = translation;
    return t;
}

static XrPosef RotateCCWAboutYAxis(float radians, XrVector3f translation) {
    XrPosef t = Identity();
    t.orientation.x = 0.f;
    t.orientation.y = std::sin(radians * 0.5f);
    t.orientation.z = 0.f;
    t.orientation.w = std::cos(radians * 0.5f);
    t.position = translation;
    return t;
}
}  // namespace Pose
}  // namespace Math

inline XrReferenceSpaceCreateInfo GetXrReferenceSpaceCreateInfo(const std::string& referenceSpaceTypeStr) {
    XrReferenceSpaceCreateInfo referenceSpaceCreateInfo{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Identity();
    if (EqualsIgnoreCase(referenceSpaceTypeStr, "View")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "ViewFront")) {
        // Render head-locked 2m in front of device.
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::Translation({0.f, 0.f, -2.f}),
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Local")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "Stage")) {
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeft")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {-2.f, 0.f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRight")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(0.f, {2.f, 0.f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageLeftRotated")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(3.14f / 3.f, {-2.f, 0.5f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else if (EqualsIgnoreCase(referenceSpaceTypeStr, "StageRightRotated")) {
        referenceSpaceCreateInfo.poseInReferenceSpace = Math::Pose::RotateCCWAboutYAxis(-3.14f / 3.f, {2.f, 0.5f, -2.f});
        referenceSpaceCreateInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE;
    } else {
        throw std::invalid_argument(Fmt("Unknown reference space type '%s'", referenceSpaceTypeStr.c_str()));
    }
    return referenceSpaceCreateInfo;
}

struct OpenXrProgram : IOpenXrProgram {
    OpenXrProgram(const std::shared_ptr<IPlatformPlugin>& platformPlugin, const std::shared_ptr<IGraphicsPlugin>& graphicsPlugin)
        : m_platformPlugin(platformPlugin), m_graphicsPlugin(graphicsPlugin) {}

    ~OpenXrProgram() override {
        if (m_input.actionSet != XR_NULL_HANDLE) {
            for (auto hand : {Side::LEFT, Side::RIGHT}) {
                xrDestroySpace(m_input.handSpace[hand]);
                if (m_input.aimSpace[hand]) xrDestroySpace(m_input.aimSpace[hand]);
            }
            xrDestroyActionSet(m_input.actionSet);
        }

        for (Swapchain swapchain : m_swapchains) {
            xrDestroySwapchain(swapchain.handle);
        }

        for (XrSpace visualizedSpace : m_visualizedSpaces) {
            xrDestroySpace(visualizedSpace);
        }

        if (m_appSpace != XR_NULL_HANDLE) {
            xrDestroySpace(m_appSpace);
        }

        if (m_session != XR_NULL_HANDLE) {
            xrDestroySession(m_session);
        }

        if (m_instance != XR_NULL_HANDLE) {
            xrDestroyInstance(m_instance);
        }
    }

    static void LogLayersAndExtensions() {
        // Write out extension properties for a given layer.
        const auto logExtensions = [](const char* layerName, int indent = 0) {
            uint32_t instanceExtensionCount;
            CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(layerName, 0, &instanceExtensionCount, nullptr));
            std::vector<XrExtensionProperties> extensions(instanceExtensionCount, {XR_TYPE_EXTENSION_PROPERTIES});
            CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(layerName, (uint32_t)extensions.size(), &instanceExtensionCount,
                                                               extensions.data()));

            const std::string indentStr(indent, ' ');
            Log::Write(Log::Level::Verbose, Fmt("%sAvailable Extensions: (%d)", indentStr.c_str(), instanceExtensionCount));
            for (const XrExtensionProperties& extension : extensions) {
                Log::Write(Log::Level::Verbose, Fmt("%s  Name=%s SpecVersion=%d", indentStr.c_str(), extension.extensionName,
                                                    extension.extensionVersion));
            }
        };

        // Log non-layer extensions (layerName==nullptr).
        logExtensions(nullptr);

        // Log layers and any of their extensions.
        {
            uint32_t layerCount;
            CHECK_XRCMD(xrEnumerateApiLayerProperties(0, &layerCount, nullptr));
            std::vector<XrApiLayerProperties> layers(layerCount, {XR_TYPE_API_LAYER_PROPERTIES});
            CHECK_XRCMD(xrEnumerateApiLayerProperties((uint32_t)layers.size(), &layerCount, layers.data()));

            Log::Write(Log::Level::Info, Fmt("Available Layers: (%d)", layerCount));
            for (const XrApiLayerProperties& layer : layers) {
                Log::Write(Log::Level::Verbose,
                           Fmt("  Name=%s SpecVersion=%s LayerVersion=%d Description=%s", layer.layerName,
                               GetXrVersionString(layer.specVersion).c_str(), layer.layerVersion, layer.description));
                logExtensions(layer.layerName, 4);
            }
        }
    }

    void LogInstanceInfo() {
        CHECK(m_instance != XR_NULL_HANDLE);

        XrInstanceProperties instanceProperties{XR_TYPE_INSTANCE_PROPERTIES};
        CHECK_XRCMD(xrGetInstanceProperties(m_instance, &instanceProperties));

        Log::Write(Log::Level::Info, Fmt("Instance RuntimeName=%s RuntimeVersion=%s", instanceProperties.runtimeName,
                                         GetXrVersionString(instanceProperties.runtimeVersion).c_str()));
    }

    void CreateInstanceInternal() {
        CHECK(m_instance == XR_NULL_HANDLE);

        // Create union of extensions required by platform and graphics plugins.
        std::vector<const char*> extensions;

        // Transform platform and graphics extension std::strings to C strings.
        const std::vector<std::string> platformExtensions = m_platformPlugin->GetInstanceExtensions();
        std::transform(platformExtensions.begin(), platformExtensions.end(), std::back_inserter(extensions),
                       [](const std::string& ext) { return ext.c_str(); });
        const std::vector<std::string> graphicsExtensions = m_graphicsPlugin->GetInstanceExtensions();
        std::transform(graphicsExtensions.begin(), graphicsExtensions.end(), std::back_inserter(extensions),
                       [](const std::string& ext) { return ext.c_str(); });

        uint32_t instanceExtensionCount;
        CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(nullptr, 0, &instanceExtensionCount, nullptr));
        std::vector<XrExtensionProperties> extensionProperties =
            std::vector<XrExtensionProperties>(instanceExtensionCount, {XR_TYPE_EXTENSION_PROPERTIES});
        CHECK_XRCMD(xrEnumerateInstanceExtensionProperties(nullptr, (uint32_t)extensionProperties.size(), &instanceExtensionCount,
                                                           extensionProperties.data()));

        // enable depth extension if supported
        auto depthExtensionProperties =
            std::find_if(extensionProperties.begin(), extensionProperties.end(), [](const XrExtensionProperties& item) {
                return 0 == strcmp(item.extensionName, XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME);
            });
        if (depthExtensionProperties != extensionProperties.end()) {
            Log::Write(Log::Level::Info, Fmt("Depth submission supported (%s)", depthExtensionProperties->extensionName));
            extensions.push_back(depthExtensionProperties->extensionName);
            m_supportsDepthLayer = true;
        } else {
            Log::Write(Log::Level::Info, Fmt("Depth submission NOT supported (%s)", XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME));
        }

        // Presentation rate control. Optional: without it the runtime simply
        // keeps whatever rate it chose, and the emulator is unaffected either
        // way -- the arcade clock never depends on the presentation clock.
        const char* refreshExtension = arcadexr::xr::RefreshExtensionName();
        auto refreshProperties =
            std::find_if(extensionProperties.begin(), extensionProperties.end(), [refreshExtension](const XrExtensionProperties& item) {
                return 0 == strcmp(item.extensionName, refreshExtension);
            });
        if (refreshProperties != extensionProperties.end()) {
            extensions.push_back(refreshProperties->extensionName);
            m_supportsDisplayRefreshRate = true;
            Log::Write(Log::Level::Info, Fmt("Display refresh rate control supported (%s)", refreshExtension));
        } else {
            Log::Write(Log::Level::Info, Fmt("Display refresh rate control NOT supported (%s)", refreshExtension));
        }

        XrInstanceCreateInfo createInfo{XR_TYPE_INSTANCE_CREATE_INFO};
        createInfo.next = m_platformPlugin->GetInstanceCreateExtension();
        createInfo.enabledExtensionCount = (uint32_t)extensions.size();
        createInfo.enabledExtensionNames = extensions.data();

        strcpy(createInfo.applicationInfo.applicationName, "HelloXR");

        // Current version is 1.1.x, but hello_xr only requires 1.0.x
        createInfo.applicationInfo.apiVersion = XR_API_VERSION_1_0;

        CHECK_XRCMD(xrCreateInstance(&createInfo, &m_instance));
        Log::Write(Log::Level::Info, "TCVR_M0 xrCreateInstance succeeded");
    }

    void CreateInstance() override {
        LogLayersAndExtensions();

        CreateInstanceInternal();

        LogInstanceInfo();
    }

    void LogViewConfigurations() {
        CHECK(m_instance != XR_NULL_HANDLE);
        CHECK(m_systemId != XR_NULL_SYSTEM_ID);

        uint32_t viewConfigTypeCount;
        CHECK_XRCMD(xrEnumerateViewConfigurations(m_instance, m_systemId, 0, &viewConfigTypeCount, nullptr));
        std::vector<XrViewConfigurationType> viewConfigTypes(viewConfigTypeCount);
        CHECK_XRCMD(xrEnumerateViewConfigurations(m_instance, m_systemId, viewConfigTypeCount, &viewConfigTypeCount,
                                                  viewConfigTypes.data()));
        CHECK((uint32_t)viewConfigTypes.size() == viewConfigTypeCount);

        Log::Write(Log::Level::Info, Fmt("Available View Configuration Types: (%d)", viewConfigTypeCount));
        for (XrViewConfigurationType viewConfigType : viewConfigTypes) {
            Log::Write(Log::Level::Verbose, Fmt("  View Configuration Type: %s %s", to_string(viewConfigType),
                                                viewConfigType == m_viewConfigType ? "(Selected)" : ""));

            XrViewConfigurationProperties viewConfigProperties{XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
            CHECK_XRCMD(xrGetViewConfigurationProperties(m_instance, m_systemId, viewConfigType, &viewConfigProperties));

            Log::Write(Log::Level::Verbose,
                       Fmt("  View configuration FovMutable=%s", viewConfigProperties.fovMutable == XR_TRUE ? "True" : "False"));

            uint32_t viewCount;
            CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfigType, 0, &viewCount, nullptr));
            if (viewCount > 0) {
                std::vector<XrViewConfigurationView> views(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
                CHECK_XRCMD(
                    xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfigType, viewCount, &viewCount, views.data()));

                for (uint32_t i = 0; i < views.size(); i++) {
                    const XrViewConfigurationView& view = views[i];

                    Log::Write(Log::Level::Verbose, Fmt("    View [%d]: Recommended Width=%d Height=%d SampleCount=%d", i,
                                                        view.recommendedImageRectWidth, view.recommendedImageRectHeight,
                                                        view.recommendedSwapchainSampleCount));
                    Log::Write(Log::Level::Verbose,
                               Fmt("    View [%d]:     Maximum Width=%d Height=%d SampleCount=%d", i, view.maxImageRectWidth,
                                   view.maxImageRectHeight, view.maxSwapchainSampleCount));
                }
            } else {
                Log::Write(Log::Level::Error, Fmt("Empty view configuration type"));
            }

            LogEnvironmentBlendMode(viewConfigType);
        }
    }

    void LogEnvironmentBlendMode(XrViewConfigurationType type) {
        CHECK(m_instance != XR_NULL_HANDLE);
        CHECK(m_systemId != 0);

        uint32_t count;
        CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, type, 0, &count, nullptr));
        CHECK(count > 0);

        Log::Write(Log::Level::Info, Fmt("  Available Environment Blend Mode count : (%d)", count));

        std::vector<XrEnvironmentBlendMode> blendModes(count);
        CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, type, count, &count, blendModes.data()));

        for (XrEnvironmentBlendMode mode : blendModes) {
            const bool blendModeMatch = (mode == m_blendMode);
            Log::Write(Log::Level::Info,
                       Fmt("    Environment Blend Mode (%s) : %s", to_string(mode), blendModeMatch ? "(Selected)" : ""));
        }
    }

    std::array<float, 4> GetBackgroundClearColor() const override {
        static const std::array<float, 4> SlateGrey{{0.184313729f, 0.309803933f, 0.309803933f, 1.0f}};
        static const std::array<float, 4> TransparentBlack{{0.0f, 0.0f, 0.0f, 0.0f}};
        static const std::array<float, 4> Black{{0.0f, 0.0f, 0.0f, 1.0f}};

        switch (m_blendMode) {
            case XR_ENVIRONMENT_BLEND_MODE_OPAQUE:
                return SlateGrey;
            case XR_ENVIRONMENT_BLEND_MODE_ADDITIVE:
                return Black;
            case XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND:
                return TransparentBlack;
            default:
                return SlateGrey;
        }
    }

    void InitializeSystem(XrFormFactor formFactor, XrViewConfigurationType viewConfigType, bool ebmOverride,
                          XrEnvironmentBlendMode ebm) override {
        CHECK(m_instance != XR_NULL_HANDLE);
        CHECK(m_systemId == XR_NULL_SYSTEM_ID);

        XrSystemGetInfo systemInfo{XR_TYPE_SYSTEM_GET_INFO};
        systemInfo.formFactor = formFactor;
        CHECK_XRCMD(xrGetSystem(m_instance, &systemInfo, &m_systemId));
        Log::Write(Log::Level::Info, Fmt("TCVR_M0 xrGetSystem succeeded: system=%d", m_systemId));

        Log::Write(Log::Level::Verbose, Fmt("Using system %d for form factor %s", m_systemId, to_string(formFactor)));
        CHECK(m_instance != XR_NULL_HANDLE);
        CHECK(m_systemId != XR_NULL_SYSTEM_ID);

        {
            // Note: If this condition is not met, the project will need to be audited
            // to see how support should be added.
            CHECK_MSG(viewConfigType == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO ||
                          viewConfigType == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                      "Unsupported view configuration type");

            m_viewConfigType = viewConfigType;
        }

        {
            uint32_t count;
            CHECK_XRCMD(xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, m_viewConfigType, 0, &count, nullptr));
            CHECK(count > 0);
            std::vector<XrEnvironmentBlendMode> blendModes(count);
            CHECK_XRCMD(
                xrEnumerateEnvironmentBlendModes(m_instance, m_systemId, m_viewConfigType, count, &count, blendModes.data()));
            m_blendModesAvailable.clear();
            for (XrEnvironmentBlendMode mode : blendModes) {
                if (!m_blendModesAvailable.empty()) m_blendModesAvailable += ",";
                m_blendModesAvailable += to_string(mode);
            }
            if (ebmOverride) {
                if (std::find(blendModes.begin(), blendModes.end(), ebm) == blendModes.end()) {
                    THROW("Selected blendmode is not available from runtime");
                }
                m_blendMode = ebm;
            } else {
                // Runtimes return blend modes in preference order
                m_blendMode = blendModes[0];
            }
        }
    }

    void InitializeDevice() override {
        LogViewConfigurations();

        // The graphics API can initialize the graphics device now that the systemId and instance
        // handle are available.
        m_graphicsPlugin->InitializeDevice(m_instance, m_systemId);
    }

    void LogReferenceSpaces() {
        CHECK(m_session != XR_NULL_HANDLE);

        uint32_t spaceCount;
        CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, 0, &spaceCount, nullptr));
        std::vector<XrReferenceSpaceType> spaces(spaceCount);
        CHECK_XRCMD(xrEnumerateReferenceSpaces(m_session, spaceCount, &spaceCount, spaces.data()));

        Log::Write(Log::Level::Info, Fmt("Available reference spaces: %d", spaceCount));
        m_referenceSpacesAvailable.clear();
        for (XrReferenceSpaceType space : spaces) {
            Log::Write(Log::Level::Verbose, Fmt("  Name: %s", to_string(space)));
            if (!m_referenceSpacesAvailable.empty()) m_referenceSpacesAvailable += ",";
            m_referenceSpacesAvailable += to_string(space);
        }
    }

    struct InputState {
        XrActionSet actionSet{XR_NULL_HANDLE};
        XrAction grabAction{XR_NULL_HANDLE};
        XrAction poseAction{XR_NULL_HANDLE};
        XrAction aimAction{XR_NULL_HANDLE};
        XrAction vibrateAction{XR_NULL_HANDLE};
        XrAction quitAction{XR_NULL_HANDLE};
        XrAction triggerAction{XR_NULL_HANDLE};
        XrAction pedalAction{XR_NULL_HANDLE};
        XrAction startAction{XR_NULL_HANDLE};
        XrAction coinAction{XR_NULL_HANDLE};
        XrAction recenterScreenAction{XR_NULL_HANDLE};
        XrAction crosshairAction{XR_NULL_HANDLE};
        std::array<XrPath, Side::COUNT> handSubactionPath;
        std::array<XrSpace, Side::COUNT> handSpace;
        std::array<XrSpace, Side::COUNT> aimSpace{};
        std::array<float, Side::COUNT> handScale = {{1.0f, 1.0f}};
        std::array<XrBool32, Side::COUNT> handActive;
        bool lastTrigger{false};
        bool lastPedal{false};
        bool lastStart{false};
        bool lastCoin{false};
    };

    void InitializeActions() {
        m_gunCalibration = arcadexr::gun::LoadCalibration();
        // Create an action set.
        {
            XrActionSetCreateInfo actionSetInfo{XR_TYPE_ACTION_SET_CREATE_INFO};
            strcpy_s(actionSetInfo.actionSetName, "gameplay");
            strcpy_s(actionSetInfo.localizedActionSetName, "Gameplay");
            actionSetInfo.priority = 0;
            CHECK_XRCMD(xrCreateActionSet(m_instance, &actionSetInfo, &m_input.actionSet));
        }

        // Get the XrPath for the left and right hands - we will use them as subaction paths.
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left", &m_input.handSubactionPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right", &m_input.handSubactionPath[Side::RIGHT]));

        // Create actions.
        {
            // Create an input action for grabbing objects with the left and right hands.
            XrActionCreateInfo actionInfo{XR_TYPE_ACTION_CREATE_INFO};
            actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
            strcpy_s(actionInfo.actionName, "grab_object");
            strcpy_s(actionInfo.localizedActionName, "Grab Object");
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.grabAction));

            // Create an input action getting the left and right hand poses.
            actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
            strcpy_s(actionInfo.actionName, "hand_pose");
            strcpy_s(actionInfo.localizedActionName, "Hand Pose");
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.poseAction));
            strcpy_s(actionInfo.actionName, "aim_pose");
            strcpy_s(actionInfo.localizedActionName, "Gun Aim Pose");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.aimAction));

            // Create output actions for vibrating the left and right controller.
            actionInfo.actionType = XR_ACTION_TYPE_VIBRATION_OUTPUT;
            strcpy_s(actionInfo.actionName, "vibrate_hand");
            strcpy_s(actionInfo.localizedActionName, "Vibrate Hand");
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.vibrateAction));

            // Create input actions for quitting the session using the left and right controller.
            // Since it doesn't matter which hand did this, we do not specify subaction paths for it.
            // We will just suggest bindings for both hands, where possible.
            actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
            strcpy_s(actionInfo.actionName, "calibrate_gun");
            strcpy_s(actionInfo.localizedActionName, "Calibrate Gun");
            actionInfo.countSubactionPaths = 0;
            actionInfo.subactionPaths = nullptr;
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.quitAction));

            actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
            strcpy_s(actionInfo.actionName, "gun_trigger");
            strcpy_s(actionInfo.localizedActionName, "Gun Trigger");
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.triggerAction));

            actionInfo.actionType = XR_ACTION_TYPE_BOOLEAN_INPUT;
            strcpy_s(actionInfo.actionName, "pedal");
            strcpy_s(actionInfo.localizedActionName, "Pedal");
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.pedalAction));
            strcpy_s(actionInfo.actionName, "start");
            strcpy_s(actionInfo.localizedActionName, "Start");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.startAction));
            strcpy_s(actionInfo.actionName, "coin");
            strcpy_s(actionInfo.localizedActionName, "Coin");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.coinAction));

            // Deliberate, explicit placement of the cabinet. Separate from gun
            // calibration, which used to drag a screen recenter along with it.
            actionInfo.countSubactionPaths = 0;
            actionInfo.subactionPaths = nullptr;
            strcpy_s(actionInfo.actionName, "recenter_screen");
            strcpy_s(actionInfo.localizedActionName, "Recenter Arcade Screen");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.recenterScreenAction));
            strcpy_s(actionInfo.actionName, "crosshair_mode");
            strcpy_s(actionInfo.localizedActionName, "Cycle Crosshair Mode");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.crosshairAction));
        }

        std::array<XrPath, Side::COUNT> selectPath;
        std::array<XrPath, Side::COUNT> squeezeValuePath;
        std::array<XrPath, Side::COUNT> squeezeForcePath;
        std::array<XrPath, Side::COUNT> squeezeClickPath;
        std::array<XrPath, Side::COUNT> posePath;
        std::array<XrPath, Side::COUNT> hapticPath;
        std::array<XrPath, Side::COUNT> menuClickPath;
        std::array<XrPath, Side::COUNT> bClickPath;
        std::array<XrPath, Side::COUNT> triggerValuePath;
        std::array<XrPath, Side::COUNT> aClickPath;
        std::array<XrPath, Side::COUNT> xClickPath;
        std::array<XrPath, Side::COUNT> yClickPath;
        std::array<XrPath, Side::COUNT> thumbstickClickPath;
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/select/click", &selectPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/select/click", &selectPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/value", &squeezeValuePath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/value", &squeezeValuePath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/force", &squeezeForcePath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/force", &squeezeForcePath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/squeeze/click", &squeezeClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/squeeze/click", &squeezeClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/grip/pose", &posePath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/grip/pose", &posePath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/output/haptic", &hapticPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/output/haptic", &hapticPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/menu/click", &menuClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/menu/click", &menuClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/b/click", &bClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/b/click", &bClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/trigger/value", &triggerValuePath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/trigger/value", &triggerValuePath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/a/click", &aClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/a/click", &aClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/x/click", &xClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/x/click", &xClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/y/click", &yClickPath[Side::LEFT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/y/click", &yClickPath[Side::RIGHT]));
        CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/thumbstick/click", &thumbstickClickPath[Side::LEFT]));
        CHECK_XRCMD(
            xrStringToPath(m_instance, "/user/hand/right/input/thumbstick/click", &thumbstickClickPath[Side::RIGHT]));
        // Suggest bindings for KHR Simple.
        {
            XrPath khrSimpleInteractionProfilePath;
            CHECK_XRCMD(
                xrStringToPath(m_instance, "/interaction_profiles/khr/simple_controller", &khrSimpleInteractionProfilePath));
            std::vector<XrActionSuggestedBinding> bindings{{// Fall back to a click input for the grab action.
                                                            {m_input.grabAction, selectPath[Side::LEFT]},
                                                            {m_input.grabAction, selectPath[Side::RIGHT]},
                                                            {m_input.poseAction, posePath[Side::LEFT]},
                                                            {m_input.poseAction, posePath[Side::RIGHT]},
                                                            {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                            {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
            XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = khrSimpleInteractionProfilePath;
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
        }
        // Suggest bindings for the Oculus Touch.
        {
            XrPath aimLeft, aimRight;
            CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/left/input/aim/pose", &aimLeft));
            CHECK_XRCMD(xrStringToPath(m_instance, "/user/hand/right/input/aim/pose", &aimRight));
            XrPath oculusTouchInteractionProfilePath;
            CHECK_XRCMD(
                xrStringToPath(m_instance, "/interaction_profiles/oculus/touch_controller", &oculusTouchInteractionProfilePath));
            std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeValuePath[Side::LEFT]},
                                                            {m_input.grabAction, squeezeValuePath[Side::RIGHT]},
                                                            {m_input.poseAction, posePath[Side::LEFT]},
                                                            {m_input.poseAction, posePath[Side::RIGHT]},
                                                            {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::RIGHT]},
                                                            {m_input.triggerAction, triggerValuePath[Side::RIGHT]},
                                                            {m_input.pedalAction, aClickPath[Side::RIGHT]},
                                                            {m_input.startAction, bClickPath[Side::RIGHT]},
                                                            {m_input.coinAction, xClickPath[Side::LEFT]},
                                                            {m_input.recenterScreenAction, yClickPath[Side::LEFT]},
                                                            {m_input.crosshairAction, thumbstickClickPath[Side::LEFT]}}};
            XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = oculusTouchInteractionProfilePath;
            bindings.push_back({m_input.aimAction, aimLeft});
            bindings.push_back({m_input.aimAction, aimRight});
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
        }
        // Suggest bindings for the Vive Controller.
        {
            XrPath viveControllerInteractionProfilePath;
            CHECK_XRCMD(
                xrStringToPath(m_instance, "/interaction_profiles/htc/vive_controller", &viveControllerInteractionProfilePath));
            std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, triggerValuePath[Side::LEFT]},
                                                            {m_input.grabAction, triggerValuePath[Side::RIGHT]},
                                                            {m_input.poseAction, posePath[Side::LEFT]},
                                                            {m_input.poseAction, posePath[Side::RIGHT]},
                                                            {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                            {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                            {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
            XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = viveControllerInteractionProfilePath;
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
        }

        // Suggest bindings for the Valve Index Controller.
        {
            XrPath indexControllerInteractionProfilePath;
            CHECK_XRCMD(
                xrStringToPath(m_instance, "/interaction_profiles/valve/index_controller", &indexControllerInteractionProfilePath));
            std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeForcePath[Side::LEFT]},
                                                            {m_input.grabAction, squeezeForcePath[Side::RIGHT]},
                                                            {m_input.poseAction, posePath[Side::LEFT]},
                                                            {m_input.poseAction, posePath[Side::RIGHT]},
                                                            {m_input.quitAction, bClickPath[Side::LEFT]},
                                                            {m_input.quitAction, bClickPath[Side::RIGHT]},
                                                            {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
            XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = indexControllerInteractionProfilePath;
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
        }

        // Suggest bindings for the Microsoft Mixed Reality Motion Controller.
        {
            XrPath microsoftMixedRealityInteractionProfilePath;
            CHECK_XRCMD(xrStringToPath(m_instance, "/interaction_profiles/microsoft/motion_controller",
                                       &microsoftMixedRealityInteractionProfilePath));
            std::vector<XrActionSuggestedBinding> bindings{{{m_input.grabAction, squeezeClickPath[Side::LEFT]},
                                                            {m_input.grabAction, squeezeClickPath[Side::RIGHT]},
                                                            {m_input.poseAction, posePath[Side::LEFT]},
                                                            {m_input.poseAction, posePath[Side::RIGHT]},
                                                            {m_input.quitAction, menuClickPath[Side::LEFT]},
                                                            {m_input.quitAction, menuClickPath[Side::RIGHT]},
                                                            {m_input.vibrateAction, hapticPath[Side::LEFT]},
                                                            {m_input.vibrateAction, hapticPath[Side::RIGHT]}}};
            XrInteractionProfileSuggestedBinding suggestedBindings{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
            suggestedBindings.interactionProfile = microsoftMixedRealityInteractionProfilePath;
            suggestedBindings.suggestedBindings = bindings.data();
            suggestedBindings.countSuggestedBindings = (uint32_t)bindings.size();
            CHECK_XRCMD(xrSuggestInteractionProfileBindings(m_instance, &suggestedBindings));
        }
        XrActionSpaceCreateInfo actionSpaceInfo{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        actionSpaceInfo.action = m_input.poseAction;
        actionSpaceInfo.poseInActionSpace.orientation.w = 1.f;
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::LEFT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::LEFT]));
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::RIGHT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::RIGHT]));
        actionSpaceInfo.action = m_input.aimAction;
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::LEFT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.aimSpace[Side::LEFT]));
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::RIGHT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.aimSpace[Side::RIGHT]));

        XrSessionActionSetsAttachInfo attachInfo{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
        attachInfo.countActionSets = 1;
        attachInfo.actionSets = &m_input.actionSet;
        CHECK_XRCMD(xrAttachSessionActionSets(m_session, &attachInfo));
    }

    void CreateVisualizedSpaces() {
        CHECK(m_session != XR_NULL_HANDLE);

        std::string visualizedSpaces[] = {"ViewFront",        "Local", "Stage", "StageLeft", "StageRight", "StageLeftRotated",
                                          "StageRightRotated"};

        for (const auto& visualizedSpace : visualizedSpaces) {
            XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo(visualizedSpace);
            XrSpace space;
            XrResult res = xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &space);
            if (XR_SUCCEEDED(res)) {
                m_visualizedSpaces.push_back(space);
            } else {
                Log::Write(Log::Level::Warning,
                           Fmt("Failed to create reference space %s with error %d", visualizedSpace.c_str(), res));
            }
        }
    }

    void InitializeSession(std::string appSpace) override {
        CHECK(m_instance != XR_NULL_HANDLE);
        CHECK(m_session == XR_NULL_HANDLE);

        {
            Log::Write(Log::Level::Verbose, Fmt("Creating session..."));

            XrSessionCreateInfo createInfo{XR_TYPE_SESSION_CREATE_INFO};
            createInfo.next = m_graphicsPlugin->GetGraphicsBinding();
            createInfo.systemId = m_systemId;
            CHECK_XRCMD(xrCreateSession(m_instance, &createInfo, &m_session));
            Log::Write(Log::Level::Info, "TCVR_M0 xrCreateSession succeeded");
        }

        LogReferenceSpaces();
        InitializeActions();
        CreateVisualizedSpaces();

        {
            XrReferenceSpaceCreateInfo referenceSpaceCreateInfo = GetXrReferenceSpaceCreateInfo(appSpace);
            CHECK_XRCMD(xrCreateReferenceSpace(m_session, &referenceSpaceCreateInfo, &m_appSpace));
            m_appSpaceRequested = appSpace;
            m_appSpaceTypeEnum = referenceSpaceCreateInfo.referenceSpaceType;
            m_appSpaceType = to_string(referenceSpaceCreateInfo.referenceSpaceType);
        }


    }

    void CreateSwapchains() override {
        CHECK(m_session != XR_NULL_HANDLE);
        CHECK(m_swapchains.empty());
        CHECK(m_configViews.empty());

        // Read graphics properties for preferred swapchain length and logging.
        XrSystemProperties systemProperties{XR_TYPE_SYSTEM_PROPERTIES};
        CHECK_XRCMD(xrGetSystemProperties(m_instance, m_systemId, &systemProperties));

        // Log system properties.
        Log::Write(Log::Level::Info,
                   Fmt("System Properties: Name=%s VendorId=%d", systemProperties.systemName, systemProperties.vendorId));
        Log::Write(Log::Level::Info, Fmt("System Graphics Properties: MaxWidth=%d MaxHeight=%d MaxLayers=%d",
                                         systemProperties.graphicsProperties.maxSwapchainImageWidth,
                                         systemProperties.graphicsProperties.maxSwapchainImageHeight,
                                         systemProperties.graphicsProperties.maxLayerCount));
        Log::Write(Log::Level::Info, Fmt("System Tracking Properties: OrientationTracking=%s PositionTracking=%s",
                                         systemProperties.trackingProperties.orientationTracking == XR_TRUE ? "True" : "False",
                                         systemProperties.trackingProperties.positionTracking == XR_TRUE ? "True" : "False"));

        // Query and cache view configuration views.
        uint32_t viewCount;
        CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewConfigType, 0, &viewCount, nullptr));
        m_configViews.resize(viewCount, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        CHECK_XRCMD(xrEnumerateViewConfigurationViews(m_instance, m_systemId, m_viewConfigType, viewCount, &viewCount,
                                                      m_configViews.data()));

        // Create and cache view buffer for xrLocateViews later.
        m_views.resize(viewCount, {XR_TYPE_VIEW});

        // Create the swapchain and get the images.
        if (viewCount > 0) {
            // Select a swapchain format.
            uint32_t swapchainFormatCount;
            CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, 0, &swapchainFormatCount, nullptr));
            std::vector<int64_t> swapchainFormats(swapchainFormatCount);
            CHECK_XRCMD(xrEnumerateSwapchainFormats(m_session, (uint32_t)swapchainFormats.size(), &swapchainFormatCount,
                                                    swapchainFormats.data()));
            CHECK(swapchainFormatCount == swapchainFormats.size());
            m_colorSwapchainFormat = m_graphicsPlugin->SelectColorSwapchainFormat(true, swapchainFormats);
            m_depthSwapchainFormat = m_graphicsPlugin->SelectDepthSwapchainFormat(false, swapchainFormats);

            if (m_depthSwapchainFormat == -1) {
                Log::Write(Log::Level::Info,
                           "Runtime does not support creating swapchains with a suitable depth format. Using our own  fallback "
                           "depth textures!");
                // can't submit our own fallback texture
                m_supportsDepthLayer = false;
            }

            // Print swapchain formats and the selected ones.
            {
                std::string swapchainFormatsString;
                for (int64_t format : swapchainFormats) {
                    const bool selected = (format == m_colorSwapchainFormat || format == m_depthSwapchainFormat);
                    swapchainFormatsString += " ";
                    if (selected) {
                        swapchainFormatsString += "[";
                    }
                    swapchainFormatsString += std::to_string(format);
                    if (selected) {
                        swapchainFormatsString += "]";
                    }
                }
                Log::Write(Log::Level::Verbose, Fmt("Swapchain Formats: %s", swapchainFormatsString.c_str()));
            }

            // Create a swapchain for each view.
            for (uint32_t i = 0; i < viewCount; i++) {
                const XrViewConfigurationView& vp = m_configViews[i];
                Log::Write(Log::Level::Info,
                           Fmt("Creating color %s swapchain for view %d with dimensions Width=%d Height=%d SampleCount=%d",
                               m_depthSwapchainFormat == -1 ? "" : "and depth", i, vp.recommendedImageRectWidth,
                               vp.recommendedImageRectHeight, vp.recommendedSwapchainSampleCount));

                // Create the swapchain.
                XrSwapchainCreateInfo swapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
                swapchainCreateInfo.arraySize = 1;
                swapchainCreateInfo.format = m_colorSwapchainFormat;
                // The emulator's rasteriser is on the CPU, but it competes with
                // the compositor for memory bandwidth, and at 207 Hz each eye is
                // cleared and redrawn over two hundred times a second. Measured
                // on a Quest 3, MAME held 59.3-59.7 fps while the app was
                // backgrounded and not rendering, and 51-58 in the foreground.
                // This scale is the lever for that, and it is a setting rather
                // than a constant because what it costs is a judgement call.
                float scale = arcadexr::config::GetFloat("xr.resolution_scale", 1.0f);
                if (scale < 0.3f) scale = 0.3f;
                if (scale > 1.0f) scale = 1.0f;
                swapchainCreateInfo.width = uint32_t(vp.recommendedImageRectWidth * scale);
                swapchainCreateInfo.height = uint32_t(vp.recommendedImageRectHeight * scale);
                Log::Write(Log::Level::Info, Fmt("TCVR_M12 eye %u swapchain %ux%u (scale %.2f of %ux%u)", i,
                                                 swapchainCreateInfo.width, swapchainCreateInfo.height, scale,
                                                 vp.recommendedImageRectWidth, vp.recommendedImageRectHeight));
                swapchainCreateInfo.mipCount = 1;
                swapchainCreateInfo.faceCount = 1;
                swapchainCreateInfo.sampleCount = m_graphicsPlugin->GetSupportedSwapchainSampleCount(vp);
                swapchainCreateInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
                Swapchain swapchain;
                swapchain.width = swapchainCreateInfo.width;
                swapchain.height = swapchainCreateInfo.height;
                CHECK_XRCMD(xrCreateSwapchain(m_session, &swapchainCreateInfo, &swapchain.handle));
                Log::Write(Log::Level::Info, Fmt("TCVR_M0 xrCreateSwapchain succeeded: view=%d", i));

                m_swapchains.push_back(swapchain);

                uint32_t imageCount;
                CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, 0, &imageCount, nullptr));

                if (m_depthSwapchainFormat != -1) {
                    XrSwapchainCreateInfo depthSwapchainCreateInfo{XR_TYPE_SWAPCHAIN_CREATE_INFO};
                    depthSwapchainCreateInfo.arraySize = 1;
                    depthSwapchainCreateInfo.format = m_depthSwapchainFormat;
                    depthSwapchainCreateInfo.width = swapchainCreateInfo.width;
                    depthSwapchainCreateInfo.height = swapchainCreateInfo.height;
                    depthSwapchainCreateInfo.mipCount = 1;
                    depthSwapchainCreateInfo.faceCount = 1;
                    depthSwapchainCreateInfo.sampleCount = m_graphicsPlugin->GetSupportedSwapchainSampleCount(vp);
                    depthSwapchainCreateInfo.usageFlags =
                        XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                    Swapchain depthSwapchain;
                    depthSwapchain.width = depthSwapchainCreateInfo.width;
                    depthSwapchain.height = depthSwapchainCreateInfo.height;
                    CHECK_XRCMD(xrCreateSwapchain(m_session, &depthSwapchainCreateInfo, &depthSwapchain.handle));

                    m_depthSwapchains.push_back(depthSwapchain);

                    uint32_t depthImageCount;
                    CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, 0, &depthImageCount, nullptr));

                    // TODO: support this
                    if (depthImageCount != imageCount) {
                        THROW("This runtime has different color and depth swapchain lengths");
                    }

                    ISwapchainImageData* swapchainImages = m_graphicsPlugin->AllocateSwapchainImageDataWithDepthSwapchain(
                        imageCount, swapchainCreateInfo, depthSwapchain.handle, depthSwapchainCreateInfo);
                    CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount,
                                                           swapchainImages->GetColorImageArray()));

                    CHECK_XRCMD(xrEnumerateSwapchainImages(depthSwapchain.handle, depthImageCount, &depthImageCount,
                                                           swapchainImages->GetDepthImageArray()));

                    m_swapchainImages.insert(std::make_pair(swapchain.handle, std::move(swapchainImages)));
                } else {
                    ISwapchainImageData* swapchainImages =
                        m_graphicsPlugin->AllocateSwapchainImageData(imageCount, swapchainCreateInfo);
                    CHECK_XRCMD(xrEnumerateSwapchainImages(swapchain.handle, imageCount, &imageCount,
                                                           swapchainImages->GetColorImageArray()));

                    m_swapchainImages.insert(std::make_pair(swapchain.handle, std::move(swapchainImages)));
                }
            }
        }
    }

    // Return event if one is available, otherwise return null.
    const XrEventDataBaseHeader* TryReadNextEvent() {
        // It is sufficient to clear the just the XrEventDataBuffer header to
        // XR_TYPE_EVENT_DATA_BUFFER
        XrEventDataBaseHeader* baseHeader = reinterpret_cast<XrEventDataBaseHeader*>(&m_eventDataBuffer);
        *baseHeader = {XR_TYPE_EVENT_DATA_BUFFER};
        const XrResult xr = xrPollEvent(m_instance, &m_eventDataBuffer);
        if (xr == XR_SUCCESS) {
            if (baseHeader->type == XR_TYPE_EVENT_DATA_EVENTS_LOST) {
                const XrEventDataEventsLost* const eventsLost = reinterpret_cast<const XrEventDataEventsLost*>(baseHeader);
                Log::Write(Log::Level::Warning, Fmt("%d events lost", eventsLost->lostEventCount));
            }

            return baseHeader;
        }
        if (xr == XR_EVENT_UNAVAILABLE) {
            return nullptr;
        }
        THROW_XR(xr, "xrPollEvent");
    }

    void PollEvents(bool* exitRenderLoop, bool* requestRestart) override {
        *exitRenderLoop = *requestRestart = false;

        // Process all pending messages.
        while (const XrEventDataBaseHeader* event = TryReadNextEvent()) {
            switch (event->type) {
                case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: {
                    const auto& instanceLossPending = *reinterpret_cast<const XrEventDataInstanceLossPending*>(event);
                    Log::Write(Log::Level::Warning, Fmt("XrEventDataInstanceLossPending by %lld", instanceLossPending.lossTime));
                    *exitRenderLoop = true;
                    *requestRestart = true;
                    return;
                }
                case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
                    auto sessionStateChangedEvent = *reinterpret_cast<const XrEventDataSessionStateChanged*>(event);
                    HandleSessionStateChangedEvent(sessionStateChangedEvent, exitRenderLoop, requestRestart);
                    break;
                }
                case XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB:
                    arcadexr::xr::OnRuntimeRateChanged(
                        reinterpret_cast<const XrEventDataDisplayRefreshRateChangedFB*>(event)->toDisplayRefreshRate);
                    break;
                case XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED:
                    LogActionSourceName(m_input.grabAction, "Grab");
                    LogActionSourceName(m_input.quitAction, "Quit");
                    LogActionSourceName(m_input.poseAction, "Pose");
                    LogActionSourceName(m_input.vibrateAction, "Vibrate");
                    LogActionSourceName(m_input.triggerAction, "Trigger");
                    LogActionSourceName(m_input.pedalAction, "Pedal");
                    LogActionSourceName(m_input.startAction, "Start");
                    LogActionSourceName(m_input.coinAction, "Coin");
            LogActionSourceName(m_input.recenterScreenAction, "Recenter Screen");
            LogActionSourceName(m_input.crosshairAction, "Crosshair Mode");
                    break;
                case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING: {
                    const auto& change = *reinterpret_cast<const XrEventDataReferenceSpaceChangePending*>(event);
                    arcadexr::debug::LogRecenterEvent(to_string(change.referenceSpaceType), m_appSpaceType.c_str(),
                                                      (long long)change.changeTime, (long long)m_lastDisplayTime,
                                                      change.poseValid == XR_TRUE, change.poseInPreviousSpace);
                    // The arcade screen is expressed in the application space.
                    // Only a redefinition of THAT space invalidates it. Meta
                    // emits STAGE changes when the boundary is drawn, exited or
                    // re-scanned -- all of which happen while the player simply
                    // walks around -- and reacting to those moved the cabinet
                    // under the player's feet.
                    if (change.referenceSpaceType == m_appSpaceTypeEnum) {
                        m_recenterTime = change.changeTime;
                        Log::Write(Log::Level::Info, "TCVR_M9 arcade screen re-anchor scheduled: app space redefined");
                    } else {
                        Log::Write(Log::Level::Info,
                                   Fmt("TCVR_M9 ignoring %s change: the arcade screen lives in %s",
                                       to_string(change.referenceSpaceType), m_appSpaceType.c_str()));
                    }
                    break;
                }
                default: {
                    Log::Write(Log::Level::Verbose, Fmt("Ignoring event type %d", event->type));
                    break;
                }
            }
        }
    }

    void HandleSessionStateChangedEvent(const XrEventDataSessionStateChanged& stateChangedEvent, bool* exitRenderLoop,
                                        bool* requestRestart) {
        const XrSessionState oldState = m_sessionState;
        m_sessionState = stateChangedEvent.state;

        Log::Write(Log::Level::Info, Fmt("XrEventDataSessionStateChanged: state %s->%s session=%lld time=%lld", to_string(oldState),
                                         to_string(m_sessionState), stateChangedEvent.session, stateChangedEvent.time));

        if ((stateChangedEvent.session != XR_NULL_HANDLE) && (stateChangedEvent.session != m_session)) {
            Log::Write(Log::Level::Error, "XrEventDataSessionStateChanged for unknown session");
            return;
        }

        switch (m_sessionState) {
            case XR_SESSION_STATE_READY: {
                CHECK(m_session != XR_NULL_HANDLE);
                XrSessionBeginInfo sessionBeginInfo{XR_TYPE_SESSION_BEGIN_INFO};
                sessionBeginInfo.primaryViewConfigurationType = m_viewConfigType;
                CHECK_XRCMD(xrBeginSession(m_session, &sessionBeginInfo));
                m_sessionRunning = true;
                // Only now does the runtime answer xrEnumerateDisplayRefreshRatesFB:
                // asked before xrBeginSession it reports an empty list and 0 Hz.
                arcadexr::xr::Initialize(m_instance, m_session, m_supportsDisplayRefreshRate);
                arcadexr::xr::ApplyConfiguredMode();
                break;
            }
            case XR_SESSION_STATE_STOPPING: {
                CHECK(m_session != XR_NULL_HANDLE);
                m_sessionRunning = false;
                CHECK_XRCMD(xrEndSession(m_session))
                break;
            }
            case XR_SESSION_STATE_EXITING: {
                *exitRenderLoop = true;
                // Do not attempt to restart because user closed this session.
                *requestRestart = false;
                break;
            }
            case XR_SESSION_STATE_LOSS_PENDING: {
                *exitRenderLoop = true;
                // Poll for a new instance.
                *requestRestart = true;
                break;
            }
            default:
                break;
        }
    }

    void LogActionSourceName(XrAction action, const std::string& actionName) const {
        XrBoundSourcesForActionEnumerateInfo getInfo = {XR_TYPE_BOUND_SOURCES_FOR_ACTION_ENUMERATE_INFO};
        getInfo.action = action;
        uint32_t pathCount = 0;
        CHECK_XRCMD(xrEnumerateBoundSourcesForAction(m_session, &getInfo, 0, &pathCount, nullptr));
        std::vector<XrPath> paths(pathCount);
        CHECK_XRCMD(xrEnumerateBoundSourcesForAction(m_session, &getInfo, uint32_t(paths.size()), &pathCount, paths.data()));

        std::string sourceName;
        for (uint32_t i = 0; i < pathCount; ++i) {
            constexpr XrInputSourceLocalizedNameFlags all = XR_INPUT_SOURCE_LOCALIZED_NAME_USER_PATH_BIT |
                                                            XR_INPUT_SOURCE_LOCALIZED_NAME_INTERACTION_PROFILE_BIT |
                                                            XR_INPUT_SOURCE_LOCALIZED_NAME_COMPONENT_BIT;

            XrInputSourceLocalizedNameGetInfo nameInfo = {XR_TYPE_INPUT_SOURCE_LOCALIZED_NAME_GET_INFO};
            nameInfo.sourcePath = paths[i];
            nameInfo.whichComponents = all;

            uint32_t size = 0;
            CHECK_XRCMD(xrGetInputSourceLocalizedName(m_session, &nameInfo, 0, &size, nullptr));
            if (size < 1) {
                continue;
            }
            std::vector<char> grabSource(size);
            CHECK_XRCMD(xrGetInputSourceLocalizedName(m_session, &nameInfo, uint32_t(grabSource.size()), &size, grabSource.data()));
            if (!sourceName.empty()) {
                sourceName += " and ";
            }
            sourceName += "'";
            sourceName += std::string(grabSource.data(), size - 1);
            sourceName += "'";
        }

        Log::Write(Log::Level::Info,
                   Fmt("%s action is bound to %s", actionName.c_str(), ((!sourceName.empty()) ? sourceName.c_str() : "nothing")));
    }

    bool IsSessionRunning() const override { return m_sessionRunning; }

    bool IsSessionFocused() const override { return m_sessionState == XR_SESSION_STATE_FOCUSED; }

    void PollActions() override {
        m_input.handActive = {{XR_FALSE, XR_FALSE}};

        // Sync actions
        const XrActiveActionSet activeActionSet{m_input.actionSet, XR_NULL_PATH};
        XrActionsSyncInfo syncInfo{XR_TYPE_ACTIONS_SYNC_INFO};
        syncInfo.countActiveActionSets = 1;
        syncInfo.activeActionSets = &activeActionSet;
        CHECK_XRCMD(xrSyncActions(m_session, &syncInfo));

        if (!m_loggedActionBindings && IsSessionFocused()) {
            LogActionSourceName(m_input.triggerAction, "Trigger");
            LogActionSourceName(m_input.pedalAction, "Pedal");
            LogActionSourceName(m_input.startAction, "Start");
            LogActionSourceName(m_input.coinAction, "Coin");
            LogActionSourceName(m_input.recenterScreenAction, "Recenter Screen");
            LogActionSourceName(m_input.crosshairAction, "Crosshair Mode");
            m_loggedActionBindings = true;
        }

        // Get pose and grab action state and start haptic vibrate when hand is 90% squeezed.
        for (auto hand : {Side::LEFT, Side::RIGHT}) {
            XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO};
            getInfo.action = m_input.grabAction;
            getInfo.subactionPath = m_input.handSubactionPath[hand];

            XrActionStateFloat grabValue{XR_TYPE_ACTION_STATE_FLOAT};
            CHECK_XRCMD(xrGetActionStateFloat(m_session, &getInfo, &grabValue));
            if (grabValue.isActive == XR_TRUE) {
                // Scale the rendered hand by 1.0f (open) to 0.5f (fully squeezed).
                m_input.handScale[hand] = 1.0f - 0.5f * grabValue.currentState;
                if (grabValue.currentState > 0.9f) {
                    XrHapticVibration vibration{XR_TYPE_HAPTIC_VIBRATION};
                    vibration.amplitude = 0.5;
                    vibration.duration = XR_MIN_HAPTIC_DURATION;
                    vibration.frequency = XR_FREQUENCY_UNSPECIFIED;

                    XrHapticActionInfo hapticActionInfo{XR_TYPE_HAPTIC_ACTION_INFO};
                    hapticActionInfo.action = m_input.vibrateAction;
                    hapticActionInfo.subactionPath = m_input.handSubactionPath[hand];
                    CHECK_XRCMD(xrApplyHapticFeedback(m_session, &hapticActionInfo, (XrHapticBaseHeader*)&vibration));
                }
            }

            getInfo.action = m_input.poseAction;
            XrActionStatePose poseState{XR_TYPE_ACTION_STATE_POSE};
            CHECK_XRCMD(xrGetActionStatePose(m_session, &getInfo, &poseState));
            m_input.handActive[hand] = poseState.isActive;
        }

        XrActionStateGetInfo triggerInfo{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, m_input.triggerAction,
                                         m_input.handSubactionPath[Side::RIGHT]};
        XrActionStateFloat triggerValue{XR_TYPE_ACTION_STATE_FLOAT};
        CHECK_XRCMD(xrGetActionStateFloat(m_session, &triggerInfo, &triggerValue));
        bool const triggerPressed = triggerValue.isActive == XR_TRUE && triggerValue.currentState > 0.5f;
        if (!triggerPressed) m_blockTriggerUntilRelease = false;
        auto reportDigital = [](const char *id, bool pressed, bool &previous) {
            if (pressed != previous) {
                Log::Write(Log::Level::Info, Fmt("TCVR_M6 %s=%s", id, pressed ? "pressed" : "released"));
                previous = pressed;
            }
            arcadexr::input::SetDigital(id, pressed);
        };
        m_captureCalibration = m_calibrating && triggerPressed && !m_triggerHeld;
        m_triggerHeld = triggerPressed;
        reportDigital("trigger", triggerPressed && !m_calibrating && !m_blockTriggerUntilRelease, m_input.lastTrigger);

        auto readButton = [this](XrAction action, int hand) {
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, action, m_input.handSubactionPath[hand]};
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            CHECK_XRCMD(xrGetActionStateBoolean(m_session, &info, &state));
            return state.isActive == XR_TRUE && state.currentState == XR_TRUE;
        };
        reportDigital("pedal", readButton(m_input.pedalAction, Side::RIGHT), m_input.lastPedal);
        reportDigital("start", readButton(m_input.startAction, Side::RIGHT), m_input.lastStart);
        reportDigital("coin", readButton(m_input.coinAction, Side::LEFT), m_input.lastCoin);

        // There were no subaction paths specified for the quit action, because we don't care which hand did it.
        XrActionStateGetInfo getInfo{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, m_input.quitAction, XR_NULL_PATH};
        XrActionStateBoolean quitValue{XR_TYPE_ACTION_STATE_BOOLEAN};
        CHECK_XRCMD(xrGetActionStateBoolean(m_session, &getInfo, &quitValue));
        if ((quitValue.isActive == XR_TRUE) && (quitValue.changedSinceLastSync == XR_TRUE) && (quitValue.currentState == XR_TRUE)) {
            m_calibrating = !m_calibrating;
            // Calibrating the gun used to reset the screen as well. Those are
            // two different intentions and now have two different buttons.
            arcadexr::input::SetDigital("trigger", false);
            Log::Write(Log::Level::Info, m_calibrating ? "TCVR_M8 calibration: point at centre then pull trigger" : "TCVR_M8 calibration cancelled");
        }

        auto pressedOnce = [this](XrAction action) {
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO, nullptr, action, XR_NULL_PATH};
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            CHECK_XRCMD(xrGetActionStateBoolean(m_session, &info, &state));
            return state.isActive == XR_TRUE && state.changedSinceLastSync == XR_TRUE && state.currentState == XR_TRUE;
        };

        if (pressedOnce(m_input.recenterScreenAction)) {
            m_virtualScreenInitialized = false;
            m_screenPlacementReason = "explicit";
            Log::Write(Log::Level::Info, "TCVR_M9 arcade screen re-anchor requested by the player");
        }

        if (pressedOnce(m_input.crosshairAction)) {
            const std::string current = arcadexr::config::GetString("crosshair", "visible");
            const char* next = (current == "visible") ? "calibration" : (current == "calibration" ? "hidden" : "visible");
            arcadexr::config::Set("crosshair", next);
            Log::Write(Log::Level::Info, Fmt("TCVR_M10 crosshair mode %s -> %s", current.c_str(), next));
        }
    }

    void RenderFrame() override {
        CHECK(m_session != XR_NULL_HANDLE);

        XrFrameWaitInfo frameWaitInfo{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState frameState{XR_TYPE_FRAME_STATE};
        CHECK_XRCMD(xrWaitFrame(m_session, &frameWaitInfo, &frameState));

        XrFrameBeginInfo frameBeginInfo{XR_TYPE_FRAME_BEGIN_INFO};
        CHECK_XRCMD(xrBeginFrame(m_session, &frameBeginInfo));

        std::vector<XrCompositionLayerBaseHeader*> layers;
        XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        std::vector<XrCompositionLayerProjectionView> projectionLayerViews;
        std::vector<XrCompositionLayerDepthInfoKHR> depthInfos;
        if (frameState.shouldRender == XR_TRUE) {
            if (RenderLayer(frameState.predictedDisplayTime, projectionLayerViews, depthInfos, layer)) {
                layers.push_back(reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer));
            }
        }

        m_lastDisplayTime = frameState.predictedDisplayTime;
        arcadexr::xr::RetryWhileUnknown(IsSessionFocused());
        arcadexr::audio::LogStatsPeriodically();
        arcadexr::config::Poll();
        {
            const std::string mode = arcadexr::config::GetString("crosshair", "visible");
            m_crosshairMode = (mode == "hidden")        ? CrosshairMode::Hidden
                              : (mode == "calibration") ? CrosshairMode::CalibrationOnly
                                                        : CrosshairMode::Visible;
            m_gunLaser = arcadexr::config::GetInt("gun.laser", 0) != 0;
            m_gunBothHands = arcadexr::config::GetString("gun.hands", "right") == "both";
        }
        XrFrameEndInfo frameEndInfo{XR_TYPE_FRAME_END_INFO};
        frameEndInfo.displayTime = frameState.predictedDisplayTime;
        frameEndInfo.environmentBlendMode = m_blendMode;
        frameEndInfo.layerCount = (uint32_t)layers.size();
        frameEndInfo.layers = layers.data();
        CHECK_XRCMD(xrEndFrame(m_session, &frameEndInfo));
        if (!m_loggedFirstEndFrame) {
            Log::Write(Log::Level::Info, "TCVR_M0 xrEndFrame succeeded");
            m_loggedFirstEndFrame = true;
        }
    }

    bool RenderLayer(XrTime predictedDisplayTime, std::vector<XrCompositionLayerProjectionView>& projectionLayerViews,
                     std::vector<XrCompositionLayerDepthInfoKHR>& depthInfos, XrCompositionLayerProjection& layer) {
        XrResult res;

        XrViewState viewState{XR_TYPE_VIEW_STATE};
        uint32_t viewCapacityInput = (uint32_t)m_views.size();
        uint32_t viewCountOutput;

        XrViewLocateInfo viewLocateInfo{XR_TYPE_VIEW_LOCATE_INFO};
        viewLocateInfo.viewConfigurationType = m_viewConfigType;
        viewLocateInfo.displayTime = predictedDisplayTime;
        viewLocateInfo.space = m_appSpace;

        res = xrLocateViews(m_session, &viewLocateInfo, &viewState, viewCapacityInput, &viewCountOutput, m_views.data());
        CHECK_XRRESULT(res, "xrLocateViews");
        if ((viewState.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) == 0 ||
            (viewState.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) == 0) {
            return false;  // There is no valid tracking poses for the views.
        }
        if (!m_loggedValidHeadPose) {
            Log::Write(Log::Level::Info, "TCVR_M0 xrLocateViews returned a valid 6DoF head pose");
            m_loggedValidHeadPose = true;
        }

        CHECK(viewCountOutput > 0 && viewCountOutput == viewCapacityInput);
        CHECK(viewCountOutput == m_configViews.size());
        CHECK(viewCountOutput == m_swapchains.size());
        XrPosef head = m_views[0].pose;
        head.position = {};
        for (const auto& view : m_views) {
            head.position.x += view.pose.position.x / viewCountOutput;
            head.position.y += view.pose.position.y / viewCountOutput;
            head.position.z += view.pose.position.z / viewCountOutput;
        }
        if (m_recenterTime && predictedDisplayTime >= m_recenterTime) {
            arcadexr::debug::LogRecenterApplied((long long)m_recenterTime, (long long)predictedDisplayTime);
            m_virtualScreenInitialized = false;
            m_recenterTime = 0;
            m_screenPlacementReason = "recenter";
        }
        if (!m_virtualScreenInitialized) {
            const XrVector3f localForward{0, 0, -1};
            XrVector3f forward;
            XrQuaternionf_RotateVector3f(&forward, &head.orientation, &localForward);
            forward.y = 0;
            if (XrVector3f_Length(&forward) < 0.1f) forward = {0,0,-1};
            XrVector3f_Normalize(&forward);
            // Size and distance are settings. The lightgun mapping is purely
            // geometric -- ray against plane, then normalised by the plane's own
            // width and height -- so changing either never invalidates the gun
            // calibration, which only describes the barrel's axis.
            const float distance = arcadexr::config::GetFloat("screen.distance", 2.0f);
            const float width = arcadexr::config::GetFloat("screen.width", 3.0f);
            arcadexr::gun::ScreenPlane plane{
                {head.position.x + forward.x*distance, head.position.y, head.position.z + forward.z*distance},
                {-forward.z,0,forward.x}, {0,1,0}, {-forward.x,0,-forward.z}, width, width*0.75f};
            arcadexr::video::SetVirtualScreen(plane);
            m_virtualScreenInitialized = true;
            arcadexr::debug::LogScreenPlaced(plane, head, m_screenPlacementReason);
            m_screenPlacementReason = "startup";
            Log::Write(Log::Level::Info, "TCVR_M8 screen recentered vertical distance=2m");
        }
        arcadexr::gun::ScreenPlane screen;
        arcadexr::video::GetVirtualScreen(screen);
        // The cabinet is fixed in the world. The render loop reads the plane and
        // never writes it: walking towards, past or around the screen changes
        // nothing, exactly as with a real arcade machine. Re-anchoring happens
        // only when the application space itself is redefined, above.
        if (!m_loggedSpaceConfig) {
            m_loggedSpaceConfig = true;
            const auto clear = GetBackgroundClearColor();
            arcadexr::debug::LogConfig(m_appSpaceRequested.c_str(), m_appSpaceType.c_str(), to_string(m_blendMode),
                                       m_blendModesAvailable, m_referenceSpacesAvailable, clear.data(),
                                       (unsigned long long)(m_blendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND
                                                                ? (XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT |
                                                                   XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT)
                                                                : 0),
                                       0);
        }
        arcadexr::debug::LogFrameThrottled((long long)predictedDisplayTime, head, screen);

        projectionLayerViews.resize(viewCountOutput);
        if (m_supportsDepthLayer) depthInfos.resize(viewCountOutput);
        std::vector<Cube> cubes;
        cubes.reserve(64);
        arcadexr::gun::SetAimState({false,0.5f,0.5f,m_calibrating,false});
        bool gunTracked = false;
        for (auto hand : {Side::LEFT, Side::RIGHT}) {
            XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
            get.action = m_input.aimAction;
            get.subactionPath = m_input.handSubactionPath[hand];
            XrActionStatePose state{XR_TYPE_ACTION_STATE_POSE};
            CHECK_XRCMD(xrGetActionStatePose(m_session, &get, &state));
            if (!state.isActive || !IsSessionFocused()) continue;
            XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
            CHECK_XRCMD(xrLocateSpace(m_input.aimSpace[hand], m_appSpace, predictedDisplayTime, &location));
            const XrSpaceLocationFlags required = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            if ((location.locationFlags & required) != required) continue;
            if (!m_loggedValidHandPose[hand]) {
                Log::Write(Log::Level::Info, Fmt("TCVR_M8 %s aim pose valid; 3D gun active", hand == Side::RIGHT ? "right" : "left"));
                m_loggedValidHandPose[hand] = true;
            }
            if (hand == Side::RIGHT && m_captureCalibration) {
                m_captureCalibration = false;
                m_blockTriggerUntilRelease = true;
                XrQuaternionf offset;
                if (arcadexr::gun::CalibrationToTarget(location.pose,
                        {screen.center.x,screen.center.y,screen.center.z}, offset)) {
                    m_gunCalibration = offset;
                    bool saved = arcadexr::gun::SaveCalibration(offset);
                    m_calibrating = false;
                    Log::Write(Log::Level::Info, Fmt("TCVR_M8 VR calibration captured saved=%d", saved));
                } else {
                    Log::Write(Log::Level::Warning, "TCVR_M8 capture rejected: aim nearer to centre");
                }
            }
            const XrQuaternionf identity{0,0,0,1};
            const XrPosef pose = arcadexr::gun::GunPose(location.pose, hand == Side::RIGHT ? m_gunCalibration : identity);
            // Time Crisis has one gun. Drawing a second in the off hand was
            // wrong, and it is the aiming hand that carries it.
            const bool drawThisHand = (hand == Side::RIGHT) || m_gunBothHands;
            if (drawThisHand) {
                const XrVector3f xAxis{1, 0, 0};
                for (const auto& part : arcadexr::gun::gunParts) {
                    XrPosef partPose = pose;
                    if (part.pitch != 0.0f) {
                        XrQuaternionf tilt;
                        XrQuaternionf_CreateFromAxisAngle(&tilt, &xAxis, part.pitch);
                        XrQuaternionf_Multiply(&partPose.orientation, &pose.orientation, &tilt);
                    }
                    XrPosef_TransformVector3f(&partPose.position, &pose, &part.center);
                    cubes.push_back(Cube{partPose,part.size,part.color});
                }
            }
            if (hand != Side::RIGHT) continue;
            gunTracked = true;
            const auto muzzle = arcadexr::gun::Muzzle(pose);
            const XrVector3f forward{0,0,-1};
            XrVector3f direction;
            XrQuaternionf_RotateVector3f(&direction, &pose.orientation, &forward);
            const auto hit = arcadexr::gun::IntersectScreen(
                {{muzzle.x,muzzle.y,muzzle.z},{direction.x,direction.y,direction.z}}, screen);
            const bool onScreen = hit.intersects && hit.on_screen;
            arcadexr::input::SetAnalog("gun_x", onScreen ? hit.normalized_x : 0.0f);
            arcadexr::input::SetAnalog("gun_y", onScreen ? hit.normalized_y : 0.0f);
            arcadexr::input::SetDigital("gun_offscreen", !onScreen);
            // Hiding the reticle is a presentation choice only: the raycast
            // above has already been computed and sent to the game, so Hidden
            // mode stays exactly as playable as Visible.
            const bool showCrosshair = onScreen && (m_crosshairMode == CrosshairMode::Visible ||
                                                    (m_crosshairMode == CrosshairMode::CalibrationOnly && m_calibrating));
            arcadexr::gun::SetAimState({onScreen,hit.normalized_x,hit.normalized_y,m_calibrating,showCrosshair});
            // No laser by default: the 1995 cabinet's gun has none, and adding
            // one is a modern affectation the player noticed immediately. Kept
            // as an aid that can be switched on while calibrating.
            if (m_gunLaser) {
                float beamLength = 1.5f;
                if (onScreen) {
                    XrVector3f point{
                        screen.center.x + screen.right.x*(hit.normalized_x-.5f)*screen.width + screen.up.x*(.5f-hit.normalized_y)*screen.height,
                        screen.center.y + screen.right.y*(hit.normalized_x-.5f)*screen.width + screen.up.y*(.5f-hit.normalized_y)*screen.height,
                        screen.center.z + screen.right.z*(hit.normalized_x-.5f)*screen.width + screen.up.z*(.5f-hit.normalized_y)*screen.height};
                    XrVector3f delta{point.x-muzzle.x,point.y-muzzle.y,point.z-muzzle.z};
                    beamLength = XrVector3f_Length(&delta);
                }
                XrPosef beam = pose;
                beam.position = {muzzle.x+direction.x*beamLength*.5f,muzzle.y+direction.y*beamLength*.5f,muzzle.z+direction.z*beamLength*.5f};
                cubes.push_back(Cube{beam,{0.0025f,0.0025f,beamLength},{1.0f,0.15f,0.08f}});
            }
            if (!m_gunStateKnown || onScreen != m_lastGunOnScreen) {
                Log::Write(Log::Level::Info,Fmt("TCVR_M7 gun=%s x=%.3f y=%.3f",onScreen?"on_screen":"offscreen",hit.normalized_x,hit.normalized_y));
                m_gunStateKnown = true;
                m_lastGunOnScreen = onScreen;
            }
        }
        if (!gunTracked) {
            m_captureCalibration = false;
            arcadexr::input::SetAnalog("gun_x",0);
            arcadexr::input::SetAnalog("gun_y",0);
            arcadexr::input::SetDigital("gun_offscreen",true);
            arcadexr::input::SetDigital("trigger",false);
        }

        // Render view to the appropriate part of the swapchain image.
        for (uint32_t i = 0; i < viewCountOutput; i++) {
            // Each view has a separate swapchain which is acquired, rendered to, and released.
            const Swapchain viewSwapchain = m_swapchains[i];

            XrSwapchainImageAcquireInfo acquireInfo{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};

            uint32_t swapchainImageIndex;
            CHECK_XRCMD(xrAcquireSwapchainImage(viewSwapchain.handle, &acquireInfo, &swapchainImageIndex));

            XrSwapchainImageWaitInfo waitInfo{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            waitInfo.timeout = XR_INFINITE_DURATION;
            CHECK_XRCMD(xrWaitSwapchainImage(viewSwapchain.handle, &waitInfo));

            m_swapchainImages[viewSwapchain.handle]->AcquireAndWaitDepthSwapchainImage(swapchainImageIndex);

            projectionLayerViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
            projectionLayerViews[i].pose = m_views[i].pose;
            projectionLayerViews[i].fov = m_views[i].fov;
            projectionLayerViews[i].subImage.swapchain = viewSwapchain.handle;
            projectionLayerViews[i].subImage.imageRect.offset = {0, 0};
            projectionLayerViews[i].subImage.imageRect.extent = {viewSwapchain.width, viewSwapchain.height};

            if (m_supportsDepthLayer) {
                projectionLayerViews[i].next = &depthInfos[i];
                depthInfos[i].type = XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR;
                depthInfos[i].subImage.swapchain = m_depthSwapchains[i].handle;
                depthInfos[i].subImage.imageRect.offset = {0, 0};
                depthInfos[i].subImage.imageRect.extent = {m_depthSwapchains[i].width, m_depthSwapchains[i].height};
                depthInfos[i].minDepth = 0;
                depthInfos[i].maxDepth = 1;
                depthInfos[i].nearZ = 0.05f;
                depthInfos[i].farZ = 100.0f;
            }

            const XrSwapchainImageBaseHeader* const swapchainImage =
                m_swapchainImages[viewSwapchain.handle]->GetGenericColorImage(swapchainImageIndex);
            m_graphicsPlugin->RenderView(projectionLayerViews[i], swapchainImage, m_colorSwapchainFormat, cubes);

            XrSwapchainImageReleaseInfo releaseInfo{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            CHECK_XRCMD(xrReleaseSwapchainImage(viewSwapchain.handle, &releaseInfo));

            m_swapchainImages[viewSwapchain.handle]->ReleaseDepthSwapchainImage();
        }

        layer.space = m_appSpace;
        layer.layerFlags =
            m_blendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND
                ? XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT
                : 0;
        layer.viewCount = (uint32_t)projectionLayerViews.size();
        layer.views = projectionLayerViews.data();
        return true;
    }

   private:
    XrEnvironmentBlendMode m_blendMode{XR_ENVIRONMENT_BLEND_MODE_MAX_ENUM};
    XrViewConfigurationType m_viewConfigType{XR_VIEW_CONFIGURATION_TYPE_MAX_ENUM};

    std::shared_ptr<IPlatformPlugin> m_platformPlugin;
    std::shared_ptr<IGraphicsPlugin> m_graphicsPlugin;
    XrInstance m_instance{XR_NULL_HANDLE};
    XrSession m_session{XR_NULL_HANDLE};
    XrSpace m_appSpace{XR_NULL_HANDLE};
    XrSystemId m_systemId{XR_NULL_SYSTEM_ID};

    // We may still use a runtime allocated depth swapchain but not submit depth if false
    bool m_supportsDepthLayer{false};
    bool m_supportsDisplayRefreshRate{false};
    bool m_loggedFirstEndFrame{false};
    bool m_virtualScreenInitialized{false};
    enum class CrosshairMode { Visible, CalibrationOnly, Hidden };
    CrosshairMode m_crosshairMode{CrosshairMode::Visible};
    bool m_gunLaser{false};
    bool m_gunBothHands{false};
    XrTime m_recenterTime{0};
    XrReferenceSpaceType m_appSpaceTypeEnum{XR_REFERENCE_SPACE_TYPE_MAX_ENUM};
    XrTime m_lastDisplayTime{0};
    std::string m_appSpaceRequested;
    std::string m_appSpaceType{"<unset>"};
    std::string m_blendModesAvailable;
    std::string m_referenceSpacesAvailable;
    const char* m_screenPlacementReason{"startup"};
    bool m_loggedSpaceConfig{false};
    bool m_calibrating{false};
    bool m_captureCalibration{false};
    bool m_triggerHeld{false};
    bool m_blockTriggerUntilRelease{false};
    XrQuaternionf m_gunCalibration{0, 0, 0, 1};
    bool m_gunStateKnown{false};
    bool m_lastGunOnScreen{false};
    bool m_loggedValidHeadPose{false};
    bool m_loggedActionBindings{false};
    std::array<bool, Side::COUNT> m_loggedValidHandPose{{false, false}};

    std::vector<XrViewConfigurationView> m_configViews;
    std::vector<Swapchain> m_swapchains;
    std::vector<Swapchain> m_depthSwapchains;
    std::map<XrSwapchain, ISwapchainImageData*> m_swapchainImages;
    std::vector<XrView> m_views;
    int64_t m_colorSwapchainFormat{-1};
    int64_t m_depthSwapchainFormat{-1};

    std::vector<XrSpace> m_visualizedSpaces;

    // Application's current lifecycle state according to the runtime
    XrSessionState m_sessionState{XR_SESSION_STATE_UNKNOWN};
    bool m_sessionRunning{false};

    XrEventDataBuffer m_eventDataBuffer;
    InputState m_input;
};
}  // namespace

std::shared_ptr<IOpenXrProgram> CreateOpenXrProgram(const std::shared_ptr<IPlatformPlugin>& platformPlugin,
                                                    const std::shared_ptr<IGraphicsPlugin>& graphicsPlugin) {
    return std::make_shared<OpenXrProgram>(platformPlugin, graphicsPlugin);
}
