// Copyright (c) 2017-2026 The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pch.h"
#include "common.h"
#include "geometry.h"
#include "graphicsplugin.h"
#include "graphics_plugin_impl_helpers.h"

#ifdef XR_USE_GRAPHICS_API_OPENGL_ES

#include "common/gfxwrapper_opengl.h"
#include <common/xr_linear.h>
#include "framebuffer_bridge.h"
#include "scene_bridge.h"
#include "menu.h"
#include <android/log.h>
#include "stereo_renderer.h"
#include "gpu_renderer.h"
#include <chrono>
#include <cmath>
#include "virtual_screen.h"
#include "settings.h"
#include "game_profile.h"
#include "aim_state.h"
#include "gun_model.h"
#include "gun_mesh.h"
#include <cstring>
#include <cstdio>

#ifndef GL_BGRA_EXT
#define GL_BGRA_EXT 0x80E1
#endif
#include "aim_state.h"

#define GL(glcmd)                                                                                                    \
    {                                                                                                                \
        GLint err = glGetError();                                                                                    \
        if (err != GL_NO_ERROR) {                                                                                    \
            Log::Write(Log::Level::Error, Fmt("GLES error=%d, %s:%d", err, __FUNCTION__, __LINE__));                 \
        }                                                                                                            \
        glcmd;                                                                                                       \
        err = glGetError();                                                                                          \
        if (err != GL_NO_ERROR) {                                                                                    \
            Log::Write(Log::Level::Error, Fmt("GLES error=%d, cmd=%s, %s:%d", err, #glcmd, __FUNCTION__, __LINE__)); \
        }                                                                                                            \
    }

namespace {

// The version statement has come on first line.
static const char* VertexShaderGlsl = R"_(#version 320 es

    in vec3 VertexPos;
    in vec3 VertexColor;

    out vec3 PSVertexColor;

    uniform mat4 ModelViewProjection;

    void main() {
       gl_Position = ModelViewProjection * vec4(VertexPos, 1.0);
       PSVertexColor = VertexColor;
    }
    )_";

// The version statement has come on first line.
static const char* FragmentShaderGlsl = R"_(#version 320 es

    precision mediump float;

    in lowp vec3 PSVertexColor;
    uniform lowp vec3 ObjectTint;
    out lowp vec4 FragColor;

    // The sample encodes which face of the box a fragment belongs to in its
    // vertex colour: bright primaries for +X/+Y/+Z, quarter-strength ones for
    // the negatives. Turning that back into a per-face shading factor is what
    // makes a stack of boxes read as one solid object instead of a flat
    // silhouette -- top lit, sides stepped, underside dark.
    float faceShade(vec3 c) {
        if (c.g > 0.5) return 1.00;   // +Y, top
        if (c.r > 0.5) return 0.86;   // +X
        if (c.b > 0.5) return 0.78;   // +Z, towards the player
        if (c.g > 0.1) return 0.40;   // -Y, underside
        if (c.r > 0.1) return 0.64;   // -X
        return 0.54;                  // -Z
    }

    void main() {
       FragColor = vec4(ObjectTint * faceShade(PSVertexColor), 1);
    }
    )_";

static const char* ScreenVertexShaderGlsl = R"_(#version 320 es
    in vec3 VertexPos;
    in vec2 TexCoord;
    out vec2 PSTexCoord;
    uniform mat4 ModelViewProjection;
    void main() {
        gl_Position = ModelViewProjection * vec4(VertexPos, 1.0);
        PSTexCoord = TexCoord;
    }
    )_";

static const char* ScreenFragmentShaderGlsl = R"_(#version 320 es
    precision highp float;
    precision highp int;
    in vec2 PSTexCoord;
    uniform sampler2D ScreenTexture;
    uniform vec2 AimPoint;
    uniform int AimVisible;
    uniform int Calibrating;
    out lowp vec4 FragColor;
    void main() {
        vec4 color = texture(ScreenTexture, PSTexCoord);
        if (AimVisible != 0) {
            vec2 delta = PSTexCoord - AimPoint;
            delta.x *= 1.333333;
            float radial = length(delta);
            bool ring = radial > 0.011 && radial < 0.015;
            bool vertical = abs(delta.x) < 0.002 && abs(delta.y) < 0.024;
            bool horizontal = abs(delta.y) < 0.002 && abs(delta.x) < 0.024;
            if (ring || vertical || horizontal)
                color = vec4(1.0, 0.12, 0.08, 1.0);
        }
        if (Calibrating != 0) {
            vec2 d = (PSTexCoord - vec2(0.5)) * vec2(1.333333,1.0);
            float r = length(d);
            if ((r > 0.021 && r < 0.029) || (abs(d.x)<0.003 && abs(d.y)<0.04) ||
                (abs(d.y)<0.003 && abs(d.x)<0.04)) color = vec4(0.1,1.0,1.0,1.0);
        }
        FragColor = vec4(color.rgb, 1.0);
    }
    )_";

// Upscale pass. Runs once per EMULATED frame -- sixty times a second -- not
// once per presented XR frame, which is what keeps the arcade clock and the
// presentation clock independent. It magnifies the 640x480 source into an
// offscreen texture at an integer multiple, with a selectable filter and an
// optional unsharp mask, and the arcade quad then samples that texture at
// whatever rate the headset presents.
//
// Every filter here is one hardware bilinear tap on a LINEAR-filtered source,
// with the coordinate remapped beforehand. That is deliberate: a multi-tap
// reconstruction at 2560x1920 and 60 Hz is gigataps per second on a chip that
// also has an emulator to run, and the point of this pass is to make the image
// better without taking anything back from MAME.
static const char* UpscaleVertexShaderGlsl = R"_(#version 320 es
    in vec3 VertexPos;
    out vec2 PSTexCoord;
    void main() {
        gl_Position = vec4(VertexPos.xy * 2.0, 0.0, 1.0);
        PSTexCoord = VertexPos.xy + 0.5;
    }
    )_";

static const char* UpscaleFragmentShaderGlsl = R"_(#version 320 es
    precision highp float;
    in vec2 PSTexCoord;
    uniform sampler2D SourceTexture;
    uniform vec2 SourceSize;      // texels of the emulated framebuffer
    uniform vec2 TargetSize;      // texels of the upscaled texture
    uniform int Filter;           // 0 nearest, 1 bilinear, 2 sharp bilinear, 3 Catmull-Rom
    uniform float Sharpen;        // 0 = off
    out vec4 FragColor;

    vec3 sampleSource(vec2 uv) { return texture(SourceTexture, uv).rgb; }

    // Catmull-Rom bicubic, nine bilinear taps. Time Crisis is rendered 3D, not
    // pixel art, so the right reconstruction is a smooth interpolating one: it
    // removes the square edges of magnified texels without the mush of plain
    // bilinear, because it keeps a mild overshoot at real edges. Nine taps
    // rather than sixteen by folding each pair of neighbours into one weighted
    // bilinear fetch.
    vec3 catmullRom(vec2 uv) {
        vec2 position = uv * SourceSize;
        vec2 centre = floor(position - 0.5) + 0.5;
        vec2 f = position - centre;

        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
        vec2 w3 = f * f * (-0.5 + 0.5 * f);

        vec2 w12 = w1 + w2;
        vec2 middle = (centre + w2 / w12) / SourceSize;
        vec2 first = (centre - 1.0) / SourceSize;
        vec2 last = (centre + 2.0) / SourceSize;

        vec3 result = vec3(0.0);
        result += sampleSource(vec2(first.x,  first.y))  * w0.x  * w0.y;
        result += sampleSource(vec2(middle.x, first.y))  * w12.x * w0.y;
        result += sampleSource(vec2(last.x,   first.y))  * w3.x  * w0.y;
        result += sampleSource(vec2(first.x,  middle.y)) * w0.x  * w12.y;
        result += sampleSource(vec2(middle.x, middle.y)) * w12.x * w12.y;
        result += sampleSource(vec2(last.x,   middle.y)) * w3.x  * w12.y;
        result += sampleSource(vec2(first.x,  last.y))   * w0.x  * w3.y;
        result += sampleSource(vec2(middle.x, last.y))   * w12.x * w3.y;
        result += sampleSource(vec2(last.x,   last.y))   * w3.x  * w3.y;
        return clamp(result, 0.0, 1.0);
    }

    void main() {
        vec2 uv = PSTexCoord;
        if (Filter == 3) {
            vec3 smoothed = catmullRom(uv);
            if (Sharpen > 0.0) {
                vec2 step = 1.0 / SourceSize;
                vec3 blur = sampleSource(uv + vec2(step.x, 0.0)) + sampleSource(uv - vec2(step.x, 0.0)) +
                            sampleSource(uv + vec2(0.0, step.y)) + sampleSource(uv - vec2(0.0, step.y));
                smoothed = clamp(smoothed + Sharpen * (smoothed - blur * 0.25), 0.0, 1.0);
            }
            FragColor = vec4(smoothed, 1.0);
            return;
        }
        if (Filter == 0) {
            // Snap to the texel centre: exact pixels, with the blockiness and
            // the shimmer that come with them.
            uv = (floor(uv * SourceSize) + 0.5) / SourceSize;
        } else if (Filter == 2) {
            // Sharp bilinear: keep each source texel flat across its own area
            // and confine the blend to a ramp one target pixel wide, so edges
            // stay crisp without the stair-stepping of nearest.
            vec2 texel = uv * SourceSize;
            vec2 centre = floor(texel) + 0.5;
            vec2 ramp = SourceSize / TargetSize;   // source texels per target pixel
            vec2 offset = clamp((texel - centre) / max(ramp, vec2(1e-6)), -0.5, 0.5);
            uv = (centre + offset * ramp) / SourceSize;
        }
        vec3 color = sampleSource(uv);

        if (Sharpen > 0.0) {
            // Unsharp mask against the four direct neighbours, one source texel
            // away, so the amount means the same thing at every scale factor.
            vec2 step = 1.0 / SourceSize;
            vec3 blur = sampleSource(uv + vec2(step.x, 0.0)) + sampleSource(uv - vec2(step.x, 0.0)) +
                        sampleSource(uv + vec2(0.0, step.y)) + sampleSource(uv - vec2(0.0, step.y));
            color = clamp(color + Sharpen * (color - blur * 0.25), 0.0, 1.0);
        }
        FragColor = vec4(color, 1.0);
    }
    )_";

// The gun: one static mesh in gun-local space, one draw call per eye. Colour
// and per-face shading are baked into the vertices at build time, so the
// fragment shader has nothing to compute.
static const char* GunVertexShaderGlsl = R"_(#version 320 es
    in vec3 VertexPos;
    in vec3 VertexColor;
    uniform mat4 ModelViewProjection;
    out vec3 GunColor;
    void main() {
        gl_Position = ModelViewProjection * vec4(VertexPos, 1.0);
        GunColor = VertexColor;
    }
    )_";

static const char* GunFragmentShaderGlsl = R"_(#version 320 es
    precision mediump float;
    in vec3 GunColor;
    out vec4 FragColor;
    void main() { FragColor = vec4(GunColor, 1.0); }
    )_";

// Edge-aware pass: FXAA over the upscaled image. Bicubic reconstruction turns
// each source texel into a smooth 4x4 block, but a polygon or glyph edge is
// still a staircase at that 4-pixel scale. FXAA finds high-contrast edges by
// luma, walks along them and blends across, so the staircase softens while a
// textured surface -- low contrast between neighbours -- is left alone. Nine
// taps; it runs at the arcade rate, not the presentation rate.
static const char* EdgeFragmentShaderGlsl = R"_(#version 320 es
    precision highp float;
    in vec2 PSTexCoord;
    uniform sampler2D SourceTexture;
    uniform vec2 TargetSize;
    out vec4 FragColor;
    float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
    void main() {
        vec2 rcp = 1.0 / TargetSize;
        vec2 uv = PSTexCoord;
        vec3 rgbNW = texture(SourceTexture, uv + vec2(-1.0, -1.0) * rcp).rgb;
        vec3 rgbNE = texture(SourceTexture, uv + vec2( 1.0, -1.0) * rcp).rgb;
        vec3 rgbSW = texture(SourceTexture, uv + vec2(-1.0,  1.0) * rcp).rgb;
        vec3 rgbSE = texture(SourceTexture, uv + vec2( 1.0,  1.0) * rcp).rgb;
        vec3 rgbM  = texture(SourceTexture, uv).rgb;
        float lNW = luma(rgbNW), lNE = luma(rgbNE), lSW = luma(rgbSW), lSE = luma(rgbSE), lM = luma(rgbM);
        float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
        float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
        vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
        float dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * (1.0 / 8.0), 1.0 / 128.0);
        float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
        dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * rcp;
        vec3 rgbA = 0.5 * (texture(SourceTexture, uv + dir * (1.0 / 3.0 - 0.5)).rgb +
                           texture(SourceTexture, uv + dir * (2.0 / 3.0 - 0.5)).rgb);
        vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(SourceTexture, uv + dir * -0.5).rgb +
                                         texture(SourceTexture, uv + dir *  0.5).rgb);
        float lB = luma(rgbB);
        FragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
    }
    )_";

// Dirt Dash deliberately alternates some opaque black effect layers (notably
// the car shadow) at the arcade frame rate. A CRT integrates those flashes;
// sample-and-hold XR makes them painfully distinct. This pass reconstructs
// that integration only where two consecutive frames differ strongly and at
// least one of them is near black. Normal motion and coloured detail remain
// the current frame, so this is not whole-frame motion blur.
static const char* DarkFlickerFragmentShaderGlsl = R"_(#version 320 es
    precision highp float;
    in vec2 PSTexCoord;
    uniform sampler2D CurrentTexture;
    uniform sampler2D PreviousTexture;
    uniform int HavePrevious;
    out vec4 FragColor;
    float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
    void main() {
        vec3 current = texture(CurrentTexture, PSTexCoord).rgb;
        if (HavePrevious == 0) { FragColor = vec4(current, 1.0); return; }
        vec3 previous = texture(PreviousTexture, PSTexCoord).rgb;
        float darkest = min(luma(current), luma(previous));
        float change = max(max(abs(current.r - previous.r), abs(current.g - previous.g)), abs(current.b - previous.b));
        float darkGate = 1.0 - smoothstep(0.08, 0.18, darkest);
        float changeGate = smoothstep(0.12, 0.28, change);
        float integrate = darkGate * changeGate;
        FragColor = vec4(mix(current, (current + previous) * 0.5, integrate), 1.0);
    }
    )_";

// Lit gun: per-face normals, a fixed key light from above and slightly in
// front, a fill from below so the underside is never black, and a Blinn
// highlight from the actual eye so the metal reads as metal as the gun turns.
static const char* GunLitVertexShaderGlsl = R"_(#version 320 es
    in vec3 VertexPos;
    in vec3 VertexNormal;
    in vec3 VertexColor;
    uniform mat4 ModelViewProjection;
    uniform mat4 Model;
    out vec3 WorldPos;
    out vec3 WorldNormal;
    out vec3 GunColor;
    void main() {
        vec4 world = Model * vec4(VertexPos, 1.0);
        gl_Position = ModelViewProjection * vec4(VertexPos, 1.0);
        WorldPos = world.xyz;
        WorldNormal = mat3(Model) * VertexNormal;
        GunColor = VertexColor;
    }
    )_";

static const char* GunLitFragmentShaderGlsl = R"_(#version 320 es
    precision mediump float;
    in vec3 WorldPos;
    in vec3 WorldNormal;
    in vec3 GunColor;
    uniform vec3 EyePos;
    out vec4 FragColor;
    void main() {
        vec3 n = normalize(WorldNormal);
        vec3 key = normalize(vec3(0.35, 0.85, 0.40));
        vec3 fill = normalize(vec3(-0.3, -0.6, -0.5));
        vec3 v = normalize(EyePos - WorldPos);
        float diff = max(dot(n, key), 0.0) * 0.85 + max(dot(n, fill), 0.0) * 0.20;
        vec3 h = normalize(key + v);
        float spec = pow(max(dot(n, h), 0.0), 40.0) * 0.35;
        vec3 c = GunColor * (0.22 + diff) + vec3(spec);
        FragColor = vec4(c, 1.0);
    }
    )_";

struct ScreenVertex {
    XrVector3f Position;
    XrVector2f TexCoord;
};

constexpr ScreenVertex c_screenVertices[] = {
    {{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f}},
    {{0.5f, -0.5f, 0.0f}, {1.0f, 1.0f}},
    {{0.5f, 0.5f, 0.0f}, {1.0f, 0.0f}},
    {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f}},
};

constexpr GLushort c_screenIndices[] = {0, 1, 2, 0, 2, 3};

struct OpenGLESGraphicsPlugin : public IGraphicsPlugin {
    OpenGLESGraphicsPlugin() {}

    OpenGLESGraphicsPlugin(const OpenGLESGraphicsPlugin&) = delete;
    OpenGLESGraphicsPlugin& operator=(const OpenGLESGraphicsPlugin&) = delete;
    OpenGLESGraphicsPlugin(OpenGLESGraphicsPlugin&&) = delete;
    OpenGLESGraphicsPlugin& operator=(OpenGLESGraphicsPlugin&&) = delete;

    ~OpenGLESGraphicsPlugin() override {
        if (m_swapchainFramebuffer != 0) {
            glDeleteFramebuffers(1, &m_swapchainFramebuffer);
        }
        if (m_program != 0) {
            glDeleteProgram(m_program);
        }
        if (m_screenProgram != 0) {
            glDeleteProgram(m_screenProgram);
        }
        if (m_vao != 0) {
            glDeleteVertexArrays(1, &m_vao);
        }
        if (m_cubeVertexBuffer != 0) {
            glDeleteBuffers(1, &m_cubeVertexBuffer);
        }
        if (m_cubeIndexBuffer != 0) {
            glDeleteBuffers(1, &m_cubeIndexBuffer);
        }
        if (m_upscaleProgram != 0) {
            glDeleteProgram(m_upscaleProgram);
        }
        if (m_gunProgram != 0) glDeleteProgram(m_gunProgram);
        if (m_gunLitProgram != 0) glDeleteProgram(m_gunLitProgram);
        if (m_gunLitVao != 0) glDeleteVertexArrays(1, &m_gunLitVao);
        if (m_gunLitVertexBuffer != 0) glDeleteBuffers(1, &m_gunLitVertexBuffer);
        if (m_gunLitIndexBuffer != 0) glDeleteBuffers(1, &m_gunLitIndexBuffer);
        if (m_edgeProgram != 0) glDeleteProgram(m_edgeProgram);
        if (m_edgeTexture != 0) glDeleteTextures(1, &m_edgeTexture);
        if (m_darkFlickerProgram != 0) glDeleteProgram(m_darkFlickerProgram);
        glDeleteTextures(4, &m_sceneTemporalRaw[0][0]);
        glDeleteTextures(4, &m_immersiveTemporalRaw[0][0]);
        if (m_gunVao != 0) glDeleteVertexArrays(1, &m_gunVao);
        if (m_gunVertexBuffer != 0) glDeleteBuffers(1, &m_gunVertexBuffer);
        if (m_upscaleVao != 0) {
            glDeleteVertexArrays(1, &m_upscaleVao);
        }
        if (m_upscaleFramebuffer != 0) {
            glDeleteFramebuffers(1, &m_upscaleFramebuffer);
        }
        if (m_upscaleTexture != 0) {
            glDeleteTextures(1, &m_upscaleTexture);
        }
        if (m_screenTexture != 0) {
            glDeleteTextures(1, &m_screenTexture);
        }
        if (m_screenVao != 0) {
            glDeleteVertexArrays(1, &m_screenVao);
        }
        if (m_menuVao != 0) glDeleteVertexArrays(1, &m_menuVao);
        if (m_screenVertexBuffer != 0) {
            glDeleteBuffers(1, &m_screenVertexBuffer);
        }
        if (m_screenIndexBuffer != 0) {
            glDeleteBuffers(1, &m_screenIndexBuffer);
        }

        ksGpuWindow_Destroy(&window);
    }

    std::vector<std::string> GetInstanceExtensions() const override { return {XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME}; }

    ksGpuWindow window{};

    void DebugMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message) {
        (void)source;
        (void)type;
        (void)id;
        (void)severity;
        Log::Write(Log::Level::Info, "GLES Debug: " + std::string(message, 0, length));
    }

    void InitializeDevice(XrInstance instance, XrSystemId systemId) override {
        // Extension function must be loaded by name
        PFN_xrGetOpenGLESGraphicsRequirementsKHR pfnGetOpenGLESGraphicsRequirementsKHR = nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance, "xrGetOpenGLESGraphicsRequirementsKHR",
                                          reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetOpenGLESGraphicsRequirementsKHR)));

        XrGraphicsRequirementsOpenGLESKHR graphicsRequirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        CHECK_XRCMD(pfnGetOpenGLESGraphicsRequirementsKHR(instance, systemId, &graphicsRequirements));

        // Initialize the gl extensions. Note we have to open a window.
        ksDriverInstance driverInstance{};
        ksGpuQueueInfo queueInfo{};
        ksGpuSurfaceColorFormat colorFormat{KS_GPU_SURFACE_COLOR_FORMAT_B8G8R8A8};
        ksGpuSurfaceDepthFormat depthFormat{KS_GPU_SURFACE_DEPTH_FORMAT_D24};
        ksGpuSampleCount sampleCount{KS_GPU_SAMPLE_COUNT_1};
        if (!ksGpuWindow_Create(&window, &driverInstance, &queueInfo, 0, colorFormat, depthFormat, sampleCount, 640, 480, false)) {
            THROW("Unable to create GL context");
        }

        GLint major = 0;
        GLint minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);

        const XrVersion desiredApiVersion = XR_MAKE_VERSION(major, minor, 0);
        if (graphicsRequirements.minApiVersionSupported > desiredApiVersion) {
            THROW("Runtime does not support desired Graphics API and/or version");
        }

        m_contextApiMajorVersion = major;

#if defined(XR_USE_PLATFORM_ANDROID)
        m_graphicsBinding.display = window.context.dpy;
        m_graphicsBinding.config = (EGLConfig)0;
        m_graphicsBinding.context = window.context.ctx;
#endif

        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(
            [](GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message,
               const void* userParam) {
                ((OpenGLESGraphicsPlugin*)userParam)->DebugMessageCallback(source, type, id, severity, length, message);
            },
            this);

        InitializeResources();
    }

    void InitializeResources() {
        s_self = this;
        arcadexr::gun::SetSceneAim(&SceneAimTrampoline);
        {
            static const char* vs = R"_(#version 320 es
                in vec3 VertexPos; in vec2 TexCoord; out vec2 uv; uniform mat4 Mvp;
                void main() { gl_Position = Mvp * vec4(VertexPos, 1.0); uv = TexCoord; })_";
            static const char* fs = R"_(#version 320 es
                precision mediump float; in vec2 uv; uniform sampler2D Tex; out vec4 color;
                void main() { color = texture(Tex, uv); })_";
            GLuint v = glCreateShader(GL_VERTEX_SHADER); glShaderSource(v, 1, &vs, nullptr); glCompileShader(v); CheckShader(v);
            GLuint f = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(f, 1, &fs, nullptr); glCompileShader(f); CheckShader(f);
            m_menuProgram = glCreateProgram(); glAttachShader(m_menuProgram, v); glAttachShader(m_menuProgram, f); glLinkProgram(m_menuProgram); CheckProgram(m_menuProgram);
            glDeleteShader(v); glDeleteShader(f);
            m_menuMvpLocation = glGetUniformLocation(m_menuProgram, "Mvp");
            m_menuTexLocation = glGetUniformLocation(m_menuProgram, "Tex");
            glGenTextures(1, &m_menuTexture);
            glBindTexture(GL_TEXTURE_2D, m_menuTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glGenFramebuffers(1, &m_swapchainFramebuffer);

        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &VertexShaderGlsl, nullptr);
        glCompileShader(vertexShader);
        CheckShader(vertexShader);

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &FragmentShaderGlsl, nullptr);
        glCompileShader(fragmentShader);
        CheckShader(fragmentShader);

        m_program = glCreateProgram();
        glAttachShader(m_program, vertexShader);
        glAttachShader(m_program, fragmentShader);
        glLinkProgram(m_program);
        CheckProgram(m_program);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        m_modelViewProjectionUniformLocation = glGetUniformLocation(m_program, "ModelViewProjection");
        m_objectTintLocation = glGetUniformLocation(m_program, "ObjectTint");

        m_vertexAttribCoords = glGetAttribLocation(m_program, "VertexPos");
        m_vertexAttribColor = glGetAttribLocation(m_program, "VertexColor");

        glGenBuffers(1, &m_cubeVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_cubeVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(Geometry::c_cubeVertices), Geometry::c_cubeVertices, GL_STATIC_DRAW);

        glGenBuffers(1, &m_cubeIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_cubeIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Geometry::c_cubeIndices), Geometry::c_cubeIndices, GL_STATIC_DRAW);

        glGenVertexArrays(1, &m_vao);
        glBindVertexArray(m_vao);
        glEnableVertexAttribArray(m_vertexAttribCoords);
        glEnableVertexAttribArray(m_vertexAttribColor);
        glBindBuffer(GL_ARRAY_BUFFER, m_cubeVertexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_cubeIndexBuffer);
        glVertexAttribPointer(m_vertexAttribCoords, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex), nullptr);
        glVertexAttribPointer(m_vertexAttribColor, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex),
                              reinterpret_cast<const void*>(sizeof(XrVector3f)));
        // Unbind before anything else touches a buffer binding.
        //
        // The element array binding is part of vertex array object state, not
        // global state. Leaving this one bound meant that creating the screen's
        // index buffer below silently replaced the cube's -- so every cube then
        // drew from a six-index buffer instead of its own thirty-six, which
        // renders exactly one face of each box. That is what "you only see the
        // left of the gun, nothing in the middle, nothing on its right" was.
        glBindVertexArray(0);

        GLuint screenVertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(screenVertexShader, 1, &ScreenVertexShaderGlsl, nullptr);
        glCompileShader(screenVertexShader);
        CheckShader(screenVertexShader);
        GLuint screenFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(screenFragmentShader, 1, &ScreenFragmentShaderGlsl, nullptr);
        glCompileShader(screenFragmentShader);
        CheckShader(screenFragmentShader);
        m_screenProgram = glCreateProgram();
        glAttachShader(m_screenProgram, screenVertexShader);
        glAttachShader(m_screenProgram, screenFragmentShader);
        glLinkProgram(m_screenProgram);
        CheckProgram(m_screenProgram);
        glDeleteShader(screenVertexShader);
        glDeleteShader(screenFragmentShader);
        m_screenMvpUniformLocation = glGetUniformLocation(m_screenProgram, "ModelViewProjection");
        m_screenTextureUniformLocation = glGetUniformLocation(m_screenProgram, "ScreenTexture");
        m_screenAimPointUniformLocation = glGetUniformLocation(m_screenProgram, "AimPoint");
        m_screenAimVisibleUniformLocation = glGetUniformLocation(m_screenProgram, "AimVisible");
        m_screenCalibratingLocation = glGetUniformLocation(m_screenProgram, "Calibrating");
        const GLint screenPosition = glGetAttribLocation(m_screenProgram, "VertexPos");
        const GLint screenTexCoord = glGetAttribLocation(m_screenProgram, "TexCoord");

        glGenBuffers(1, &m_screenVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, m_screenVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(c_screenVertices), c_screenVertices, GL_STATIC_DRAW);
        glGenBuffers(1, &m_screenIndexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenIndexBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(c_screenIndices), c_screenIndices, GL_STATIC_DRAW);
        glGenVertexArrays(1, &m_screenVao);
        glBindVertexArray(m_screenVao);
        glEnableVertexAttribArray(screenPosition);
        glEnableVertexAttribArray(screenTexCoord);
        glBindBuffer(GL_ARRAY_BUFFER, m_screenVertexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenIndexBuffer);
        glVertexAttribPointer(screenPosition, 3, GL_FLOAT, GL_FALSE, sizeof(ScreenVertex), nullptr);
        glVertexAttribPointer(screenTexCoord, 2, GL_FLOAT, GL_FALSE, sizeof(ScreenVertex),
                              reinterpret_cast<const void*>(sizeof(XrVector3f)));
        glBindVertexArray(0);
        // Attribute locations are assigned per linked shader program. Sharing
        // the screen VAO with the menu shader silently drops the panel on GPUs
        // that choose different locations for VertexPos and TexCoord.
        const GLint menuPosition = glGetAttribLocation(m_menuProgram, "VertexPos");
        const GLint menuTexCoord = glGetAttribLocation(m_menuProgram, "TexCoord");
        glGenVertexArrays(1, &m_menuVao);
        glBindVertexArray(m_menuVao);
        glBindBuffer(GL_ARRAY_BUFFER, m_screenVertexBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenIndexBuffer);
        glEnableVertexAttribArray(menuPosition);
        glEnableVertexAttribArray(menuTexCoord);
        glVertexAttribPointer(menuPosition, 3, GL_FLOAT, GL_FALSE, sizeof(ScreenVertex), nullptr);
        glVertexAttribPointer(menuTexCoord, 2, GL_FLOAT, GL_FALSE, sizeof(ScreenVertex),
                              reinterpret_cast<const void*>(sizeof(XrVector3f)));
        glBindVertexArray(0);
        Log::Write(Log::Level::Info, Fmt("TCVR_MENU shader locations menu=%d,%d screen=%d,%d",
                                           menuPosition, menuTexCoord, screenPosition, screenTexCoord));
        GLuint upscaleVertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(upscaleVertexShader, 1, &UpscaleVertexShaderGlsl, nullptr);
        glCompileShader(upscaleVertexShader);
        CheckShader(upscaleVertexShader);
        GLuint upscaleFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(upscaleFragmentShader, 1, &UpscaleFragmentShaderGlsl, nullptr);
        glCompileShader(upscaleFragmentShader);
        CheckShader(upscaleFragmentShader);
        m_upscaleProgram = glCreateProgram();
        glAttachShader(m_upscaleProgram, upscaleVertexShader);
        glAttachShader(m_upscaleProgram, upscaleFragmentShader);
        glLinkProgram(m_upscaleProgram);
        CheckProgram(m_upscaleProgram);
        glDeleteShader(upscaleVertexShader);
        glDeleteShader(upscaleFragmentShader);
        m_upscaleSourceLocation = glGetUniformLocation(m_upscaleProgram, "SourceTexture");
        m_upscaleSourceSizeLocation = glGetUniformLocation(m_upscaleProgram, "SourceSize");
        m_upscaleTargetSizeLocation = glGetUniformLocation(m_upscaleProgram, "TargetSize");
        m_upscaleFilterLocation = glGetUniformLocation(m_upscaleProgram, "Filter");
        m_upscaleSharpenLocation = glGetUniformLocation(m_upscaleProgram, "Sharpen");
        {
            const GLint upscalePosition = glGetAttribLocation(m_upscaleProgram, "VertexPos");
            glGenVertexArrays(1, &m_upscaleVao);
            glBindVertexArray(m_upscaleVao);
            glBindBuffer(GL_ARRAY_BUFFER, m_screenVertexBuffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_screenIndexBuffer);
            glEnableVertexAttribArray(upscalePosition);
            glVertexAttribPointer(upscalePosition, 3, GL_FLOAT, GL_FALSE, sizeof(ScreenVertex), nullptr);
            glBindVertexArray(0);
        }
        {
            GLuint vs = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &UpscaleVertexShaderGlsl, nullptr);
            glCompileShader(vs);
            CheckShader(vs);
            GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &EdgeFragmentShaderGlsl, nullptr);
            glCompileShader(fs);
            CheckShader(fs);
            m_edgeProgram = glCreateProgram();
            glAttachShader(m_edgeProgram, vs);
            glAttachShader(m_edgeProgram, fs);
            glLinkProgram(m_edgeProgram);
            CheckProgram(m_edgeProgram);
            glDeleteShader(vs);
            glDeleteShader(fs);
            m_edgeSourceLocation = glGetUniformLocation(m_edgeProgram, "SourceTexture");
            m_edgeTargetSizeLocation = glGetUniformLocation(m_edgeProgram, "TargetSize");
            glGenTextures(1, &m_edgeTexture);
            glBindTexture(GL_TEXTURE_2D, m_edgeTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        {
            GLuint vs = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &UpscaleVertexShaderGlsl, nullptr);
            glCompileShader(vs);
            CheckShader(vs);
            GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &DarkFlickerFragmentShaderGlsl, nullptr);
            glCompileShader(fs);
            CheckShader(fs);
            m_darkFlickerProgram = glCreateProgram();
            glAttachShader(m_darkFlickerProgram, vs);
            glAttachShader(m_darkFlickerProgram, fs);
            glLinkProgram(m_darkFlickerProgram);
            CheckProgram(m_darkFlickerProgram);
            glDeleteShader(vs);
            glDeleteShader(fs);
            m_darkFlickerCurrentLocation = glGetUniformLocation(m_darkFlickerProgram, "CurrentTexture");
            m_darkFlickerPreviousLocation = glGetUniformLocation(m_darkFlickerProgram, "PreviousTexture");
            m_darkFlickerHavePreviousLocation = glGetUniformLocation(m_darkFlickerProgram, "HavePrevious");
        }
        glGenFramebuffers(1, &m_upscaleFramebuffer);
        glGenTextures(1, &m_upscaleTexture);
        glBindTexture(GL_TEXTURE_2D, m_upscaleTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

        // Can the driver take MAME's XRGB words as they are? bitmap_rgb32 stores
        // 0x00RRGGBB, which in little-endian memory is B,G,R,x -- exactly
        // GL_BGRA_EXT with GL_UNSIGNED_BYTE. With that, the frame goes from the
        // emulator's buffer to the texture with no per-pixel loop and no copy
        // on this thread at all.
        {
            const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
            m_bgraDirect = extensions && std::strstr(extensions, "GL_EXT_texture_format_BGRA8888") != nullptr;
            Log::Write(Log::Level::Info, Fmt("TCVR_M14 direct BGRA upload %s", m_bgraDirect ? "available" : "NOT available, falling back to CPU conversion"));
        }

        {
            GLuint vs = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &GunVertexShaderGlsl, nullptr);
            glCompileShader(vs);
            CheckShader(vs);
            GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &GunFragmentShaderGlsl, nullptr);
            glCompileShader(fs);
            CheckShader(fs);
            m_gunProgram = glCreateProgram();
            glAttachShader(m_gunProgram, vs);
            glAttachShader(m_gunProgram, fs);
            glLinkProgram(m_gunProgram);
            CheckProgram(m_gunProgram);
            glDeleteShader(vs);
            glDeleteShader(fs);
            m_gunMvpLocation = glGetUniformLocation(m_gunProgram, "ModelViewProjection");
            const GLint gunPos = glGetAttribLocation(m_gunProgram, "VertexPos");
            const GLint gunCol = glGetAttribLocation(m_gunProgram, "VertexColor");

            // Bake every part into one vertex array in gun-local space. The
            // sample's cube keeps its winding (GL_CW front faces), so the gun
            // culls correctly under the same state as the other cubes.
            std::vector<Geometry::Vertex> mesh;
            mesh.reserve(ArraySize(arcadexr::gun::gunParts) * ArraySize(Geometry::c_cubeVertices));
            auto faceShade = [](const XrVector3f& c) {
                if (c.y > 0.5f) return 1.00f;
                if (c.x > 0.5f) return 0.86f;
                if (c.z > 0.5f) return 0.78f;
                if (c.y > 0.1f) return 0.40f;
                if (c.x > 0.1f) return 0.64f;
                return 0.54f;
            };
            for (const auto& part : arcadexr::gun::gunParts) {
                const float c = std::cos(part.pitch), s = std::sin(part.pitch);
                const float shadeScale = 1.0f;
                (void)shadeScale;
                for (const auto& v : Geometry::c_cubeVertices) {
                    float x = v.Position.x * part.size.x;
                    float y = v.Position.y * part.size.y;
                    float z = v.Position.z * part.size.z;
                    // pitch about X: positive tilts the top backwards (+Z)
                    const float y2 = y * c - z * s;
                    const float z2 = y * s + z * c;
                    const float k = faceShade(v.Color);
                    mesh.push_back({{x + part.center.x, y2 + part.center.y, z2 + part.center.z},
                                    {part.color.x * k, part.color.y * k, part.color.z * k}});
                }
            }
            m_gunVertexCount = static_cast<GLsizei>(mesh.size());
            glGenBuffers(1, &m_gunVertexBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, m_gunVertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.size() * sizeof(Geometry::Vertex)), mesh.data(), GL_STATIC_DRAW);
            glGenVertexArrays(1, &m_gunVao);
            glBindVertexArray(m_gunVao);
            glEnableVertexAttribArray(gunPos);
            glEnableVertexAttribArray(gunCol);
            glVertexAttribPointer(gunPos, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex), nullptr);
            glVertexAttribPointer(gunCol, 3, GL_FLOAT, GL_FALSE, sizeof(Geometry::Vertex),
                                  reinterpret_cast<const void*>(sizeof(XrVector3f)));
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            Log::Write(Log::Level::Info, Fmt("TCVR_M14 gun mesh: %d vertices, one draw call per eye", (int)m_gunVertexCount));
        }
        {
            GLuint vs = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vs, 1, &GunLitVertexShaderGlsl, nullptr);
            glCompileShader(vs);
            CheckShader(vs);
            GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fs, 1, &GunLitFragmentShaderGlsl, nullptr);
            glCompileShader(fs);
            CheckShader(fs);
            m_gunLitProgram = glCreateProgram();
            glAttachShader(m_gunLitProgram, vs);
            glAttachShader(m_gunLitProgram, fs);
            glLinkProgram(m_gunLitProgram);
            CheckProgram(m_gunLitProgram);
            glDeleteShader(vs);
            glDeleteShader(fs);
            m_gunLitMvpLocation = glGetUniformLocation(m_gunLitProgram, "ModelViewProjection");
            m_gunLitModelLocation = glGetUniformLocation(m_gunLitProgram, "Model");
            m_gunLitEyeLocation = glGetUniformLocation(m_gunLitProgram, "EyePos");
            const GLint pos = glGetAttribLocation(m_gunLitProgram, "VertexPos");
            const GLint nrm = glGetAttribLocation(m_gunLitProgram, "VertexNormal");
            const GLint col = glGetAttribLocation(m_gunLitProgram, "VertexColor");
            const auto& mesh = arcadexr::gun::GunMesh();
            m_gunLitIndexCount = static_cast<GLsizei>(mesh.indices.size());
            glGenBuffers(1, &m_gunLitVertexBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, m_gunLitVertexBuffer);
            glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(mesh.vertices.size() * sizeof(arcadexr::gun::MeshVertex)), mesh.vertices.data(), GL_STATIC_DRAW);
            glGenBuffers(1, &m_gunLitIndexBuffer);
            glGenVertexArrays(1, &m_gunLitVao);
            glBindVertexArray(m_gunLitVao);
            glBindBuffer(GL_ARRAY_BUFFER, m_gunLitVertexBuffer);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gunLitIndexBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(mesh.indices.size() * sizeof(std::uint16_t)), mesh.indices.data(), GL_STATIC_DRAW);
            glEnableVertexAttribArray(pos);
            glEnableVertexAttribArray(nrm);
            glEnableVertexAttribArray(col);
            glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, sizeof(arcadexr::gun::MeshVertex), nullptr);
            glVertexAttribPointer(nrm, 3, GL_FLOAT, GL_FALSE, sizeof(arcadexr::gun::MeshVertex), reinterpret_cast<const void*>(sizeof(XrVector3f)));
            glVertexAttribPointer(col, 3, GL_FLOAT, GL_FALSE, sizeof(arcadexr::gun::MeshVertex), reinterpret_cast<const void*>(2 * sizeof(XrVector3f)));
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            Log::Write(Log::Level::Info, Fmt("TCVR_M14 lit gun mesh: %d vertices, %d triangles", (int)mesh.vertices.size(), (int)(mesh.indices.size() / 3)));
        }

        glGenTextures(1, &m_screenTexture);
        glBindTexture(GL_TEXTURE_2D, m_screenTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void CheckShader(GLuint shader) {
        GLint r = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &r);
        if (r == GL_FALSE) {
            GLchar msg[4096] = {};
            GLsizei length;
            glGetShaderInfoLog(shader, sizeof(msg), &length, msg);
            THROW(Fmt("Compile shader failed: %s", msg));
        }
    }

    void CheckProgram(GLuint prog) {
        GLint r = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &r);
        if (r == GL_FALSE) {
            GLchar msg[4096] = {};
            GLsizei length;
            glGetProgramInfoLog(prog, sizeof(msg), &length, msg);
            THROW(Fmt("Link program failed: %s", msg));
        }
    }

    // Select the preferred swapchain format from the list of available formats.
    int64_t SelectColorSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override {
        // List of supported color swapchain formats.
        // The order of this list does not effect the priority of selecting formats, the runtime list defines that.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                GL_RGB10_A2,
                GL_RGBA16,
                GL_RGBA16F,
                GL_RGBA32F,

                // The two below should only be used as a fallback, as they are linear color formats without enough bits for color
                // depth, thus leading to banding.
                GL_RGBA8,
                GL_SRGB8_ALPHA8,
            });
    }

    int64_t SelectDepthSwapchainFormat(bool throwIfNotFound, span<const int64_t> imageFormatArray) const override {
        // List of supported depth swapchain formats.
        return SelectSwapchainFormat(  //
            throwIfNotFound, imageFormatArray,
            {
                GL_DEPTH24_STENCIL8,
                GL_DEPTH_COMPONENT24,
                GL_DEPTH_COMPONENT16,
                GL_DEPTH_COMPONENT32F,
            });
    }

    const XrBaseInStructure* GetGraphicsBinding() const override {
        return reinterpret_cast<const XrBaseInStructure*>(&m_graphicsBinding);
    }

    struct OpenGLESFallbackDepthTexture {
       public:
        OpenGLESFallbackDepthTexture() = default;
        ~OpenGLESFallbackDepthTexture() { Reset(); }
        void Reset() {
            if (Allocated()) {
                GL(glDeleteTextures(1, &m_texture));
            }
            m_texture = 0;
            m_xrImage.image = 0;
        }
        bool Allocated() const { return m_texture != 0; }

        void Allocate(GLuint width, GLuint height, uint32_t arraySize) {
            Reset();
            const bool isArray = arraySize > 1;
            GLenum target = isArray ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
            GL(glGenTextures(1, &m_texture));
            GL(glBindTexture(target, m_texture));
            GL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
            GL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
            GL(glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
            GL(glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
            if (isArray) {
                GL(glTexImage3D(target, 0, GL_DEPTH_COMPONENT24, width, height, arraySize, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT,
                                nullptr));
            } else {
                GL(glTexImage2D(target, 0, GL_DEPTH_COMPONENT24, width, height, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr));
            }
            m_xrImage.image = m_texture;
        }
        const XrSwapchainImageOpenGLESKHR& GetTexture() const { return m_xrImage; }

       private:
        uint32_t m_texture{0};
        XrSwapchainImageOpenGLESKHR m_xrImage{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR, NULL, 0};
    };

    class OpenGLESSwapchainImageData : public SwapchainImageDataBase<XrSwapchainImageOpenGLESKHR> {
       public:
        OpenGLESSwapchainImageData(uint32_t capacity, const XrSwapchainCreateInfo& createInfo)
            : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR, capacity, createInfo),
              m_internalDepthTextures(capacity) {}

        OpenGLESSwapchainImageData(uint32_t capacity, const XrSwapchainCreateInfo& createInfo, XrSwapchain depthSwapchain,
                                   const XrSwapchainCreateInfo& depthCreateInfo)
            : SwapchainImageDataBase(XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR, capacity, createInfo, depthSwapchain, depthCreateInfo) {
        }

       protected:
        const XrSwapchainImageOpenGLESKHR& GetFallbackDepthSwapchainImage(uint32_t i) override {
            if (!m_internalDepthTextures[i].Allocated()) {
                m_internalDepthTextures[i].Allocate(this->Width(), this->Height(), this->ArraySize());
            }

            return m_internalDepthTextures[i].GetTexture();
        }

       private:
        std::vector<OpenGLESFallbackDepthTexture> m_internalDepthTextures;
    };

    ISwapchainImageData* AllocateSwapchainImageData(size_t size, const XrSwapchainCreateInfo& swapchainCreateInfo) override {
        auto typedResult = std::make_unique<OpenGLESSwapchainImageData>(uint32_t(size), swapchainCreateInfo);

        // Cast our derived type to the caller-expected type.
        auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

        m_swapchainImageDataMap.Adopt(std::move(typedResult));

        return ret;
    }

    inline ISwapchainImageData* AllocateSwapchainImageDataWithDepthSwapchain(
        size_t size, const XrSwapchainCreateInfo& colorSwapchainCreateInfo, XrSwapchain depthSwapchain,
        const XrSwapchainCreateInfo& depthSwapchainCreateInfo) override {
        auto typedResult = std::make_unique<OpenGLESSwapchainImageData>(uint32_t(size), colorSwapchainCreateInfo, depthSwapchain,
                                                                        depthSwapchainCreateInfo);

        // Cast our derived type to the caller-expected type.
        auto ret = static_cast<ISwapchainImageData*>(typedResult.get());

        m_swapchainImageDataMap.Adopt(std::move(typedResult));

        return ret;
    }

    void RenderView(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
                    int64_t swapchainFormat, const std::vector<Cube>& cubes) override {
        CHECK(layerView.subImage.imageArrayIndex == 0);  // Texture arrays not supported.
        UNUSED_PARM(swapchainFormat);                    // Not used in this function for now.

        OpenGLESSwapchainImageData* swapchainData;
        uint32_t imageIndex;
        std::tie(swapchainData, imageIndex) = m_swapchainImageDataMap.GetDataAndIndexFromBasePointer(swapchainImage);

        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);

        const uint32_t colorTexture = reinterpret_cast<const XrSwapchainImageOpenGLESKHR*>(swapchainImage)->image;
        const uint32_t depthTexture = swapchainData->GetDepthImageForColorIndex(imageIndex).image;

        glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
                   static_cast<GLint>(layerView.subImage.imageRect.offset.y),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));

        glFrontFace(GL_CW);
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);

        // Clear swapchain and depth buffer.
        glClearColor(m_clearColor[0], m_clearColor[1], m_clearColor[2], m_clearColor[3]);
        glClearDepthf(1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // The immersive backend consumes the System 22 camera-space vertices
        // directly. If it is unavailable during cold start, fall back to the
        // proven Arcade Screen path instead of presenting a blank view.
        // Scene mode 2 (MAME records instead of rasterising) is decided HERE, at
        // the call site, and not inside RenderImmersiveModel2.
        //
        // That function has seven early exits -- m2.immersive off, no scene
        // source, GPU init failed, a still-zero focal length, no scene
        // acquired, PrepareFrame failed, presentation changed -- and a guard
        // placed in one of them leaves the other six able to strand MAME in
        // mode 2 with nothing drawing it. The symptom is brutal and was
        // reported twice: the picture freezes while the menu still answers and
        // the sound keeps playing.
        //
        // Deciding on the RESULT, every frame, cannot be stranded: whatever the
        // reason the immersive pass did not produce a frame, the flat path is
        // what runs, and the flat path needs MAME's rasteriser.
        const bool m2Immersive = RenderImmersiveModel2(viewIndex, layerView, colorTexture);
        // m2.skipCpuRaster: 0 (default) keeps MAME rasterising even in immersive.
        //
        // Mode 2 -- MAME records instead of rasterising -- is a real 10 ms/frame
        // saving and it took the emulation from frameskip 8 at 87-93% to
        // frameskip 0 at 100%. But it removes the only other source of a fresh
        // picture: the moment the immersive pass stops producing frames, for any
        // reason, the last image stays glued on screen while the emulation and
        // the sound carry on. That was reported twice from the headset, and it
        // is not acceptable as a default until the freeze itself is understood.
        //
        // FIXED 16/09: the freeze was mode 2 staying on when the immersive pass
        // drew but had NO real 3D view -- intro and menus, where the board's
        // "main view" is off-centre and RenderImmersiveModel2 only lays 2D
        // billboards. MAME then recorded instead of rasterising and nobody
        // produced a picture. So skip CPU raster ONLY when a centred main view
        // exists this frame (a race); otherwise mode 1, MAME keeps drawing. Safe
        // to default on now: the race gets 100% emulation, the rest never gluess.
        const bool allowSkip = arcadexr::config::GetInt("m2.skipCpuRaster", 1) != 0;
        const bool raceView = m2Immersive && m_m2Gpu.HaveMainView();
        if (viewIndex == 0 && m_m2Requested) SetM2SceneMode((raceView && allowSkip) ? 2 : 1);
        if (!m2Immersive && !RenderImmersiveArcadeScene(viewIndex, layerView, colorTexture)) {
            RenderArcadeScreen(viewIndex, layerView);
        }

        // Set shaders and uniform variables.
        glUseProgram(m_program);

        const auto& pose = layerView.pose;
        XrMatrix4x4f proj;
        XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL_ES, layerView.fov, 0.05f, 100.0f);
        XrMatrix4x4f toView;
        XrMatrix4x4f_CreateFromRigidTransform(&toView, &pose);
        XrMatrix4x4f view;
        XrMatrix4x4f_InvertRigidBody(&view, &toView);
        XrMatrix4x4f vp;
        XrMatrix4x4f_Multiply(&vp, &proj, &view);

        // Set cube primitive data.
        glBindVertexArray(m_vao);

        // Render each cube
        for (const Cube& cube : cubes) {
            glUniform3f(m_objectTintLocation,cube.Tint.x,cube.Tint.y,cube.Tint.z);
            // Compute the model-view-projection transform and set it..
            XrMatrix4x4f model;
            XrMatrix4x4f_CreateTranslationRotationScale(&model, &cube.Pose.position, &cube.Pose.orientation, &cube.Scale);
            XrMatrix4x4f mvp;
            XrMatrix4x4f_Multiply(&mvp, &vp, &model);
            glUniformMatrix4fv(m_modelViewProjectionUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));

            // Draw the cube.
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ArraySize(Geometry::c_cubeIndices)), GL_UNSIGNED_SHORT, nullptr);
        }

        glBindVertexArray(0);

        // The gun belongs to gun-game play only. Menus and driving profiles use
        // the Touch controls without drawing Time Crisis' cosmetic weapon.
        // stays as a debug fallback (gun.model=boxes). debug.tcvr.gundemo=1 parks
        // one in front of the head so it can be photographed without a controller.
        {
            auto guns = arcadexr::gun::GetGunPoses();
            if (arcadexr::config::GetInt("gundemo", 0) == 1) {
                XrPosef demo = pose;   // eye pose of this view
                const XrVector3f ahead{0.12f, -0.10f, -0.45f};
                XrPosef_TransformVector3f(&demo.position, &pose, &ahead);
                XrQuaternionf yaw;
                const XrVector3f yAxis{0, 1, 0};
                XrQuaternionf_CreateFromAxisAngle(&yaw, &yAxis, -0.9f);
                XrQuaternionf_Multiply(&demo.orientation, &pose.orientation, &yaw);
                guns.count = 1;
                guns.pose[0] = demo;
            }
            const bool boxes = arcadexr::config::GetString("gun.model", "mesh") == "boxes";
            if (guns.count > 0 && !arcadexr::ui::Menu::Get().IsOpen() && !arcadexr::profiles::IsDriving()) {
                if (boxes && m_gunVertexCount > 0) {
                    glUseProgram(m_gunProgram);
                    glBindVertexArray(m_gunVao);
                } else {
                    glUseProgram(m_gunLitProgram);
                    glBindVertexArray(m_gunLitVao);
                    glUniform3f(m_gunLitEyeLocation, pose.position.x, pose.position.y, pose.position.z);
                }
                for (int g = 0; g < guns.count; ++g) {
                    XrMatrix4x4f model;
                    const XrVector3f unit{1, 1, 1};
                    XrMatrix4x4f_CreateTranslationRotationScale(&model, &guns.pose[g].position, &guns.pose[g].orientation, &unit);
                    XrMatrix4x4f mvp;
                    XrMatrix4x4f_Multiply(&mvp, &vp, &model);
                    if (boxes) {
                        glUniformMatrix4fv(m_gunMvpLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
                        glDrawArrays(GL_TRIANGLES, 0, m_gunVertexCount);
                    } else {
                        glUniformMatrix4fv(m_gunLitMvpLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
                        glUniformMatrix4fv(m_gunLitModelLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&model));
                        glDrawElements(GL_TRIANGLES, m_gunLitIndexCount, GL_UNSIGNED_SHORT, nullptr);
                    }
                }
                glBindVertexArray(0);
            }
        }
        // Arcade rendering may have used a private offscreen FBO. Composite
        // the menu into this eye's current XR swapchain, not that scratch FBO.
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
        glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
                   static_cast<GLint>(layerView.subImage.imageRect.offset.y),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
        RenderMenu(pose, vp, viewIndex);
        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // Keep both menu pages in front of the player's head even if the arcade
    // screen was anchored while the headset was lying on a desk. One pose is
    // latched on view 0 and shared with view 1 for stable binocular depth.
    void RenderMenu(const XrPosef& eyePose, const XrMatrix4x4f& vp, uint32_t viewIndex) {
        auto& menu = arcadexr::ui::Menu::Get();
        if (!menu.IsOpen()) return;
        if (viewIndex == 0 || !m_menuFramePoseValid) {
            m_menuFramePose = eyePose;
            m_menuFramePoseValid = true;
        }
        const XrPosef& anchor = m_menuFramePose;
        const XrVector3f localRight{1, 0, 0}, localUp{0, 1, 0}, localNormal{0, 0, 1};
        XrVector3f right, up, normal;
        XrQuaternionf_RotateVector3f(&right, &anchor.orientation, &localRight);
        XrQuaternionf_RotateVector3f(&up, &anchor.orientation, &localUp);
        XrQuaternionf_RotateVector3f(&normal, &anchor.orientation, &localNormal);
        std::vector<unsigned char> rgba;
        int w = 0, h = 0;
        if (menu.Render(rgba, w, h)) {
            glBindTexture(GL_TEXTURE_2D, m_menuTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
            glBindTexture(GL_TEXTURE_2D, 0);
            m_menuAspect = float(w) / float(h);
        }
        const float width = 0.95f;
        const float height = width / m_menuAspect;
        const XrVector3f center{
            anchor.position.x - normal.x * 1.4f,
            anchor.position.y - normal.y * 1.4f,
            anchor.position.z - normal.z * 1.4f};
        XrMatrix4x4f model{};
        model.m[0] = right.x * width;
        model.m[1] = right.y * width;
        model.m[2] = right.z * width;
        model.m[4] = up.x * height;
        model.m[5] = up.y * height;
        model.m[6] = up.z * height;
        model.m[8] = normal.x;
        model.m[9] = normal.y;
        model.m[10] = normal.z;
        model.m[12] = center.x;
        model.m[13] = center.y;
        model.m[14] = center.z;
        model.m[15] = 1.0f;
        XrMatrix4x4f mvp;
        XrMatrix4x4f_Multiply(&mvp, &vp, &model);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(m_menuProgram);
        glUniformMatrix4fv(m_menuMvpLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m_menuTexture);
        glUniform1i(m_menuTexLocation, 0);
        glBindVertexArray(m_menuVao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
    }

    void RunDarkFlickerPass(GLuint current, GLuint previous, GLuint target, int width, int height, bool havePrevious) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_upscaleFramebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
        const GLenum one[1] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, one);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glUseProgram(m_darkFlickerProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, current);
        glUniform1i(m_darkFlickerCurrentLocation, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, previous);
        glUniform1i(m_darkFlickerPreviousLocation, 1);
        glUniform1i(m_darkFlickerHavePreviousLocation, havePrevious ? 1 : 0);
        glBindVertexArray(m_upscaleVao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
    }

    void ResetSceneAssetsForSelectedGame() {
        const std::string game = arcadexr::profiles::CurrentGame();
        if (game == m_sceneGame) return;
        if (m_scene.AssetsReady()) m_scene.ResetAssets();
        m_sceneGame = game;
        m_sceneSequence = 0;
        m_sceneSignature.clear();
        m_immersivePreparedSequence = 0;
        m_sceneActive = m_immersiveActive = false;
        m_sceneTemporalValid[0] = m_sceneTemporalValid[1] = false;
        m_immersiveTemporalValid[0] = m_immersiveTemporalValid[1] = false;
        m_immersiveHasImage[0] = m_immersiveHasImage[1] = false;
        Log::Write(Log::Level::Info, Fmt("ArcadeXR scene assets selected for profile %s", game.c_str()));
    }

    bool RenderImmersiveArcadeScene(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView,
                                    uint32_t colorTexture) {
        ResetSceneAssetsForSelectedGame();
        if (viewIndex >= 2 || arcadexr::profiles::GetString("render", "cpu") != "gpu" ||
            arcadexr::profiles::GetString("presentation", "screen") != "immersive") {
            return false;
        }

        const int sceneMode = arcadexr::profiles::GetInt("scene.cpuRaster", 0) ? 1 : 2;
        if (m_sceneEnabled != sceneMode) {
            m_sceneEnabled = sceneMode;
            arcadexr::hardware::namco_system22::EnableScene(sceneMode);
            m_sceneActive = false;
            Log::Write(Log::Level::Info, Fmt("TCVR_M15 presentation=immersive scene recording mode %d", sceneMode));
        }
        if (!m_sceneInit) {
            m_sceneInit = true;
            if (!m_scene.Initialize()) {
                Log::Write(Log::Level::Error, Fmt("TCVR_M15 scene renderer failed: %s", m_scene.LastError().c_str()));
            }
            glGenTextures(2, m_sceneTex);
        }
        if (!m_scene.Ready()) return false;

        // Acquire only for view zero. The MAME bridge pins that triple-buffer
        // slot until the next AcquireScene call, so both eyes necessarily use
        // the same emulated state even if MAME publishes between them.
        if (viewIndex == 0 || !m_immersiveFrame) {
            m_immersiveFrame = arcadexr::hardware::namco_system22::AcquireScene();
        }
        const tcvr_scene_frame* frame = m_immersiveFrame;
        if (!frame || frame->width <= 0 || frame->height <= 0) return false;
        if (!m_scene.AssetsReady()) {
            tcvr_scene_assets assets{};
            if (!arcadexr::hardware::namco_system22::SceneAssets(assets) || !m_scene.UploadAssets(assets)) return false;
        }

        if (frame->sequence != m_immersivePreparedSequence) {
            const auto begin = std::chrono::steady_clock::now();
            if (!m_scene.PrepareFrame(*frame)) return false;
            const auto end = std::chrono::steady_clock::now();
            m_immersivePreparedSequence = frame->sequence;
            m_immersivePrepMs += std::chrono::duration<double, std::milli>(end - begin).count();
            ++m_immersivePreparedFrames;
        }

        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) return false;
        const float distance = std::max(0.25f, arcadexr::config::GetFloat("screen.distance", 2.0f));
        const float depthUnits = std::max(100.0f, arcadexr::config::GetFloat("immersive.depth", 5000.0f));
        const float worldScale = distance / depthUnits;
        const arcadexr::gun::Vec3 camera{
            screen.center.x + screen.normal.x * distance,
            screen.center.y + screen.normal.y * distance,
            screen.center.z + screen.normal.z * distance};

        // System 22 camera coordinates are +X right, +Y up, +Z forward.
        // screen.normal points back at the player, hence the negative third
        // basis vector. This anchor recovers the HMD pose used by recentering.
        XrMatrix4x4f arcadeToWorld{};
        arcadeToWorld.m[0] = screen.right.x * worldScale;
        arcadeToWorld.m[1] = screen.right.y * worldScale;
        arcadeToWorld.m[2] = screen.right.z * worldScale;
        arcadeToWorld.m[4] = screen.up.x * worldScale;
        arcadeToWorld.m[5] = screen.up.y * worldScale;
        arcadeToWorld.m[6] = screen.up.z * worldScale;
        arcadeToWorld.m[8] = -screen.normal.x * worldScale;
        arcadeToWorld.m[9] = -screen.normal.y * worldScale;
        arcadeToWorld.m[10] = -screen.normal.z * worldScale;
        arcadeToWorld.m[12] = camera.x;
        arcadeToWorld.m[13] = camera.y;
        arcadeToWorld.m[14] = camera.z;
        arcadeToWorld.m[15] = 1.0f;

        m_anchorCamera = camera;
        m_anchorRight = screen.right;
        m_anchorUp = screen.up;
        m_anchorNormal = screen.normal;
        m_anchorScale = worldScale;
        m_anchorValid = true;
        XrMatrix4x4f projection;
        // The board draws its backdrops up to two million units away; at
        // immersive.depth units per screen distance that is kilometres. A 100 m
        // far plane clipped the whole back of the scene (sky, far buildings).
        const float farMetres = std::max(200.0f, arcadexr::config::GetFloat("immersive.far", 20000.0f));
        // The board draws from z ~ 0: foreground plants, the title logo. At 5000
        // units per screen distance a 5 cm near plane is 125 units and cut them.
        const float nearMetres = std::max(0.001f, std::min(0.05f, arcadexr::config::GetFloat("immersive.near", 0.005f)));
        // immersive.freezePose=1: render from the first pose seen and keep it, so
        // dumps of different settings are comparable whatever the head does.
        const bool freeze = arcadexr::config::GetInt("immersive.freezePose", 0) != 0;
        if (freeze && !m_frozenValid[viewIndex]) { m_frozenPose[viewIndex] = layerView.pose; m_frozenFov[viewIndex] = layerView.fov; m_frozenValid[viewIndex] = true; }
        if (!freeze) m_frozenValid[viewIndex] = false;
        const XrPosef renderPose = freeze ? m_frozenPose[viewIndex] : layerView.pose;
        const XrFovf renderFov = freeze ? m_frozenFov[viewIndex] : layerView.fov;
        XrMatrix4x4f_CreateProjectionFov(&projection, GRAPHICS_OPENGL_ES, renderFov, nearMetres, farMetres);
        XrMatrix4x4f eyeToWorld;
        XrMatrix4x4f_CreateFromRigidTransform(&eyeToWorld, &renderPose);
        XrMatrix4x4f worldToEye;
        XrMatrix4x4f_InvertRigidBody(&worldToEye, &eyeToWorld);
        XrMatrix4x4f viewProjection;
        XrMatrix4x4f_Multiply(&viewProjection, &projection, &worldToEye);
        XrMatrix4x4f mvp;
        XrMatrix4x4f_Multiply(&mvp, &viewProjection, &arcadeToWorld);

        // The game's 2D sprites/HUD belong to its original projection plane,
        // not to the player's face. One model matrix per eye gives them the
        // comfortable disparity of the virtual arcade screen.
        XrMatrix4x4f hudToWorld{};
        hudToWorld.m[0] = screen.right.x * screen.width;
        hudToWorld.m[1] = screen.right.y * screen.width;
        hudToWorld.m[2] = screen.right.z * screen.width;
        hudToWorld.m[4] = screen.up.x * screen.height;
        hudToWorld.m[5] = screen.up.y * screen.height;
        hudToWorld.m[6] = screen.up.z * screen.height;
        hudToWorld.m[8] = screen.normal.x;
        hudToWorld.m[9] = screen.normal.y;
        hudToWorld.m[10] = screen.normal.z;
        hudToWorld.m[12] = screen.center.x;
        hudToWorld.m[13] = screen.center.y;
        hudToWorld.m[14] = screen.center.z;
        hudToWorld.m[15] = 1.0f;
        XrMatrix4x4f hudMvp;
        XrMatrix4x4f_Multiply(&hudMvp, &viewProjection, &hudToWorld);

        m_scene.SetImmersiveDepth(arcadexr::profiles::GetInt("immersive.depthTest", 1) != 0,
                                  arcadexr::profiles::GetFloat("immersive.depthBias", 4e-8f));
        m_scene.SetFogVoid(arcadexr::config::GetInt("immersive.fogVoid", 1) != 0);
        {
            // immersive.void = game | fog | r,g,b ; immersive.hudBand = top,bottom (fractions)
            const std::string v = arcadexr::config::GetString("immersive.void", "game");
            unsigned r = 0, g = 0, b = 0;
            if (v == "game") m_scene.SetVoid(0, 0, 0, 0);
            else if (std::sscanf(v.c_str(), "%u,%u,%u", &r, &g, &b) == 3) m_scene.SetVoid(2, r & 255, g & 255, b & 255);
            else m_scene.SetVoid(1, 0, 0, 0);
            float top = 0.16f, bottom = 0.84f;
            const std::string band = arcadexr::config::GetString("immersive.hudBand", "0.16,0.84");
            if (std::sscanf(band.c_str(), "%f,%f", &top, &bottom) == 2) m_scene.SetHudBand(top, bottom);
        }
        // Render once per emulated frame, not once per XR frame: at 120 Hz x 2
        // eyes x 1680x1760 x 3 passes the full-rate rendering starved the
        // emulator (MAME 53 -> 33 fps, scene rate down to 10/s). Between two
        // emulated frames the eye keeps the image and the pose it was rendered
        // with; the compositor reprojects the rotation from that pose.
        float renderScale = arcadexr::profiles::GetFloat("immersive.scale", 1.0f);
        const bool fxaa = arcadexr::profiles::GetInt("immersive.fxaa", 1) != 0 && m_edgeProgram != 0;
        {
            const int texAa = arcadexr::profiles::GetInt("immersive.texAA", 1);
            m_scene.SetTextureSamples(texAa <= 0 ? 1 : (texAa == 1 ? 4 : 16));
            const int msaa = arcadexr::config::GetInt("immersive.msaa", 0);
            m_scene.SetMsaa(msaa >= 4 ? 4 : (msaa >= 2 ? 2 : 1));
            m_scene.SetSpriteMinDepth(arcadexr::config::GetFloat("immersive.spriteMinDepth", 50.0f));
        }
        renderScale = std::max(0.3f, std::min(2.0f, renderScale));   // > 1 = supersampling, box-reduced by the linear blit at 2.0
        const int eyeW = layerView.subImage.imageRect.extent.width, eyeH = layerView.subImage.imageRect.extent.height;
        const int rw = std::max(64, int(eyeW * renderScale)), rh = std::max(64, int(eyeH * renderScale));
        if (m_immersiveTexW != rw || m_immersiveTexH != rh) {
            for (int e = 0; e < 2; ++e) {
                if (!m_immersiveTex[e]) glGenTextures(1, &m_immersiveTex[e]);
                glBindTexture(GL_TEXTURE_2D, m_immersiveTex[e]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                m_immersiveHasImage[e] = false;
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            if (!m_immersiveBlitFbo) glGenFramebuffers(1, &m_immersiveBlitFbo);
            m_immersiveTexW = rw;
            m_immersiveTexH = rh;
            m_immersiveTemporalValid[0] = m_immersiveTemporalValid[1] = false;
            Log::Write(Log::Level::Info, Fmt("TCVR_M15 immersive render scale %.2f -> %dx%d per eye", renderScale, rw, rh));
        }
        const bool stabilizeDark = arcadexr::profiles::GetInt("temporal.darkFlicker", 0) != 0 && m_darkFlickerProgram != 0;
        if (stabilizeDark && !m_immersiveTemporalRaw[0][0]) glGenTextures(4, &m_immersiveTemporalRaw[0][0]);
        if (stabilizeDark && (rw != m_immersiveTemporalW || rh != m_immersiveTemporalH)) {
            for (int e = 0; e < 2; ++e) for (int i = 0; i < 2; ++i) {
                glBindTexture(GL_TEXTURE_2D, m_immersiveTemporalRaw[e][i]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            m_immersiveTemporalW = rw; m_immersiveTemporalH = rh;
            m_immersiveTemporalValid[0] = m_immersiveTemporalValid[1] = false;
        }
        const bool everyFrame = arcadexr::config::GetInt("immersive.everyFrame", 0) != 0;
        bool rendered = false;
        const auto begin = std::chrono::steady_clock::now();
        if (everyFrame || frame->sequence != m_immersiveRenderedSeq[viewIndex] || !m_immersiveHasImage[viewIndex]) {
            if (stabilizeDark) {
                const int current = m_immersiveTemporalIndex[viewIndex] ^ 1;
                rendered = m_scene.RenderEye(*frame, 0.0f, 0.0f, m_immersiveTemporalRaw[viewIndex][current], rw, rh,
                                             reinterpret_cast<const float*>(&mvp), reinterpret_cast<const float*>(&hudMvp));
                if (rendered) {
                    RunDarkFlickerPass(m_immersiveTemporalRaw[viewIndex][current],
                                       m_immersiveTemporalRaw[viewIndex][m_immersiveTemporalIndex[viewIndex]],
                                       m_immersiveTex[viewIndex], rw, rh, m_immersiveTemporalValid[viewIndex]);
                    m_immersiveTemporalIndex[viewIndex] = current;
                    m_immersiveTemporalValid[viewIndex] = true;
                }
            } else {
                rendered = m_scene.RenderEye(*frame, 0.0f, 0.0f, m_immersiveTex[viewIndex], rw, rh,
                                             reinterpret_cast<const float*>(&mvp), reinterpret_cast<const float*>(&hudMvp));
                m_immersiveTemporalValid[viewIndex] = false;
            }
            if (rendered && fxaa) {
                // The same FXAA pass as the screen window's edge filter, on the eye image.
                if (m_immersiveAaW != rw || m_immersiveAaH != rh) {
                    for (int e = 0; e < 2; ++e) {
                        if (!m_immersiveAa[e]) glGenTextures(1, &m_immersiveAa[e]);
                        glBindTexture(GL_TEXTURE_2D, m_immersiveAa[e]);
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    }
                    glBindTexture(GL_TEXTURE_2D, 0);
                    m_immersiveAaW = rw;
                    m_immersiveAaH = rh;
                }
                glBindFramebuffer(GL_FRAMEBUFFER, m_upscaleFramebuffer);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_immersiveAa[viewIndex], 0);
                glViewport(0, 0, rw, rh);
                glDisable(GL_DEPTH_TEST);
                glDisable(GL_CULL_FACE);
                glDisable(GL_BLEND);
                glUseProgram(m_edgeProgram);
                glUniform2f(m_edgeTargetSizeLocation, float(rw), float(rh));
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, m_immersiveTex[viewIndex]);
                glUniform1i(m_edgeSourceLocation, 0);
                glBindVertexArray(m_upscaleVao);
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
                glBindVertexArray(0);
                glBindTexture(GL_TEXTURE_2D, 0);
                glUseProgram(0);
            }
            if (rendered) {
                m_immersiveRenderedSeq[viewIndex] = frame->sequence;
                m_immersivePose[viewIndex] = renderPose;
                m_immersiveFov[viewIndex] = renderFov;
                m_immersiveHasImage[viewIndex] = true;
                ++m_immersiveRenders;
            }
        } else {
            // Same image as last time: tell the compositor which pose it was made
            // for, so its reprojection is right. The struct we were handed is the
            // one the program submits after this call.
            auto& submitted = const_cast<XrCompositionLayerProjectionView&>(layerView);
            submitted.pose = m_immersivePose[viewIndex];
            submitted.fov = m_immersiveFov[viewIndex];
            rendered = true;
        }
        {
            // debug.tcvr.dump=<tag>: the eye texture before the compositor, one PPM per eye.
            const std::string tag = arcadexr::config::GetString("dump", "");
            if (rendered && !tag.empty() && tag != m_immersiveDumpTag[viewIndex]) {
                m_immersiveDumpTag[viewIndex] = tag;
                const GLuint src = (fxaa && m_immersiveAa[viewIndex]) ? m_immersiveAa[viewIndex] : m_immersiveTex[viewIndex];
                std::vector<unsigned char> rgba;
                if (m_scene.ReadBack(src, rw, rh, rgba)) {
                    const std::string path = arcadexr::config::ExternalDirectory() + "/dump-" + tag + (viewIndex == 0 ? "-imm-L.ppm" : "-imm-R.ppm");
                    if (FILE* f = std::fopen(path.c_str(), "wb")) {
                        std::fprintf(f, "P6\n%d %d\n255\n", rw, rh);
                        std::vector<unsigned char> row(size_t(rw) * 3);
                        for (int y = rh - 1; y >= 0; --y) {
                            const unsigned char* s = rgba.data() + size_t(y) * size_t(rw) * 4;
                            for (int x = 0; x < rw; ++x) { row[x*3] = s[x*4]; row[x*3+1] = s[x*4+1]; row[x*3+2] = s[x*4+2]; }
                            std::fwrite(row.data(), 1, row.size(), f);
                        }
                        std::fclose(f);
                        Log::Write(Log::Level::Info, Fmt("TCVR_M15 dumped %s (%dx%d)", path.c_str(), rw, rh));
                    }
                }
            }
        }
        if (rendered) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_immersiveBlitFbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                                   (fxaa && m_immersiveAa[viewIndex]) ? m_immersiveAa[viewIndex] : m_immersiveTex[viewIndex], 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_swapchainFramebuffer);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
            glBlitFramebuffer(0, 0, rw, rh, layerView.subImage.imageRect.offset.x, layerView.subImage.imageRect.offset.y,
                              layerView.subImage.imageRect.offset.x + eyeW, layerView.subImage.imageRect.offset.y + eyeH,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR);
        }
        const auto end = std::chrono::steady_clock::now();
        m_immersiveDrawMs += std::chrono::duration<double, std::milli>(end - begin).count();
        ++m_immersiveViews;
        m_sceneActive = rendered;
        m_immersiveActive = rendered;

        // Restore the framebuffer state expected by the gun/cube renderer.
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);
        glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
                   static_cast<GLint>(layerView.subImage.imageRect.offset.y),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);

        if (rendered && (!m_loggedImmersiveRoute[viewIndex] || m_immersiveViews % 1200 == 0)) {
            m_loggedImmersiveRoute[viewIndex] = true;
            Log::Write(Log::Level::Info, Fmt(
                "TCVR_M15 immersive view=%u seq=%llu renders=%u target=%ux%u depth=%.1f scale=%.8f prep_avg=%.2fms submit_avg=%.2fms",
                viewIndex, static_cast<unsigned long long>(frame->sequence), m_immersiveRenders,
                layerView.subImage.imageRect.extent.width, layerView.subImage.imageRect.extent.height,
                depthUnits, worldScale,
                m_immersivePreparedFrames ? m_immersivePrepMs / m_immersivePreparedFrames : 0.0,
                m_immersiveViews ? m_immersiveDrawMs / m_immersiveViews : 0.0));
        }
        return rendered;
    }

    // The aim, immersive: the gun ray walks the rendered scene (board camera
    // space) and the game receives the screen position of what it meets.
    bool SceneAim(const XrVector3f& origin, const XrVector3f& direction, float& nx, float& ny, XrVector3f& hitWorld) {
        if (!m_immersiveActive || !m_anchorValid || !m_immersiveFrame) return false;
        const auto dot = [](const XrVector3f& a, const arcadexr::gun::Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
        const XrVector3f rel{origin.x - m_anchorCamera.x, origin.y - m_anchorCamera.y, origin.z - m_anchorCamera.z};
        const float o[3] = {dot(rel, m_anchorRight) / m_anchorScale, dot(rel, m_anchorUp) / m_anchorScale, -dot(rel, m_anchorNormal) / m_anchorScale};
        const float d[3] = {dot(direction, m_anchorRight), dot(direction, m_anchorUp), -dot(direction, m_anchorNormal)};
        float sx = 0, sy = 0, hit[3] = {0, 0, 0};
        if (!m_scene.RayCast(o, d, m_immersiveFrame->width, m_immersiveFrame->height, sx, sy, hit)) return false;
        nx = sx / float(m_immersiveFrame->width);
        ny = sy / float(m_immersiveFrame->height);
        hitWorld = {m_anchorCamera.x + (m_anchorRight.x * hit[0] + m_anchorUp.x * hit[1] - m_anchorNormal.x * hit[2]) * m_anchorScale,
                    m_anchorCamera.y + (m_anchorRight.y * hit[0] + m_anchorUp.y * hit[1] - m_anchorNormal.y * hit[2]) * m_anchorScale,
                    m_anchorCamera.z + (m_anchorRight.z * hit[0] + m_anchorUp.z * hit[1] - m_anchorNormal.z * hit[2]) * m_anchorScale};
        return true;
    }
    static OpenGLESGraphicsPlugin* s_self;
    static bool SceneAimTrampoline(const XrVector3f& origin, const XrVector3f& direction, float& nx, float& ny, XrVector3f& hitWorld) {
        return s_self && s_self->SceneAim(origin, direction, nx, ny, hitWorld);
    }

    void RenderArcadeScreen(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView) {
        m_immersiveActive = false;
        ResetSceneAssetsForSelectedGame();
        CHECK(viewIndex < 2);
        // render=gpu: the true-3D path. Recording in the emulator is switched
        // on only while someone reads the scene.
        const bool gpuRender = arcadexr::profiles::GetString("render", "cpu") == "gpu";
        // Measured 21:50 on the same scene, toggling live: with the CPU rasteriser
        // kept as a fallback the emulator thread blocked in the saturated
        // rasteriser queue (dispatch 24-37 ms, MAME 55 fps, 25-75 audio underruns
        // a second); without it, dispatch 0.5 ms, MAME 59.9, no underruns. The
        // GPU draws the frame: the CPU rasteriser is off unless asked for.
        const int mode = gpuRender ? (arcadexr::profiles::GetInt("scene.cpuRaster", 0) ? 1 : 2) : 0;
        if (mode != m_sceneEnabled) {
            m_sceneEnabled = mode;
            arcadexr::hardware::namco_system22::EnableScene(mode);
            m_sceneActive = false;
            Log::Write(Log::Level::Info, Fmt("TCVR_M12 render=%s scene recording mode %d", gpuRender ? "gpu" : "cpu", mode));
        }
        if (gpuRender) RenderSceneIfNew(viewIndex, layerView); else m_sceneActive = false;
        // The XR presentation clock runs far faster than the arcade clock, and
        // is deliberately not tied to it: between two emulated frames the screen
        // simply keeps the texture it already has. Asking what the latest frame
        // IS costs nothing; copying it costs a full framebuffer memcpy, under a
        // mutex shared with the emulator thread, once per eye. Paying that on
        // every XR frame starved the emulator.
        arcadexr::video::FrameInfo info;
        if (!arcadexr::video::PeekLatestFrameInfo(info)) {
            return;
        }
        const bool haveNewFrame =
            info.sequence != m_lastFrameSequence || info.width != m_frameWidth || info.height != m_frameHeight;
        // With the emulator frozen no new frame ever arrives, yet a filter or a
        // dump request must still take effect: re-run the pass when the
        // settings signature changes. Checked once a frame, not once an eye.
        if (!haveNewFrame && m_frameWidth > 0 && (++m_settingsPoll & 1) == 0) {
            const std::string signature = arcadexr::profiles::GetString("filter", "edge") + "|" +
                                          arcadexr::profiles::GetString("sharpen", "0") + "|" +
                                          arcadexr::profiles::GetString("upscale", "1") + "|" +
                                          arcadexr::config::GetString("dump", "");
            if (signature != m_settingsSignature) {
                m_settingsSignature = signature;
                RunUpscalePass(layerView);
            }
        }
        if (haveNewFrame && m_bgraDirect) {
            // Zero-copy path: the emulator's own buffer goes straight to the
            // driver. No memcpy, no conversion loop, no reallocation.
            arcadexr::video::FrameInfo got;
            const std::uint32_t* pixels = arcadexr::video::AcquireLatestFrame(got);
            if (!pixels) return;
            glBindTexture(GL_TEXTURE_2D, m_screenTexture);
            if (got.width != m_frameWidth || got.height != m_frameHeight) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA_EXT, got.width, got.height, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, nullptr);
                m_frameWidth = got.width;
                m_frameHeight = got.height;
            }
            glPixelStorei(GL_UNPACK_ROW_LENGTH, got.stride);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, got.width, got.height, GL_BGRA_EXT, GL_UNSIGNED_BYTE, pixels);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_lastFrameSequence = got.sequence;
            MaybeRenderModel2Gpu();
            RunUpscalePass(layerView);
            if (!m_loggedScreenUpload) {
                Log::Write(Log::Level::Info, Fmt("TCVR_M4 XR screen texture upload seq=%llu size=%dx%d (direct BGRA)",
                                                 static_cast<unsigned long long>(got.sequence), got.width, got.height));
                m_loggedScreenUpload = true;
            }
        } else if (haveNewFrame && !arcadexr::video::CopyLatestFrame(m_framePixels, info)) {
            return;
        } else if (haveNewFrame) {
            m_frameWidth = info.width;
            m_frameHeight = info.height;
            m_lastFrameSequence = info.sequence;
            m_frameRgba.resize(static_cast<size_t>(info.width) * static_cast<size_t>(info.height) * 4);
            for (int y = 0; y < info.height; ++y) {
                for (int x = 0; x < info.width; ++x) {
                    const std::uint32_t pixel = m_framePixels[static_cast<size_t>(y) * info.stride + x];
                    const size_t dst = (static_cast<size_t>(y) * info.width + x) * 4;
                    m_frameRgba[dst + 0] = static_cast<std::uint8_t>((pixel >> 16) & 0xff);
                    m_frameRgba[dst + 1] = static_cast<std::uint8_t>((pixel >> 8) & 0xff);
                    m_frameRgba[dst + 2] = static_cast<std::uint8_t>(pixel & 0xff);
                    m_frameRgba[dst + 3] = 0xff;
                }
            }
            glBindTexture(GL_TEXTURE_2D, m_screenTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, info.width, info.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         m_frameRgba.data());
            glBindTexture(GL_TEXTURE_2D, 0);
            RunUpscalePass(layerView);
            if (!m_loggedScreenUpload) {
                Log::Write(Log::Level::Info, Fmt("TCVR_M4 XR screen texture upload seq=%llu size=%dx%d",
                                                 static_cast<unsigned long long>(info.sequence), info.width, info.height));
                m_loggedScreenUpload = true;
            }
        }

        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) {
            return;
        }
        arcadexr::video::UpdateVirtualScreenAspect(
            static_cast<float>(m_frameWidth) / static_cast<float>(m_frameHeight));
        arcadexr::video::GetVirtualScreen(screen);

        // The model is built from the same application-space plane that the
        // lightgun uses. It is intentionally not recomputed per eye.
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
        XrMatrix4x4f proj;
        XrMatrix4x4f_CreateProjectionFov(&proj, GRAPHICS_OPENGL_ES, layerView.fov, 0.05f, 100.0f);
        XrMatrix4x4f toView;
        XrMatrix4x4f_CreateFromRigidTransform(&toView, &layerView.pose);
        XrMatrix4x4f view;
        XrMatrix4x4f_InvertRigidBody(&view, &toView);
        XrMatrix4x4f vp;
        XrMatrix4x4f_Multiply(&vp, &proj, &view);
        XrMatrix4x4f mvp;
        XrMatrix4x4f_Multiply(&mvp, &vp, &model);

        // One-sided cabinet screen. The model matrix maps the quad's local +Z
        // onto screen.normal, which points at the player, so the player-facing
        // side is wound counter-clockwise in window space -- the opposite of the
        // GL_CW convention the cube geometry uses. Flipping the winding for this
        // draw keeps the front visible and culls the back, so walking behind the
        // screen shows nothing instead of a mirrored copy of the game.
        glFrontFace(GL_CCW);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glUseProgram(m_screenProgram);
        glUniformMatrix4fv(m_screenMvpUniformLocation, 1, GL_FALSE, reinterpret_cast<const GLfloat*>(&mvp));
        glActiveTexture(GL_TEXTURE0);
        // A flat profile deliberately has no per-eye scene disparity. Reuse
        // the left scene texture for both projection views instead of paying
        // for a second identical System 22 raster pass.
        const bool flat = arcadexr::profiles::GetString("presentation", "screen") == "flat";
        const uint32_t sceneEye = flat ? 0u : viewIndex;
        glBindTexture(GL_TEXTURE_2D,
                      m_sceneActive ? m_sceneTex[sceneEye]
                      : m_m2SourceTexture != 0
                            ? ((m_m2Stereo && m_m2TexR != 0 && sceneEye == 1) ? m_m2TexR : m_m2SourceTexture)
                            : (m_upscaleFactor > 1 ? m_finalTexture : m_screenTexture));
        if (m_sceneActive && !m_loggedEyeRoute[viewIndex]) {
            m_loggedEyeRoute[viewIndex] = true;
            Log::Write(Log::Level::Info, Fmt("TCVR_M14 projection view %u samples scene eye %u texture=%u",
                viewIndex, sceneEye, m_sceneTex[sceneEye]));
        }
        glUniform1i(m_screenTextureUniformLocation, 0);
        const auto aim = arcadexr::gun::GetAimState();
        glUniform2f(m_screenAimPointUniformLocation, aim.normalized_x, aim.normalized_y);
        glUniform1i(m_screenAimVisibleUniformLocation, aim.show_crosshair ? 1 : 0);
        glUniform1i(m_screenCalibratingLocation, aim.calibrating ? 1 : 0);
        glBindVertexArray(m_screenVao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
        glFrontFace(GL_CW);
    }

    // render=gpu: rasterise the recorded scene on the GPU, once per emulated
    // frame, for both eyes at once so they always show the same frame. The
    // CPU frame keeps being uploaded underneath as the fallback.
    bool RenderSceneIfNew(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView) {
        // RenderView is called once per projection view. Latch and rasterise a
        // new MAME scene only for view 0, then make view 1 sample that exact
        // pair. This guarantees that both eyes see one immutable emulated frame.
        if (viewIndex != 0) return m_sceneActive;
        if (!m_sceneInit) {
            m_sceneInit = true;
            if (!m_scene.Initialize()) {
                Log::Write(Log::Level::Error, Fmt("TCVR_M12 scene renderer failed: %s", m_scene.LastError().c_str()));
            }
            glGenTextures(2, m_sceneTex);
        }
        if (!m_scene.Ready()) return false;
        const tcvr_scene_frame* frame = arcadexr::hardware::namco_system22::AcquireScene();
        if (!frame || frame->width <= 0 || frame->height <= 0) return false;
        if (!m_scene.AssetsReady()) {
            tcvr_scene_assets assets{};
            if (!arcadexr::hardware::namco_system22::SceneAssets(assets)) return false;
            if (!m_scene.UploadAssets(assets)) {
                Log::Write(Log::Level::Error, Fmt("TCVR_M12 assets failed: %s", m_scene.LastError().c_str()));
                return false;
            }
        }
        int scale = arcadexr::profiles::GetInt("scene.scale", 2);
        scale = std::max(1, std::min(4, scale));
        const bool forceMono = arcadexr::profiles::GetString("presentation", "screen") == "flat";
        const float strength = forceMono ? 0.0f : arcadexr::profiles::GetFloat("stereo.strength", 0.0f);
        const float convergence = arcadexr::profiles::GetFloat("stereo.convergence", 0.0f);
        const std::string tag = arcadexr::config::GetString("dump", "");
        const std::string signature = Fmt("%d|%.4f|%.2f|%d|%s", scale, strength, convergence, forceMono ? 1 : 0, tag.c_str());
        if (frame->sequence == m_sceneSequence && signature == m_sceneSignature) {
            m_sceneActive = true;
            return true;
        }
        m_sceneSequence = frame->sequence;
        m_sceneSignature = signature;
        const int width = frame->width * scale, height = frame->height * scale;
        if (width != m_sceneW || height != m_sceneH) {
            for (int e = 0; e < 2; ++e) {
                glBindTexture(GL_TEXTURE_2D, m_sceneTex[e]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            m_sceneW = width;
            m_sceneH = height;
            m_sceneTemporalValid[0] = m_sceneTemporalValid[1] = false;
        }
        const bool stabilizeDark = arcadexr::profiles::GetInt("temporal.darkFlicker", 0) != 0 && m_darkFlickerProgram != 0;
        if (stabilizeDark && !m_sceneTemporalRaw[0][0]) glGenTextures(4, &m_sceneTemporalRaw[0][0]);
        if (stabilizeDark && (width != m_sceneTemporalW || height != m_sceneTemporalH)) {
            for (int e = 0; e < 2; ++e) for (int i = 0; i < 2; ++i) {
                glBindTexture(GL_TEXTURE_2D, m_sceneTemporalRaw[e][i]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            m_sceneTemporalW = width; m_sceneTemporalH = height;
            m_sceneTemporalValid[0] = m_sceneTemporalValid[1] = false;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool prepared = m_scene.PrepareFrame(*frame);
        const auto t1 = std::chrono::steady_clock::now();
        if (prepared) {
            const int eyeCount = forceMono ? 1 : 2;
            for (int e = 0; e < eyeCount; ++e) {
                const float offset = (e == 0 ? -0.5f : 0.5f) * strength;
                if (stabilizeDark) {
                    const int current = m_sceneTemporalIndex[e] ^ 1;
                    m_scene.RenderEye(*frame, offset, convergence, m_sceneTemporalRaw[e][current], width, height);
                    RunDarkFlickerPass(m_sceneTemporalRaw[e][current], m_sceneTemporalRaw[e][m_sceneTemporalIndex[e]],
                                       m_sceneTex[e], width, height, m_sceneTemporalValid[e]);
                    m_sceneTemporalIndex[e] = current;
                    m_sceneTemporalValid[e] = true;
                } else {
                    m_scene.RenderEye(*frame, offset, convergence, m_sceneTex[e], width, height);
                    m_sceneTemporalValid[e] = false;
                }
            }
        }
        const auto t2 = std::chrono::steady_clock::now();
        // The first published scene can legitimately contain no primitives
        // while the System 22 video devices finish starting. Do not consume a
        // one-shot proof tag on that empty bootstrap frame: both eyes would be
        // background-only and could falsely "prove" zero or non-zero stereo.
        if (prepared && m_scene.LastStereoPrimCount() > 0 && !tag.empty() && tag != m_sceneDumpTag) {
            m_sceneDumpTag = tag;
            std::vector<unsigned char> rgba;
            const int eyeCount = forceMono ? 1 : 2;
            for (int e = 0; e < eyeCount; ++e) {
                if (!m_scene.ReadBack(m_sceneTex[e], width, height, rgba)) continue;
                const std::string path = arcadexr::config::ExternalDirectory() + "/dump-" + tag + (e == 0 ? "-L.ppm" : "-R.ppm");
                if (FILE* f = std::fopen(path.c_str(), "wb")) {
                    std::fprintf(f, "P6\n%d %d\n255\n", width, height);
                    std::vector<unsigned char> row(size_t(width) * 3);
                    // The scene shader deliberately writes arcade row zero to
                    // GL texture row zero so the existing virtual-screen UVs
                    // show it upright. glReadPixels therefore already returns
                    // rows in the file order we want; reversing them made only
                    // the diagnostic dump appear upside down.
                    for (int y = 0; y < height; ++y) {
                        const unsigned char* src = rgba.data() + size_t(y) * size_t(width) * 4;
                        for (int x = 0; x < width; ++x) { row[x*3] = src[x*4]; row[x*3+1] = src[x*4+1]; row[x*3+2] = src[x*4+2]; }
                        std::fwrite(row.data(), 1, row.size(), f);
                    }
                    std::fclose(f);
                    Log::Write(Log::Level::Info, Fmt("TCVR_M12 dumped %s (%dx%d strength=%.3f conv=%.1f stereo_prims=%u zoom=%.6g..%.6g z=%.6g..%.6g)",
                        path.c_str(), width, height, strength, convergence, m_scene.LastStereoPrimCount(),
                        m_scene.LastStereoZoomMin(), m_scene.LastStereoZoomMax(), m_scene.LastStereoDepthMin(),
                        m_scene.LastStereoDepthMax()));
                }
            }
        }
        // Hand the eye's framebuffer and viewport back exactly as they were.
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);
        glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
                   static_cast<GLint>(layerView.subImage.imageRect.offset.y),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glDisable(GL_BLEND);
        glActiveTexture(GL_TEXTURE0);
        const double prepMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double drawMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        m_scenePrepMs += prepMs; m_sceneDrawMs += drawMs; ++m_sceneFrames;
        if (m_sceneFrames == 1 || m_sceneFrames % 600 == 0) {
            Log::Write(Log::Level::Info, Fmt("TCVR_M12 scene seq=%llu %ux%u prims=%u stereo_prims=%u tris=%u prepare=%.2fms submit=%.2fms (avg %.2f/%.2f over %u) scale=%d strength=%.3f",
                static_cast<unsigned long long>(frame->sequence), width, height, m_scene.LastPrimCount(), m_scene.LastStereoPrimCount(), m_scene.LastIndexCount() / 3,
                prepMs, drawMs, m_scenePrepMs / m_sceneFrames, m_sceneDrawMs / m_sceneFrames, m_sceneFrames, scale, strength));
        }
        m_sceneActive = prepared;
        return prepared;
    }

    // Magnifies the emulated frame into an offscreen texture. Called only when
    // a new emulated frame has just been uploaded, so its cost is tied to the
    // arcade clock -- sixty times a second -- and never to the presentation
    // clock, which on this headset runs at 207 Hz.
    // Rasterise the recorded Model 2 scene on the GPU, into a texture the
    // upscale pass then reads instead of the emulator's framebuffer. Opt-in,
    // and MAME's own rasteriser keeps running as the oracle: the point of this
    // first step is to compare, not yet to look better.

    // Sega Model 2 games in immersive presentation: the recorded scene through
    // the eye's projection, 2D layers on the arcade plane. Same anchor, pose
    // cache and blit as the System 22 path.
    // Recording mode for the Model 2 scene store, in one place so the flat and
    // immersive paths cannot fight over it.
    //
    // Mode 2 tells MAME to record the scene INSTEAD of rasterising it: the
    // immersive pass draws that scene itself and never reads destmap, and the
    // CPU raster costs 9.5-11.0 ms per frame (measured as the render_polygons
    // `dispatch` figure, wait and join at zero), which is nearly all of
    // screen_update and what dropped the emulation to frameskip 8 at 91-92%.
    //
    // It is only ever requested once an immersive frame has actually been
    // rendered, and dropped back to 1 as soon as it has not: every early exit
    // of RenderImmersiveModel2 falls back on the flat window, which does
    // composite on the emulator's own frame.
    void SetM2SceneMode(int mode) {
        if (mode == m_m2SceneMode) return;
        m_m2SceneMode = mode;
        arcadexr::hardware::sega_model2::EnableScene(mode);
        __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU",
                            "scene recording mode %d (%s)", mode,
                            mode >= 2 ? "GPU draws it, MAME does not rasterise"
                                      : "alongside MAME's CPU raster");
    }

    bool RenderImmersiveModel2(uint32_t viewIndex, const XrCompositionLayerProjectionView& layerView, uint32_t colorTexture) {
        // Work in progress: scale calibration, sky dome and the stereo window are
        // missing. Off unless m2.immersive=1, so IMMERSIF falls back to the 4x window.
        const std::string presentation = arcadexr::profiles::GetString("presentation", "screen");
        const bool immLog = (++m_m2ImmTick % 120u) == 1u;
        auto immTrace = [&](const char* why) { if (immLog) __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "immersive exit=%s view=%u failed=%d requested=%d present=%s", why, viewIndex, m_m2GpuFailed?1:0, m_m2Requested?1:0, presentation.c_str()); };
        if (viewIndex >= 2 || presentation != "immersive") {
            if (!m_m2ImmLoggedGate) { m_m2ImmLoggedGate = true; __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "immersive gate: presentation=%s", presentation.c_str()); }
            return false;
        }
        if (arcadexr::profiles::GetInt("m2.immersive", 1) == 0) { if (!m_m2ImmLoggedGate) { m_m2ImmLoggedGate = true; __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "immersive gate: m2.immersive=0"); } return false; }
        if (!arcadexr::hardware::sega_model2::HaveSceneSource()) { if (!m_m2ImmLoggedGate) { m_m2ImmLoggedGate = true; __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "immersive gate: no Model 2 scene source"); } return false; }
        if (!m_m2Requested) {
            m_m2Requested = true;
            SetM2SceneMode(1);
            __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "immersive: recording requested");
            return false;
        }
        if (!m_m2Gpu.Ready() && !m_m2GpuFailed) {
            if (!m_m2Gpu.Initialize()) { m_m2GpuFailed = true; Log::Write(Log::Level::Error, Fmt("TCVR_M2GPU disabled: %s", m_m2Gpu.LastError().c_str())); return false; }
        }
        if (m_m2GpuFailed) { immTrace("gpu_failed"); return false; }
        if (viewIndex == 0 || !m_m2ImmFrame) m_m2ImmFrame = arcadexr::hardware::sega_model2::AcquireScene();
        const tcvr_m2_frame* frame = m_m2ImmFrame;
        if (!frame) { immTrace("no_frame"); return false; }
        // The geometry engine's focal length is zero until the game programs
        // it. Undoing the projection then divides by 1e-6 and the scene
        // explodes, so fall back on the flat window until it is real.
        // FIXED 16/09: the geometry engine's focal length reads ZERO on most
        // frames in scene-recording mode (measured: 47 of 51). Bailing then
        // (the old behaviour) reused the last eye texture -- the frozen picture,
        // while MAME published fresh geometry every frame. The focus is a slowly
        // changing camera parameter, so cache the last real one and keep
        // rendering the live geometry with it; only bail before we ever saw one.
        // The geometry engine programs its focal length only when a 3D camera
        // is active; in attract and menus it reads zero, and Sega Rally's focus
        // is in fact the constant 512. Bailing on zero (the old behaviour) fell
        // through to the flat window, which is empty and expensive in scene mode
        // 2 -- the frozen, slow picture. Never bail: use the last real focus, or
        // a sane default, and always render the live geometry.
        if (frame->focus_x > 1.0f && frame->focus_y > 1.0f) { m_m2FocusX = frame->focus_x; m_m2FocusY = frame->focus_y; }
        const float defFocus = arcadexr::config::GetFloat("m2.immersiveFocus", 512.0f);
        const float focusX = m_m2FocusX > 1.0f ? m_m2FocusX : defFocus;
        const float focusY = m_m2FocusY > 1.0f ? m_m2FocusY : defFocus;
        const bool forceRefresh = arcadexr::config::GetInt("m2.forceRefresh", 0) != 0;
        if (forceRefresh || frame->sequence != m_m2ImmPrepared) {
            if (!m_m2Gpu.PrepareFrame(*frame)) { immTrace("prepare_failed"); return false; }
            m_m2ImmPrepared = frame->sequence;
        }
        arcadexr::gun::ScreenPlane screen;
        if (!arcadexr::video::GetVirtualScreen(screen)) { immTrace("no_virtual_screen"); return false; }
        const float distance = std::max(0.25f, arcadexr::config::GetFloat("screen.distance", 2.0f));
        // Board units per metre, as on the System 22 path: depthUnits board
        // units sit at `distance` metres, so worldScale = distance/depthUnits.
        //
        // MEASURED on srallyc in game (9572 vertices): z percentiles p10=13.7
        // p50=175 p90=502 -- the scene spans about 500 board units of depth,
        // which 1200 squeezed into 0.83 m in front of the player, so a 20 cm
        // head movement crossed a quarter of the world. Two independent
        // physical references agree (what is just ahead ~3 m, the far scenery
        // ~100 m): about 4.7 units/m, hence 9.4 at 2 m.
        //
        // The scale cannot be read off the picture -- scaling the world and the
        // camera distance together gives an identical image -- it shows up only
        // as parallax, which is exactly what the complaint was about.
        const float depthUnits = std::max(0.5f, arcadexr::config::GetFloat("m2.immersiveDepth", 9.4f));
        const float worldScale = distance / depthUnits;
        const arcadexr::gun::Vec3 camera{screen.center.x + screen.normal.x * distance, screen.center.y + screen.normal.y * distance, screen.center.z + screen.normal.z * distance};
        // Camera pitch, READ from the background layer's horizon row: Sega
        // Rally's camera looks down at the road, and mapping its forward axis
        // onto the horizontal tilted the whole race downwards. The horizon
        // direction (0, tan, 1) of the board is rotated onto the anchor's
        // forward. The HUD and the secondary views stay on the screen; the
        // backdrop pitches with the world. m2.immersivePitch=0 disables it.
        float pitchTarget = 0.0f;
        if (arcadexr::config::GetInt("m2.immersivePitch", 1) != 0 && m_m2Gpu.HaveMainView() && m_m2Gpu.HorizonRow() >= 0.0f) {
            const float t = ((384.0f - float(m_m2Gpu.MainCenterY())) + float(frame->crtc_yoffset) - m_m2Gpu.HorizonRow()) / std::max(focusY, 1.0f);
            pitchTarget = std::max(-0.35f, std::min(0.35f, ::atanf(t)));
        }
        pitchTarget += arcadexr::config::GetFloat("m2.immersivePitchOffset", 0.0f);
        if (viewIndex == 0) m_m2Pitch += (pitchTarget - m_m2Pitch) * 0.1f;
        const float cp = ::cosf(m_m2Pitch), sp = ::sinf(m_m2Pitch);
        const arcadexr::gun::Vec3 upP{cp * screen.up.x - sp * screen.normal.x, cp * screen.up.y - sp * screen.normal.y, cp * screen.up.z - sp * screen.normal.z};
        const arcadexr::gun::Vec3 normalP{cp * screen.normal.x + sp * screen.up.x, cp * screen.normal.y + sp * screen.up.y, cp * screen.normal.z + sp * screen.up.z};
        const arcadexr::gun::Vec3 centerP{camera.x - normalP.x * distance, camera.y - normalP.y * distance, camera.z - normalP.z * distance};
        XrMatrix4x4f arcadeToWorld{};
        arcadeToWorld.m[0] = screen.right.x * worldScale;  arcadeToWorld.m[1] = screen.right.y * worldScale;  arcadeToWorld.m[2] = screen.right.z * worldScale;
        arcadeToWorld.m[4] = upP.x * worldScale;     arcadeToWorld.m[5] = upP.y * worldScale;     arcadeToWorld.m[6] = upP.z * worldScale;
        arcadeToWorld.m[8] = -normalP.x * worldScale; arcadeToWorld.m[9] = -normalP.y * worldScale; arcadeToWorld.m[10] = -normalP.z * worldScale;
        arcadeToWorld.m[12] = camera.x; arcadeToWorld.m[13] = camera.y; arcadeToWorld.m[14] = camera.z; arcadeToWorld.m[15] = 1.0f;
        const float farMetres = std::max(200.0f, arcadexr::config::GetFloat("immersive.far", 20000.0f));
        // Near plane, with its own key and its own ceiling -- the System 22
        // occurrence of this line is deliberately left alone. That path clamps
        // to 5 cm, which is right at its scale. At the Model 2 scale measured
        // here the board emits a few vertices below z = 1 unit: they used to
        // sit at 1.2 mm -- behind any near plane, invisible -- and at the
        // corrected scale they land 15 cm from the eye and fill the view.
        // Those were the polygons flying in all directions. 25 cm is also
        // where a VR near plane belongs; closer cannot be fused comfortably.
        const float nearMetres = std::max(0.002f, std::min(1.0f,
            arcadexr::config::GetFloat("m2.immersiveNear", 0.25f)));
        XrMatrix4x4f projection; XrMatrix4x4f_CreateProjectionFov(&projection, GRAPHICS_OPENGL_ES, layerView.fov, nearMetres, farMetres);
        XrMatrix4x4f eyeToWorld; XrMatrix4x4f_CreateFromRigidTransform(&eyeToWorld, &layerView.pose);
        XrMatrix4x4f worldToEye; XrMatrix4x4f_InvertRigidBody(&worldToEye, &eyeToWorld);
        XrMatrix4x4f viewProjection; XrMatrix4x4f_Multiply(&viewProjection, &projection, &worldToEye);
        XrMatrix4x4f mvp; XrMatrix4x4f_Multiply(&mvp, &viewProjection, &arcadeToWorld);
        // The 2D planes (HUD, secondary views, backdrop) at their own distance,
        // same angular size as the screen: m2.hudDistance metres (3 by default),
        // further than the 2 m world reference so the HUD stops sitting on the
        // player's nose while the road runs underneath.
        const float hudDistance = std::max(0.5f, arcadexr::config::GetFloat("m2.hudDistance", 10.0f));
        const float hudK = hudDistance / distance;
        // MEASURED 15/09: Sega Rally puts the speed/time HUD along the BOTTOM
        // of its 4:3 frame. On a flat plane facing the player that bottom edge
        // dips below the eye and the vertical plane pokes into the road ("le
        // compteur est dans la route"). Lift the whole HUD plane up along the
        // screen up axis. m2.hudLift in metres.
        const float hudLift = arcadexr::config::GetFloat("m2.hudLift", 1.8f);
        const arcadexr::gun::Vec3 hudCenter{camera.x - screen.normal.x * hudDistance + screen.up.x * hudLift, camera.y - screen.normal.y * hudDistance + screen.up.y * hudLift, camera.z - screen.normal.z * hudDistance + screen.up.z * hudLift};
        XrMatrix4x4f hudToWorld{};
        hudToWorld.m[0] = screen.right.x * screen.width * hudK; hudToWorld.m[1] = screen.right.y * screen.width * hudK; hudToWorld.m[2] = screen.right.z * screen.width * hudK;
        hudToWorld.m[4] = screen.up.x * screen.height * hudK;   hudToWorld.m[5] = screen.up.y * screen.height * hudK;   hudToWorld.m[6] = screen.up.z * screen.height * hudK;
        hudToWorld.m[8] = screen.normal.x; hudToWorld.m[9] = screen.normal.y; hudToWorld.m[10] = screen.normal.z;
        hudToWorld.m[12] = hudCenter.x; hudToWorld.m[13] = hudCenter.y; hudToWorld.m[14] = hudCenter.z; hudToWorld.m[15] = 1.0f;
        XrMatrix4x4f hudMvp; XrMatrix4x4f_Multiply(&hudMvp, &viewProjection, &hudToWorld);
        XrMatrix4x4f hudToWorldBack{};
        const arcadexr::gun::Vec3 backCenter{camera.x - normalP.x * hudDistance + screen.up.x * hudLift, camera.y - normalP.y * hudDistance + screen.up.y * hudLift, camera.z - normalP.z * hudDistance + screen.up.z * hudLift};
        hudToWorldBack.m[0] = screen.right.x * screen.width * hudK; hudToWorldBack.m[1] = screen.right.y * screen.width * hudK; hudToWorldBack.m[2] = screen.right.z * screen.width * hudK;
        hudToWorldBack.m[4] = upP.x * screen.height * hudK; hudToWorldBack.m[5] = upP.y * screen.height * hudK; hudToWorldBack.m[6] = upP.z * screen.height * hudK;
        hudToWorldBack.m[8] = normalP.x; hudToWorldBack.m[9] = normalP.y; hudToWorldBack.m[10] = normalP.z;
        hudToWorldBack.m[12] = backCenter.x; hudToWorldBack.m[13] = backCenter.y; hudToWorldBack.m[14] = backCenter.z; hudToWorldBack.m[15] = 1.0f;
        XrMatrix4x4f hudMvpBack; XrMatrix4x4f_Multiply(&hudMvpBack, &viewProjection, &hudToWorldBack);
        // MEASURED 15/09: full per-eye resolution with MSAA 4x drops the race
        // to ~30/120 (GPU bound: two eyes, two passes, high overdraw). 0.8 and
        // MSAA 2x hold it; both are live-tunable if the player wants sharper.
        const float renderScale = std::max(0.3f, std::min(2.0f, arcadexr::config::GetFloat("m2.immersiveScale", 1.2f)));
        const int eyeW = layerView.subImage.imageRect.extent.width, eyeH = layerView.subImage.imageRect.extent.height;
        const int rw = std::max(64, int(eyeW * renderScale)), rh = std::max(64, int(eyeH * renderScale));
        if (m_immersiveTexW != rw || m_immersiveTexH != rh) {
            for (int e = 0; e < 2; ++e) {
                if (!m_immersiveTex[e]) glGenTextures(1, &m_immersiveTex[e]);
                glBindTexture(GL_TEXTURE_2D, m_immersiveTex[e]);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rw, rh, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                m_immersiveHasImage[e] = false;
            }
            glBindTexture(GL_TEXTURE_2D, 0);
            if (!m_immersiveBlitFbo) glGenFramebuffers(1, &m_immersiveBlitFbo);
            m_immersiveTexW = rw; m_immersiveTexH = rh;
        }
        // Painter's bias: the sign is baked into the vertex shader, this sets
        // only the magnitude, in NDC per primitive index.
        // MEASURED 15/09: 4e-8 per index is below the 24-bit depth quantum
        // (1.2e-7 NDC), so coplanar polygons (road markings, car previews,
        // the mirror) z-fought: see-through shimmer. 6e-7 is five quanta.
        m_m2Gpu.SetDepthBias(arcadexr::config::GetFloat("m2.immersiveDepthBias", 6.0e-7f));
        // One frame in 15, not every frame. The probe's glReadPixels forces the
        // MSAA buffer to be materialised out of tile memory and cost 0.43 ms per
        // frame (5.74 vs 5.31 ms) for a ground colour that changes slowly.
        m_m2Gpu.SetProbeInterval(arcadexr::config::GetInt("m2.probeInterval", 15));
        m_m2Gpu.SetEdgeFade(arcadexr::config::GetFloat("m2.immersiveEdgeFade", 20.0f));
        // 4x by default, not 2x. Measured on srallyc with a RELAUNCH between each
        // reading -- changing this hot does not reallocate the framebuffer, and a
        // hot change is what made an earlier reading claim MSAA cost 16 ms:
        //   MSAA 0 -> 5.24 ms | MSAA 2 -> 5.31 ms | MSAA 4 -> 5.52 ms
        // So 4x samples cost 0.28 ms, and the pass holds 120/120 either way.
        m_m2Gpu.SetMsaa(std::max(0, std::min(4, arcadexr::config::GetInt("m2.immersiveMsaa", 4))));
        m_m2Gpu.SetRaw(arcadexr::config::GetInt("m2.immersiveRaw", 1) != 0);
        m_m2Gpu.SetFarMin(arcadexr::config::GetInt("m2.immersiveFarMin", 0));
        // E1 (16/09): depth = the board's draw order, one value per polygon. m2.depthOrder=0 goes back to geometry + bias.
        m_m2Gpu.SetDepthOrder(arcadexr::config::GetInt("m2.depthOrder", 1) != 0);
        // E2 (16/09): anisotropic taps against grazing-angle texel shimmer (road, car decals). GPU has the headroom.
        m_m2Gpu.SetAniso(std::max(1, std::min(8, arcadexr::config::GetInt("m2.aniso", 1))));
        // m2.mipBias in mml units (128 = one mip level blurrier); tames text/decal shimmer.
        m_m2Gpu.SetMipBias(std::max(0, std::min(512, arcadexr::config::GetInt("m2.mipBias", 0))));
        m_m2Gpu.SetHideHud(arcadexr::config::GetInt("m2.hideHud", 0) != 0);
        // E3 (17/09): SELECTIVE depth pre-pass. m2.prepassMode 0=off,1=untextured solids,2=+microtextured,3=all opaque.
        m_m2Gpu.SetPrepassMode(std::max(0, std::min(3, arcadexr::config::GetInt("m2.prepassMode", 0))));
        immTrace("render");
        // NO-HYPOTHESIS CHAIN TABLE (16/09, Guillaume's demand): one line, the
        // six ids, so we SEE which one freezes first. rawHash is computed here
        // over the pre-XR raw geometry the immersive is about to consume.
        if (viewIndex == 0) {
            // Colleague's test (16/09): on the SAME frame, hash the raw 3D the
            // immersive consumes AND the 2D layer that visibly animates. If 2D
            // changes while raw3D stays constant during motion, the bug is in the
            // copy/publication of the 3D stream, not the game.
            std::uint32_t rawHash = 2166136261u;
            const std::size_t nb = std::min<std::size_t>(frame->raw_vertex_count * 5u, 40000u);
            const float* rv = reinterpret_cast<const float*>(frame->raw_vertices);
            for (std::size_t i = 0; i < nb; ++i) { std::uint32_t b; std::memcpy(&b, &rv[i], 4); rawHash = (rawHash ^ b) * 16777619u; }
            std::uint32_t clipHash = 2166136261u;
            const std::size_t nbc = std::min<std::size_t>(frame->vertex_count, 8000u);
            for (std::size_t i = 0; i < nbc; ++i) { std::uint32_t b; std::memcpy(&b, &frame->vertices[i].x, 4); clipHash = (clipHash ^ b) * 16777619u; std::memcpy(&b, &frame->vertices[i].y, 4); clipHash = (clipHash ^ b) * 16777619u; }
            std::uint32_t hud2d = 2166136261u;
            if (frame->front2d && frame->front2d_stride && frame->width > 0 && frame->height > 0) {
                for (int y = 0; y < frame->height; y += 8) { const std::uint32_t* row = frame->front2d + std::size_t(y) * frame->front2d_stride; for (int x = 0; x < frame->width; x += 8) hud2d = (hud2d ^ row[x]) * 16777619u; }
            }
            std::uint32_t back2d = 2166136261u;
            if (frame->back2d && frame->back2d_stride && frame->width > 0 && frame->height > 0) {
                for (int y = 0; y < frame->height; y += 8) { const std::uint32_t* row = frame->back2d + std::size_t(y) * frame->back2d_stride; for (int x = 0; x < frame->width; x += 8) back2d = (back2d ^ row[x]) * 16777619u; }
            }
            static unsigned s_t = 0;
            if ((s_t++ % 30u) == 0u)
                __android_log_print(ANDROID_LOG_INFO, "TCVR_TABLE",
                    "mameFrame=%u emuTime=%.3f raw3D=%08x clip3D=%08x hud2d=%08x back2d=%08x rawCount=%u sceneGen=%llu",
                    frame->mame_frame, frame->emu_time, rawHash, clipHash, hud2d, back2d, frame->raw_vertex_count,
                    (unsigned long long)frame->sequence);
        }
        const bool chainReRender = (forceRefresh || frame->sequence != m_immersiveRenderedSeq[viewIndex] || !m_immersiveHasImage[viewIndex]);
        bool chainBlit = false;
        bool rendered = false;
        if (chainReRender) {
            rendered = m_m2Gpu.RenderImmersive(m_immersiveTex[viewIndex], rw, rh, reinterpret_cast<const float*>(&mvp), reinterpret_cast<const float*>(&hudMvp), reinterpret_cast<const float*>(&hudMvpBack),
                                               focusX, focusY, frame->crtc_xoffset, frame->crtc_yoffset);
            if (rendered) { m_immersiveRenderedSeq[viewIndex] = frame->sequence; m_immersivePose[viewIndex] = layerView.pose; m_immersiveFov[viewIndex] = layerView.fov; m_immersiveHasImage[viewIndex] = true; ++m_immersiveRenders; }
            else { m_m2GpuFailed = true; __android_log_print(ANDROID_LOG_ERROR, "TCVR_M2GPU", "immersive render failed: %s", m_m2Gpu.LastError().c_str()); }
        } else {
            auto& submitted = const_cast<XrCompositionLayerProjectionView&>(layerView);
            submitted.pose = m_immersivePose[viewIndex]; submitted.fov = m_immersiveFov[viewIndex];
            rendered = true;
        }
        if (rendered) {
            chainBlit = true;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, m_immersiveBlitFbo);
            glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_immersiveTex[viewIndex], 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_swapchainFramebuffer);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);
            glBlitFramebuffer(0, 0, rw, rh, layerView.subImage.imageRect.offset.x, layerView.subImage.imageRect.offset.y,
                              layerView.subImage.imageRect.offset.x + eyeW, layerView.subImage.imageRect.offset.y + eyeH, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        }
        if (viewIndex == 0) {
            static std::uint64_t s_lastAcq = 0; static unsigned s_reRender = 0, s_reuse = 0, s_blit = 0, s_tick = 0;
            if (chainReRender) s_reRender++; else s_reuse++;
            if (chainBlit) s_blit++;
            if ((++s_tick % 120u) == 0u) {
                __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU",
                    "CHAIN/120f: acqSeq=%llu (advanced=%d) renderedSeq=%llu reRender=%u reuse=%u blit=%u swapTex=%u",
                    (unsigned long long)frame->sequence, frame->sequence != s_lastAcq ? 1 : 0,
                    (unsigned long long)m_immersiveRenderedSeq[viewIndex], s_reRender, s_reuse, s_blit, colorTexture);
                s_lastAcq = frame->sequence; s_reRender = s_reuse = s_blit = 0;
            }
        }
        {
            // dump=<tag>: the Model 2 immersive eye texture, one PPM per eye,
            // so the result can be judged without a human in the headset.
            const std::string tag = arcadexr::config::GetString("dump", "");
            if (rendered && !tag.empty() && tag != m_m2DumpTag[viewIndex]) {
                m_m2DumpTag[viewIndex] = tag;
                std::vector<unsigned char> rgba;
                if (m_m2Gpu.ReadBack(m_immersiveTex[viewIndex], rw, rh, rgba)) {
                    const std::string path = arcadexr::config::ExternalDirectory() + "/dump-" + tag +
                                             (viewIndex == 0 ? "-m2imm-L.ppm" : "-m2imm-R.ppm");
                    if (FILE* f = std::fopen(path.c_str(), "wb")) {
                        std::fprintf(f, "P6\n%d %d\n255\n", rw, rh);
                        std::vector<unsigned char> row(size_t(rw) * 3);
                        for (int y = rh - 1; y >= 0; --y) {
                            const unsigned char* s2 = rgba.data() + size_t(y) * size_t(rw) * 4;
                            for (int x = 0; x < rw; ++x) { row[x*3] = s2[x*4]; row[x*3+1] = s2[x*4+1]; row[x*3+2] = s2[x*4+2]; }
                            std::fwrite(row.data(), 1, row.size(), f);
                        }
                        std::fclose(f);
                        __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "dumped %s (%dx%d)", path.c_str(), rw, rh);
                    }
                }
            }
        }
        m_immersiveActive = rendered;
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);
        glViewport(layerView.subImage.imageRect.offset.x, layerView.subImage.imageRect.offset.y, eyeW, eyeH);
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glDisable(GL_BLEND); glActiveTexture(GL_TEXTURE0);
        if (rendered && !m_loggedM2Imm[viewIndex]) { m_loggedM2Imm[viewIndex] = true; __android_log_print(ANDROID_LOG_INFO, "TCVR_M2GPU", "%s", Fmt("TCVR_M2GPU immersive view=%u %dx%d focus=%.1f,%.1f crtc=%d,%d depthUnits=%.0f", viewIndex, rw, rh, frame->focus_x, frame->focus_y, frame->crtc_xoffset, frame->crtc_yoffset, depthUnits).c_str()); }
        return rendered;
    }

    void MaybeRenderModel2Gpu() {
        if (arcadexr::profiles::GetString("presentation", "screen") == "immersive") return;
        static const int requested = [] {
            // Settings file first (m2.gpuRaster, default 4 = on), the test property overrides.
            char value[PROP_VALUE_MAX] = {};
            int n = arcadexr::config::GetInt("m2.gpuRaster", 4);
            if (__system_property_get("debug.tcvr.m2.gpuRaster", value) > 0) n = atoi(value);
            return (n < 0) ? 0 : (n > 8 ? 8 : n);
        }();
        if (requested == 0) return;
        if (!arcadexr::hardware::sega_model2::HaveSceneSource()) return;

        // Ask for the recording only once something reads it.
        if (!m_m2Requested) {
            m_m2Requested = true;
            SetM2SceneMode(1);
            Log::Write(Log::Level::Info, Fmt("TCVR_M2GPU recording requested, scale x%d", requested));
            return;   // the first scene lands on the next frame
        }
        // The flat window composites on the emulator's own frame and falls back
        // on it, so MAME must keep rasterising here.
        SetM2SceneMode(1);
        if (!m_m2Gpu.Ready() && !m_m2GpuFailed) {
            if (!m_m2Gpu.Initialize()) {
                m_m2GpuFailed = true;
                Log::Write(Log::Level::Error,
                           Fmt("TCVR_M2GPU disabled: %s", m_m2Gpu.LastError().c_str()));
                return;
            }
        }
        if (m_m2GpuFailed) return;

        const tcvr_m2_frame* frame = arcadexr::hardware::sega_model2::AcquireScene();
        if (frame == nullptr || frame->sequence == m_m2LastSequence) return;
        m_m2LastSequence = frame->sequence;

        const int width = (frame->width > 0 ? frame->width : 496) * requested;
        const int height = (frame->height > 0 ? frame->height : 384) * requested;
        if (m_m2Texture == 0) glGenTextures(1, &m_m2Texture);
        if (width != m_m2Width || height != m_m2Height) {
            glBindTexture(GL_TEXTURE_2D, m_m2Texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         nullptr);
            // Minified on the virtual screen: without mipmaps a 4x image aliases and shimmers on every edge.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_m2Width = width;
            m_m2Height = height;
        }

        {
            char stipple[PROP_VALUE_MAX] = {};
            // m2.stipple: 1 = order-exact 50% blend (default), 0 = the board's one pixel in two.
            bool blend = arcadexr::config::GetInt("m2.stipple", 1) != 0;
            if (__system_property_get("debug.tcvr.m2.stipple", stipple) > 0) blend = (stipple[0] == '1');
            m_m2Gpu.SetStippleBlend(blend);
            m_m2Gpu.SetLayerSmooth(arcadexr::config::GetInt("m2.layerSmooth", 2));
        }
        if (!m_m2Gpu.PrepareFrame(*frame)) return;
        // Stereo window (FENETRE 3D): one render per eye with the camera moved
        // by +-strength/2 board units, converging at m2.stereoConvergence.
        // ECRAN PLAT (presentation=flat) shows the left texture to both eyes,
        // so it must be rendered without any eye offset: with the offset kept,
        // every polygon was shifted by strength*focus/(2z) pixels against the
        // 2D layers -- up to 150 px for the nearest ones (seen live 15/09).
        const bool flatPresentation = arcadexr::profiles::GetString("presentation", "screen") == "flat";
        const float strength = flatPresentation ? 0.0f : arcadexr::config::GetFloat("m2.stereoStrength", 0.0f);
        const float convergence = arcadexr::config::GetFloat("m2.stereoConvergence", 0.0f);
        m_m2Stereo = strength != 0.0f;
        if (m_m2Stereo && m_m2TexR == 0) {
            glGenTextures(1, &m_m2TexR);
            glBindTexture(GL_TEXTURE_2D, m_m2TexR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_m2TexRW = width; m_m2TexRH = height;
        }
        if (m_m2Stereo && (m_m2TexRW != width || m_m2TexRH != height)) {
            glBindTexture(GL_TEXTURE_2D, m_m2TexR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_m2TexRW = width; m_m2TexRH = height;
        }
        const int eyes = m_m2Stereo ? 2 : 1;
        for (int eye = 0; eye < eyes; ++eye) {
            m_m2Gpu.SetStereo(m_m2Stereo ? (eye == 0 ? -0.5f : 0.5f) * strength : 0.0f, convergence);
            const GLuint target = (eye == 0) ? m_m2Texture : m_m2TexR;
            if (!m_m2Gpu.RenderTo(target, width, height, requested, m_screenTexture, m_frameWidth, m_frameHeight)) {
                m_m2GpuFailed = true;
                Log::Write(Log::Level::Error, Fmt("TCVR_M2GPU render failed: %s", m_m2Gpu.LastError().c_str()));
                return;
            }
            glBindTexture(GL_TEXTURE_2D, target);
            glGenerateMipmap(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        m_m2SourceTexture = m_m2Texture;

        if (++m_m2Frames % 60 == 0) {
            Log::Write(Log::Level::Info,
                       Fmt("TCVR_M2GPU %dx%d prims=%u verts=%u dirty_blocks=%u dropped=%u",
                           width, height, m_m2Gpu.LastPrimCount(), m_m2Gpu.LastVertexCount(),
                           m_m2Gpu.LastDirtyBlocks(), frame->dropped_prims));
        }
    }

    void RunUpscalePass(const XrCompositionLayerProjectionView& layerView) {
        int requested = arcadexr::profiles::GetInt("upscale", 1);
        if (requested < 1) requested = 1;
        if (requested > 4) requested = 4;
        if (requested == 1 || m_frameWidth <= 0 || m_frameHeight <= 0) {
            if (m_loggedUpscaleFactor != 1) {
                m_loggedUpscaleFactor = 1;
                m_loggedUpscale = true;
                Log::Write(Log::Level::Info, "TCVR_M11 upscale off: the quad samples the 640x480 source directly");
            }
            m_upscaleFactor = 1;
            return;
        }

        const int width = m_frameWidth * requested;
        const int height = m_frameHeight * requested;
        if (width != m_upscaleWidth || height != m_upscaleHeight) {
            glBindTexture(GL_TEXTURE_2D, m_upscaleTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            m_upscaleWidth = width;
            m_upscaleHeight = height;
        }
        m_upscaleFactor = requested;

        // Default to the smooth reconstruction: the source is rendered 3D, and
        // sharp-bilinear exists to preserve pixel-art blocks, which is the
        // opposite of what this content wants.
        // Default: edge. Judged on the same frozen frame at 1:1 (docs/validation/
        // filters/comparatif-*.png): bicubic keeps texture definition that
        // bilinear washes out, and the FXAA pass on top is the only mode that
        // visibly softens the polygon staircases without touching texture
        // interiors; text stays readable. +0.06 ms of GPU over bicubic.
        const std::string filter = arcadexr::profiles::GetString("filter", "edge");
        const bool edge = (filter == "edge");
        const int filterIndex = (filter == "nearest")    ? 0
                                : (filter == "bilinear") ? 1
                                : (filter == "sharp")    ? 2
                                                         : 3;   // bicubic, catmull, edge
        float sharpen = arcadexr::profiles::GetFloat("sharpen", 0.0f);
        if (sharpen < 0.0f) sharpen = 0.0f;
        if (sharpen > 2.0f) sharpen = 2.0f;

        glBindFramebuffer(GL_FRAMEBUFFER, m_upscaleFramebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_upscaleTexture, 0);
        glViewport(0, 0, width, height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glUseProgram(m_upscaleProgram);
        glUniform2f(m_upscaleSourceSizeLocation, float(m_frameWidth), float(m_frameHeight));
        glUniform2f(m_upscaleTargetSizeLocation, float(width), float(height));
        glUniform1i(m_upscaleFilterLocation, filterIndex);
        glUniform1f(m_upscaleSharpenLocation, sharpen);
        glActiveTexture(GL_TEXTURE0);
        // The Model 2 GPU pass, when it runs, produces the same 3D layer the
        // CPU rasteriser would have written -- at any resolution. Everything
        // downstream (filter, quad, stereo) is left untouched so the two paths
        // are comparable.
        glBindTexture(GL_TEXTURE_2D, m_screenTexture);
        glUniform1i(m_upscaleSourceLocation, 0);
        glBindVertexArray(m_upscaleVao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        m_finalTexture = m_upscaleTexture;
        if (edge) {
            if (width != m_edgeWidth || height != m_edgeHeight) {
                glBindTexture(GL_TEXTURE_2D, m_edgeTexture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
                glBindTexture(GL_TEXTURE_2D, 0);
                m_edgeWidth = width;
                m_edgeHeight = height;
            }
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_edgeTexture, 0);
            glUseProgram(m_edgeProgram);
            glUniform2f(m_edgeTargetSizeLocation, float(width), float(height));
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_upscaleTexture);
            glUniform1i(m_edgeSourceLocation, 0);
            glBindVertexArray(m_upscaleVao);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glUseProgram(0);
            m_finalTexture = m_edgeTexture;
        }

        // debug.tcvr.dump=<tag>: write the final texture to the external files
        // directory as a PPM, once per distinct tag. Lets the same frozen frame
        // be compared across filters at full resolution, outside the headset.
        {
            const std::string tag = arcadexr::config::GetString("dump", "");
            if (!tag.empty() && tag != m_lastDumpTag) {
                m_lastDumpTag = tag;
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_finalTexture, 0);
                std::vector<unsigned char> rgba(size_t(width) * size_t(height) * 4);
                glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
                const std::string path = arcadexr::config::ExternalDirectory() + "/dump-" + tag + ".ppm";
                if (FILE* f = std::fopen(path.c_str(), "wb")) {
                    std::fprintf(f, "P6\n%d %d\n255\n", width, height);
                    std::vector<unsigned char> row(size_t(width) * 3);
                    for (int y = height - 1; y >= 0; --y) {   // GL rows are bottom-up
                        const unsigned char* src = rgba.data() + size_t(y) * size_t(width) * 4;
                        for (int x = 0; x < width; ++x) { row[x*3] = src[x*4]; row[x*3+1] = src[x*4+1]; row[x*3+2] = src[x*4+2]; }
                        std::fwrite(row.data(), 1, row.size(), f);
                    }
                    std::fclose(f);
                    Log::Write(Log::Level::Info, Fmt("TCVR_M11 dumped %s (%dx%d, filter=%s sharpen=%.2f)", path.c_str(), width, height, filter.c_str(), sharpen));
                } else {
                    Log::Write(Log::Level::Warning, Fmt("TCVR_M11 dump failed: cannot open %s", path.c_str()));
                }
            }
        }

        // Hand the eye's framebuffer and viewport back exactly as they were.
        glBindFramebuffer(GL_FRAMEBUFFER, m_swapchainFramebuffer);
        glViewport(static_cast<GLint>(layerView.subImage.imageRect.offset.x),
                   static_cast<GLint>(layerView.subImage.imageRect.offset.y),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.width),
                   static_cast<GLsizei>(layerView.subImage.imageRect.extent.height));
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        if (!m_loggedUpscale || requested != m_loggedUpscaleFactor || filterIndex != m_loggedFilter ||
            sharpen != m_loggedSharpen) {
            m_loggedUpscale = true;
            m_loggedUpscaleFactor = requested;
            m_loggedFilter = filterIndex;
            m_loggedSharpen = sharpen;
            Log::Write(Log::Level::Info, Fmt("TCVR_M11 upscale x%d -> %dx%d filter=%s sharpen=%.2f", requested, width,
                                             height, filter.c_str(), sharpen));
        }
    }

    uint32_t GetSupportedSwapchainSampleCount(const XrViewConfigurationView&) override { return 1; }

    void SetClearColor(const std::array<float, 4> clearColor) override { m_clearColor = clearColor; }

   private:
#ifdef XR_USE_PLATFORM_ANDROID
    XrGraphicsBindingOpenGLESAndroidKHR m_graphicsBinding{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
#endif

    SwapchainImageDataMap<OpenGLESSwapchainImageData> m_swapchainImageDataMap;
    GLuint m_swapchainFramebuffer{0};
    GLuint m_program{0};
    GLint m_modelViewProjectionUniformLocation{0};
    GLint m_vertexAttribCoords{0};
    GLint m_vertexAttribColor{0};
    GLuint m_vao{0};
    GLuint m_cubeVertexBuffer{0};
    GLuint m_cubeIndexBuffer{0};
    GLuint m_screenProgram{0};
    bool m_bgraDirect{false};
    GLuint m_edgeProgram{0};
    GLuint m_edgeTexture{0};
    GLint m_edgeSourceLocation{0};
    GLint m_edgeTargetSizeLocation{0};
    GLuint m_darkFlickerProgram{0};
    GLint m_darkFlickerCurrentLocation{-1};
    GLint m_darkFlickerPreviousLocation{-1};
    GLint m_darkFlickerHavePreviousLocation{-1};
    int m_edgeWidth{0};
    int m_edgeHeight{0};
    GLuint m_finalTexture{0};
    // Model 2 GPU rasteriser: its output texture, and the override the upscale
    // pass reads instead of the emulator framebuffer once it has produced one.
    arcadexr::hardware::sega_model2::GpuRenderer m_m2Gpu;
    GLuint m_m2Texture{0};
    GLuint m_m2SourceTexture{0};
    GLuint m_m2TexR{0};
    int m_m2TexRW{0};
    int m_m2TexRH{0};
    bool m_m2Stereo{false};
    int m_m2Width{0}, m_m2Height{0};
    bool m_m2Requested{false};
    const tcvr_m2_frame* m_m2ImmFrame{nullptr};
    std::uint64_t m_m2ImmPrepared{0};
    bool m_loggedM2Imm[2]{false, false};
    bool m_m2ImmLoggedGate{false};
    int m_m2SceneMode{0};
    bool m_m2ImmLoggedFocus{false};
    unsigned m_m2ImmTick{0};
    float m_m2FocusX{0.0f}, m_m2FocusY{0.0f};
    float m_m2Pitch{0.0f};
    std::string m_m2DumpTag[2];
    bool m_m2GpuFailed{false};
    std::uint64_t m_m2LastSequence{0};
    unsigned m_m2Frames{0};
    std::string m_lastDumpTag;
    std::string m_settingsSignature;
    unsigned m_settingsPoll{0};
    GLuint m_gunProgram{0};
    GLuint m_gunLitProgram{0};
    GLuint m_gunLitVao{0};
    GLuint m_gunLitVertexBuffer{0};
    GLuint m_gunLitIndexBuffer{0};
    GLint m_gunLitMvpLocation{0};
    GLint m_gunLitModelLocation{0};
    GLint m_gunLitEyeLocation{0};
    GLsizei m_gunLitIndexCount{0};
    GLuint m_gunVao{0};
    GLuint m_gunVertexBuffer{0};
    GLint m_gunMvpLocation{0};
    GLsizei m_gunVertexCount{0};
    GLuint m_upscaleProgram{0};
    GLuint m_upscaleVao{0};
    GLuint m_upscaleFramebuffer{0};
    GLuint m_upscaleTexture{0};
    GLint m_upscaleSourceLocation{0};
    GLint m_upscaleSourceSizeLocation{0};
    GLint m_upscaleTargetSizeLocation{0};
    GLint m_upscaleFilterLocation{0};
    GLint m_upscaleSharpenLocation{0};
    int m_upscaleFactor{0};
    int m_upscaleWidth{0};
    int m_upscaleHeight{0};
    bool m_loggedUpscale{false};
    int m_loggedUpscaleFactor{0};
    int m_loggedFilter{-1};
    float m_loggedSharpen{-1.0f};
    GLuint m_screenTexture{0};
    GLuint m_screenVao{0};
    GLuint m_menuVao{0};
    GLuint m_screenVertexBuffer{0};
    GLuint m_screenIndexBuffer{0};
    GLint m_screenMvpUniformLocation{0};
    GLint m_screenTextureUniformLocation{0};
    GLint m_screenAimPointUniformLocation{0};
    GLint m_screenAimVisibleUniformLocation{0};
    GLint m_screenCalibratingLocation{-1};
    GLint m_objectTintLocation{-1};
    arcadexr::hardware::namco_system22::SceneRenderer m_scene;
    bool m_sceneInit{false};
    bool m_sceneActive{false};
    bool m_loggedEyeRoute[2]{false, false};
    bool m_loggedImmersiveRoute[2]{false, false};
    int m_sceneEnabled{-1};
    GLuint m_sceneTex[2]{0, 0};
    GLuint m_sceneTemporalRaw[2][2]{{0, 0}, {0, 0}};
    int m_sceneTemporalIndex[2]{0, 0};
    bool m_sceneTemporalValid[2]{false, false};
    int m_sceneTemporalW{0}, m_sceneTemporalH{0};
    int m_sceneW{0};
    int m_sceneH{0};
    std::uint64_t m_sceneSequence{0};
    std::string m_sceneSignature;
    std::string m_sceneGame;
    std::string m_sceneDumpTag;
    const tcvr_scene_frame* m_immersiveFrame{nullptr};
    bool m_immersiveActive{false};
    bool m_anchorValid{false};
    arcadexr::gun::Vec3 m_anchorCamera{}, m_anchorRight{}, m_anchorUp{}, m_anchorNormal{};
    float m_anchorScale{1.0f};
    GLuint m_immersiveTex[2]{0, 0};
    GLuint m_immersiveTemporalRaw[2][2]{{0, 0}, {0, 0}};
    int m_immersiveTemporalIndex[2]{0, 0};
    bool m_immersiveTemporalValid[2]{false, false};
    int m_immersiveTemporalW{0}, m_immersiveTemporalH{0};
    GLuint m_menuProgram{0};
    GLuint m_menuTexture{0};
    GLint m_menuMvpLocation{-1};
    GLint m_menuTexLocation{-1};
    float m_menuAspect{1.6f};
    XrPosef m_menuFramePose{};
    bool m_menuFramePoseValid{false};
    GLuint m_immersiveAa[2]{0, 0};
    int m_immersiveAaW{0};
    int m_immersiveAaH{0};
    GLuint m_immersiveBlitFbo{0};
    std::uint64_t m_immersiveRenderedSeq[2]{0, 0};
    XrPosef m_immersivePose[2]{};
    XrFovf m_immersiveFov[2]{};
    bool m_immersiveHasImage[2]{false, false};
    bool m_frozenValid[2]{false, false};
    XrPosef m_frozenPose[2]{};
    XrFovf m_frozenFov[2]{};
    std::string m_immersiveDumpTag[2];
    unsigned m_immersiveRenders{0};
    int m_immersiveTexW{0};
    int m_immersiveTexH{0};
    std::uint64_t m_immersivePreparedSequence{0};
    double m_immersivePrepMs{0};
    double m_immersiveDrawMs{0};
    unsigned m_immersivePreparedFrames{0};
    unsigned m_immersiveViews{0};
    double m_scenePrepMs{0};
    double m_sceneDrawMs{0};
    unsigned m_sceneFrames{0};
    std::vector<std::uint32_t> m_framePixels;
    std::vector<std::uint8_t> m_frameRgba;
    int m_frameWidth{0};
    int m_frameHeight{0};
    std::uint64_t m_lastFrameSequence{0};
    bool m_loggedScreenUpload{false};
    GLint m_contextApiMajorVersion{0};
    std::array<float, 4> m_clearColor;
};
OpenGLESGraphicsPlugin* OpenGLESGraphicsPlugin::s_self = nullptr;
}  // namespace

std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_OpenGLES() { return std::make_shared<OpenGLESGraphicsPlugin>(); }

#endif
