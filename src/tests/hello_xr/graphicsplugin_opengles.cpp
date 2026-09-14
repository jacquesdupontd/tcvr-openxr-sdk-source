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
#include "virtual_screen.h"
#include "settings.h"
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
        FragColor = color;
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
        glGenFramebuffers(1, &m_upscaleFramebuffer);
        glGenTextures(1, &m_upscaleTexture);
        glBindTexture(GL_TEXTURE_2D, m_upscaleTexture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);

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

    void RenderView(const XrCompositionLayerProjectionView& layerView, const XrSwapchainImageBaseHeader* swapchainImage,
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

        RenderArcadeScreen(layerView);

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
        glUseProgram(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void RenderArcadeScreen(const XrCompositionLayerProjectionView& layerView) {
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
        if (haveNewFrame && !arcadexr::video::CopyLatestFrame(m_framePixels, info)) {
            return;
        }
        if (haveNewFrame) {
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
        glBindTexture(GL_TEXTURE_2D, m_upscaleFactor > 1 ? m_upscaleTexture : m_screenTexture);
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

    // Magnifies the emulated frame into an offscreen texture. Called only when
    // a new emulated frame has just been uploaded, so its cost is tied to the
    // arcade clock -- sixty times a second -- and never to the presentation
    // clock, which on this headset runs at 207 Hz.
    void RunUpscalePass(const XrCompositionLayerProjectionView& layerView) {
        int requested = arcadexr::config::GetInt("upscale", 1);
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
        const std::string filter = arcadexr::config::GetString("filter", "catmull");
        const int filterIndex = (filter == "nearest")    ? 0
                                : (filter == "bilinear") ? 1
                                : (filter == "sharp")    ? 2
                                                         : 3;
        float sharpen = arcadexr::config::GetFloat("sharpen", 0.0f);
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
        glBindTexture(GL_TEXTURE_2D, m_screenTexture);
        glUniform1i(m_upscaleSourceLocation, 0);
        glBindVertexArray(m_upscaleVao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

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
    GLuint m_screenVertexBuffer{0};
    GLuint m_screenIndexBuffer{0};
    GLint m_screenMvpUniformLocation{0};
    GLint m_screenTextureUniformLocation{0};
    GLint m_screenAimPointUniformLocation{0};
    GLint m_screenAimVisibleUniformLocation{0};
    GLint m_screenCalibratingLocation{-1};
    GLint m_objectTintLocation{-1};
    std::vector<std::uint32_t> m_framePixels;
    std::vector<std::uint8_t> m_frameRgba;
    int m_frameWidth{0};
    int m_frameHeight{0};
    std::uint64_t m_lastFrameSequence{0};
    bool m_loggedScreenUpload{false};
    GLint m_contextApiMajorVersion{0};
    std::array<float, 4> m_clearColor;
};
}  // namespace

std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_OpenGLES() { return std::make_shared<OpenGLESGraphicsPlugin>(); }

#endif
