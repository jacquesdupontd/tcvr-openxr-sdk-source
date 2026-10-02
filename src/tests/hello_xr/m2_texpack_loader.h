#pragma once

// HD texture packs, the part that touches files (02/10/2026): the colour formula of each format read from the pack's
// own shaders (mode_N.ps), and PNG decoding off the render thread. Identifiers and the .pat are in m2_texpack.h.
//
// Included by vulkan_m2_renderer.h ONLY: stb_image is compiled here with static linkage, in that translation unit.

#include "m2_texpack.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include "../../../../mame/3rdparty/bimg/3rdparty/stb/stb_image.h"   // third_party/mame/3rdparty/bimg/3rdparty/stb, v2.26
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace arcadexr::texpack {

// One coloured format: rgb *= light * rgb ; alpha *= alpha. light: 0 = the polygon's light (LightMap.r), 1 = the game's
// luma table read at that light (lbase = tex2D(Luma, ...)), 2 = none.
struct Mode {
    float rgb = 1.0f, light = 0.0f, alpha = 1.0f;
    bool ok = false;
    std::string why;
};

// The emulator compiles SCRIPTS/mode_1.ps .. mode_16.ps at load and draws a coloured format N with mode_N.ps.
// Understood here: the form the packs use,
//     T.Color = tex2D(Texture, In.TexCoord0) [* m];      T.Color.rgb *= (In.LightMap.r | lbase) * k;
// with lbase = tex2D(Luma, Lum).x. Anything else is NOT guessed: that format keeps the original textures, and
// Report() says why. A format above 16 (the Daytona pack's "14", read by the emulator as 0x14 = 20) makes the
// emulator read past its 16 shader slots; what it then draws is unknown, so the choice is a setting
// (SetOutOfRange: 0 = rgb * light * 2 like the pack's own _mode_14.ps, 1 = like format 1, 2 = unlit, 3 = original).
class Modes {
public:
    static constexpr uint32_t kOutOfRange = 17;   // shader code of every format above 16

    void Load(const std::string& dir) {
        m_report.clear();
        for (Mode& m : m_modes) m = Mode{};
        std::unordered_map<std::string, std::string> files;
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) files.emplace(Lower(e->d_name), e->d_name);
            closedir(d);
        }
        for (uint32_t n = 1; n <= 16; ++n) {
            const auto it = files.find("mode_" + std::to_string(n) + ".ps");
            if (it == files.end()) { m_modes[n].why = "absent"; continue; }
            std::ifstream f(dir + "/" + it->second);
            std::stringstream ss;
            ss << f.rdbuf();
            Parse(ss.str(), m_modes[n]);
            char buf[160];
            if (m_modes[n].ok)
                std::snprintf(buf, sizeof buf, " %u:rgb*%s*%.3g a*%.3g", n, m_modes[n].light > 0.5f ? "luma" : "light",
                              m_modes[n].rgb, m_modes[n].alpha);
            else
                std::snprintf(buf, sizeof buf, " %u:NOT UNDERSTOOD (%s)", n, m_modes[n].why.c_str());
            m_report += buf;
        }
        SetOutOfRange(m_outOfRange);
    }

    void SetOutOfRange(int variant) {
        m_outOfRange = variant;
        Mode& m = m_modes[kOutOfRange];
        m = Mode{};
        if (variant == 0) { m.rgb = 2.0f; m.light = 0.0f; m.ok = true; }
        else if (variant == 1) { m = m_modes[1]; if (!m.ok) { m.rgb = 1.1f; m.light = 0.0f; m.alpha = 1.0f; m.ok = true; } }
        else if (variant == 2) { m.rgb = 1.0f; m.light = 2.0f; m.ok = true; }
        else m.why = "kept original (setting)";
    }

    // Shader code of a pack format: 0 = grey through the board's colour chain, 1..16 = mode_N.ps, kOutOfRange above;
    // -1 = keep the original texture (a shader we could not read).
    int Code(uint32_t format) const {
        if (format == 0) return 0;
        const uint32_t c = format <= 16 ? format : kOutOfRange;
        return m_modes[c].ok ? int(c) : -1;
    }
    const Mode& operator[](uint32_t code) const { return m_modes[code < 18 ? code : 0]; }
    const std::string& Report() const { return m_report; }

private:
    static std::string Lower(std::string s) {
        for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }
    static bool Number(const std::string& s, size_t& p, float& v) {
        const char* b = s.c_str() + p;
        char* e = nullptr;
        v = std::strtof(b, &e);
        if (e == b) return false;
        p += size_t(e - b);
        return true;
    }
    static void Parse(const std::string& text, Mode& m) {
        std::string s;   // comments and blanks removed
        for (size_t i = 0; i < text.size(); ++i) {
            if (text.compare(i, 2, "//") == 0) { while (i < text.size() && text[i] != '\n') ++i; continue; }
            if (text.compare(i, 2, "/*") == 0) { const size_t e = text.find("*/", i + 2); i = (e == std::string::npos) ? text.size() : e + 1; continue; }
            if (!std::isspace(static_cast<unsigned char>(text[i]))) s += text[i];
        }
        size_t uses = 0;
        for (size_t p = s.find("T.Color"); p != std::string::npos; p = s.find("T.Color", p + 1)) ++uses;
        const std::string a = "T.Color=tex2D(Texture,In.TexCoord0)", b = "T.Color.rgb*=";
        size_t p = s.find(a);
        if (p == std::string::npos) { m.why = "no T.Color=tex2D(Texture,In.TexCoord0)"; return; }
        p += a.size();
        float mul = 1.0f;
        if (s[p] == '*') { ++p; if (!Number(s, p, mul)) { m.why = "texture factor"; return; } }
        if (p >= s.size() || s[p] != ';') { m.why = "texture statement"; return; }
        size_t q = s.find(b);
        if (q == std::string::npos) { m.why = "no T.Color.rgb*="; return; }
        q += b.size();
        float light;
        const std::string lm = "In.LightMap.r*", lb = "lbase*";
        if (s.compare(q, lm.size(), lm) == 0) { light = 0.0f; q += lm.size(); }
        else if (s.compare(q, lb.size(), lb) == 0) {
            if (s.find("lbase=tex2D(Luma,Lum).x") == std::string::npos) { m.why = "lbase not from Luma"; return; }
            light = 1.0f; q += lb.size();
        } else { m.why = "light term"; return; }
        float k = 1.0f;
        if (!Number(s, q, k) || q >= s.size() || s[q] != ';') { m.why = "light factor"; return; }
        if (uses != 2) { m.why = "other uses of T.Color"; return; }
        m.rgb = mul * k; m.light = light; m.alpha = mul; m.ok = true;
    }

    Mode m_modes[18];
    int m_outOfRange = 0;
    std::string m_report;
};

// A decoded replacement, in the encoding the region images use (vulkan_m2_regions.h):
//   grey (format 0): R = grey, G = grey * alpha, B = alpha, A = 255   -- the board's colour chain does the rest
//   colour:          premultiplied RGBA                                  -- clean mips; the shader divides
// Larger than 1024 on a side: halved (box) until it fits, so one image fits the staging budget of a frame.
struct Image {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;
    bool alpha = false;
    std::string error;
    std::string path;
};

inline Image DecodeImage(const std::string& path, bool grey) {
    Image img;
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!data) { img.error = stbi_failure_reason() ? stbi_failure_reason() : "decode failed"; return img; }
    std::vector<uint8_t> rgba(data, data + size_t(w) * h * 4);
    stbi_image_free(data);
    while (w > 1024 || h > 1024) {
        const int nw = std::max(1, w / 2), nh = std::max(1, h / 2);
        std::vector<uint8_t> half(size_t(nw) * nh * 4);
        for (int y = 0; y < nh; ++y)
            for (int x = 0; x < nw; ++x)
                for (int c = 0; c < 4; ++c) {
                    const int x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1);
                    const int y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
                    const int sum = rgba[(size_t(y0) * w + x0) * 4 + c] + rgba[(size_t(y0) * w + x1) * 4 + c] +
                                    rgba[(size_t(y1) * w + x0) * 4 + c] + rgba[(size_t(y1) * w + x1) * 4 + c];
                    half[(size_t(y) * nw + x) * 4 + c] = uint8_t((sum + 2) / 4);
                }
        rgba.swap(half); w = nw; h = nh;
    }
    img.w = uint32_t(w); img.h = uint32_t(h);
    img.px.resize(size_t(w) * h * 4);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
        const uint8_t* s = &rgba[i * 4];
        uint8_t* d = &img.px[i * 4];
        const uint32_t a = s[3];
        if (a < 255) img.alpha = true;
        if (grey) { d[0] = s[0]; d[1] = uint8_t((s[0] * a + 127) / 255); d[2] = uint8_t(a); d[3] = 255; }
        else { d[0] = uint8_t((s[0] * a + 127) / 255); d[1] = uint8_t((s[1] * a + 127) / 255); d[2] = uint8_t((s[2] * a + 127) / 255); d[3] = uint8_t(a); }
    }
    return img;
}

// Decodes on its own thread, in request order; the render thread collects what is ready.
class Loader {
public:
    ~Loader() { Stop(); }

    void Request(uint64_t key, const std::string& path, bool grey) {
        {
            std::lock_guard<std::mutex> lk(m_mu);
            m_jobs.push_back({key, path, grey});
            if (!m_started) { m_started = true; m_stop = false; m_thread = std::thread([this] { Run(); }); }
        }
        m_cv.notify_one();
    }

    // Moves finished images out (at most `max` per call).
    template <class F>
    void Collect(F&& take, size_t max = 8) {
        std::lock_guard<std::mutex> lk(m_mu);
        while (max-- > 0 && !m_done.empty()) {
            take(m_done.front().first, std::move(m_done.front().second));
            m_done.pop_front();
        }
    }

    size_t Pending() {
        std::lock_guard<std::mutex> lk(m_mu);
        return m_jobs.size() + (m_busy ? 1u : 0u);
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(m_mu);
            if (!m_started) return;
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_thread.joinable()) m_thread.join();
        std::lock_guard<std::mutex> lk(m_mu);
        m_started = false;
        m_jobs.clear();
        m_done.clear();
    }

private:
    struct Job { uint64_t key; std::string path; bool grey; };
    void Run() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(m_mu);
                m_cv.wait(lk, [this] { return m_stop || !m_jobs.empty(); });
                if (m_stop) return;
                job = std::move(m_jobs.front());
                m_jobs.pop_front();
                m_busy = true;
            }
            Image img = DecodeImage(job.path, job.grey);
            img.path = job.path;
            std::lock_guard<std::mutex> lk(m_mu);
            m_done.emplace_back(job.key, std::move(img));
            m_busy = false;
        }
    }

    std::mutex m_mu;
    std::condition_variable m_cv;
    std::deque<Job> m_jobs;
    std::deque<std::pair<uint64_t, Image>> m_done;
    bool m_stop = false, m_started = false, m_busy = false;
    std::thread m_thread;   // last: started after the members it uses exist
};

}  // namespace arcadexr::texpack
