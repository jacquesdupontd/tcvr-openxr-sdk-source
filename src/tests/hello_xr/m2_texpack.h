#pragma once

// HD texture packs written for ElSemi's Model 2 Emulator 1.1a (02/10/2026): the .pat file and the exact ID1/ID2 pair.
//
// A pack is SCRIPTS/<romset>.pat, one line per replaced texture: "ID1 ID2 Format File.png", read by the emulator with
// sscanf("%x %x %x %s") -- all three numbers are HEXADECIMAL ("14" is format 0x14 = 20). A line starting with '/' is a
// comment. The emulator replaces a texture only when BOTH identifiers match:
//   ID1 = (th2 << 16) | (th0 & ~0x04C0)             th0 without its repeat bits (6, 7) and bit 10
//   ID2 = zlib crc32, seeded 0, chained over H/2 block rows of 2*W bytes, 1024 bytes apart, in the RAW texture RAM
//         (16-bit words of 2x2 texels, 512 words per block row), from byte ((y >> 1) << 10) + x where
//         x = (th2 & 0x1f) << 5, y = ((th2 >> 6) & 0x1f) << 5, plus 1024 to y when th2 & 0x20 (the sheet's fold).
// 2*W bytes is TWICE the texture's width: the fingerprint also covers the band of the same width to its right.
// Read in EMULATOR.EXE (routines 0x4B7B70, 0x4531E0, 0x4553A0 = zlib crc32, 0x4B87EA) and checked on Daytona USA:
// 86 textured headers out of 86 whose ID1 is in stf999's pack get their exact ID2 (skills/quest-vr-port/references/
// textures-hd-packs-model2-emulator.md). Windows ignores the case of file names; Android does not, so files are
// matched without case, and "name.png.png" is tried too (a file of the Daytona pack only exists that way).
//
// The raw sheet is MAME's share as the frame carries it (tcvr_m2_frame::textureram[sheet], u32 words, the even 16-bit
// word in the low half, little-endian): the renderer's m_texShadow[sheet] is that copy.
//
// Pure C++, no Vulkan, no MAME: compiled on the PC by scripts/texpack/test_m2_texpack.cpp against real dumps.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <dirent.h>

namespace arcadexr::texpack {

// zlib's crc32(crc, buf, len): reflected 0xEDB88320, pre- and post-inverted, so chaining works as in zlib.
inline uint32_t Crc32(uint32_t crc, const uint8_t* p, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

inline uint32_t Id1(uint32_t th0, uint32_t th2) { return ((th2 & 0xffffu) << 16) | (th0 & 0xffffu & ~0x04C0u); }

inline uint32_t Width(uint32_t th0) { return 32u << (th0 & 7u); }
inline uint32_t Height(uint32_t th0) { return 32u << ((th0 >> 3) & 7u); }

inline size_t StartByte(uint32_t th2) {
    uint32_t x = (th2 & 0x1fu) << 5, y = ((th2 >> 6) & 0x1fu) << 5;
    if (th2 & 0x20u) y += 0x400u;
    return (size_t(y >> 1) << 10) + x;
}

// sheet: raw bytes of the sheet that th2 selects (bit 12: textureram1). A row that runs past the end is hashed on the
// bytes that exist (the emulator would read whatever memory follows; no texture of the Daytona pack does this).
inline uint32_t Id2(uint32_t th0, uint32_t th2, const uint8_t* sheet, size_t bytes) {
    const uint32_t w = Width(th0), h = Height(th0);
    const size_t start = StartByte(th2);
    uint32_t crc = 0;
    for (uint32_t r = 0; r < (h - 1) / 2 + 1; ++r) {
        const size_t o = start + size_t(r) * 1024u;
        if (o >= bytes) break;
        crc = Crc32(crc, sheet + o, std::min<size_t>(2u * w, bytes - o));
    }
    return crc;
}

struct Entry {
    uint32_t id1 = 0, id2 = 0;
    uint32_t format = 0;     // as the emulator reads it: hexadecimal ("14" -> 20)
    std::string name;        // as written in the .pat
    std::string path;        // resolved file, empty when not found
    std::string note;        // "case", "double extension", "missing"
};

class Pack {
public:
    // Reads <dir>/<romset>.pat (name matched without case). Returns false when there is no pack for this game.
    bool Load(const std::string& dir, const std::string& romset) {
        m_entries.clear(); m_byId1.clear(); m_report.clear();
        std::unordered_map<std::string, std::string> files;   // lower-case name -> real name
        if (DIR* d = opendir(dir.c_str())) {
            while (dirent* e = readdir(d)) files.emplace(Lower(e->d_name), e->d_name);
            closedir(d);
        }
        const auto pat = files.find(Lower(romset + ".pat"));
        if (pat == files.end()) return false;
        FILE* f = std::fopen((dir + "/" + pat->second).c_str(), "r");
        if (!f) return false;
        char line[512];
        unsigned missing = 0, cased = 0, doubled = 0, comments = 0;
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '/') { ++comments; continue; }
            unsigned a = 0, b = 0, c = 0;
            char name[400] = {};
            if (std::sscanf(line, "%x %x %x %399[^\r\n]", &a, &b, &c, name) != 4) continue;
            std::string n = Trim(name);
            Entry e;
            e.id1 = a; e.id2 = b; e.format = c; e.name = n;
            auto it = files.find(Lower(n));
            if (it != files.end()) {
                e.path = dir + "/" + it->second;
                if (it->second != n) { e.note = "case"; ++cased; }
            } else if ((it = files.find(Lower(n + ".png"))) != files.end()) {
                e.path = dir + "/" + it->second; e.note = "double extension"; ++doubled;
            } else {
                e.note = "missing"; ++missing;
            }
            m_byId1[e.id1].push_back(m_entries.size());
            m_entries.push_back(std::move(e));
        }
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf, "%zu entries, %zu ID1, %u comment lines, %u missing, %u case fixed, %u double extension",
                      m_entries.size(), m_byId1.size(), comments, missing, cased, doubled);
        m_report = buf;
        return true;
    }

    bool HasId1(uint32_t id1) const { return m_byId1.count(id1) != 0; }

    const Entry* Find(uint32_t id1, uint32_t id2) const {
        const auto it = m_byId1.find(id1);
        if (it == m_byId1.end()) return nullptr;
        for (size_t i : it->second)
            if (m_entries[i].id2 == id2) return &m_entries[i];
        return nullptr;
    }

    const std::vector<Entry>& Entries() const { return m_entries; }
    const std::string& Report() const { return m_report; }

private:
    static std::string Lower(std::string s) {
        for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    }
    static std::string Trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    }
    std::vector<Entry> m_entries;
    std::unordered_map<uint32_t, std::vector<size_t>> m_byId1;
    std::string m_report;
};

}  // namespace arcadexr::texpack
