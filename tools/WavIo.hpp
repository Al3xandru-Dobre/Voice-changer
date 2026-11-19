// WavIo — minimal mono WAV reader/writer shared by the offline runners
// (fv_vc_file, fv_s2ut_file). Standard library only.
//
// read_mono_wav accepts mono 16-bit PCM or 32-bit float (also inside
// WAVE_FORMAT_EXTENSIBLE); write_mono_wav writes 32-bit float or 16-bit PCM.
// Both return false and set `error` instead of throwing on bad files.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace fluxvoice::tools {

inline std::uint16_t read_u16_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(p[0] | (std::uint16_t{p[1]} << 8));
}

inline std::uint32_t read_u32_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline bool read_mono_wav(const std::filesystem::path& path, std::uint32_t& sample_rate,
                   std::vector<float>& samples, std::string& error) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        error = "cannot open input WAV: " + path.string();
        return false;
    }
    const auto file_size = f.tellg();
    if (file_size < 44) {
        error = "input WAV is too short: " + path.string();
        return false;
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(file_size));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(buf.data()), file_size)) {
        error = "failed to read input WAV: " + path.string();
        return false;
    }
    if (std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        error = "input file is not a RIFF/WAVE file: " + path.string();
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;
    std::uint16_t format_tag = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits_per_sample = 0;
    const std::uint8_t* data_ptr = nullptr;
    std::size_t data_bytes = 0;

    std::size_t pos = 12;
    while (pos + 8 <= buf.size()) {
        const char* id = reinterpret_cast<const char*>(buf.data() + pos);
        const std::uint32_t chunk_size = read_u32_le(buf.data() + pos + 4);
        const std::size_t payload = pos + 8;
        if (payload + chunk_size > buf.size()) break;

        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                error = "malformed fmt chunk in " + path.string();
                return false;
            }
            format_tag = read_u16_le(buf.data() + payload);
            channels = read_u16_le(buf.data() + payload + 2);
            sample_rate = read_u32_le(buf.data() + payload + 4);
            bits_per_sample = read_u16_le(buf.data() + payload + 14);
            if (format_tag == 0xFFFEu && chunk_size >= 40) {
                format_tag = read_u16_le(buf.data() + payload + 24);
            }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_ptr = buf.data() + payload;
            data_bytes = chunk_size;
            have_data = true;
            break;
        }
        pos = payload + chunk_size + (chunk_size & 1u);
    }

    if (!have_fmt || !have_data) {
        error = "WAV file missing fmt or data chunk: " + path.string();
        return false;
    }
    if (channels != 1) {
        error = "WAV must be mono (got " + std::to_string(channels) + " channels)";
        return false;
    }

    if (format_tag == 1 && bits_per_sample == 16) {
        const std::size_t n = data_bytes / 2;
        samples.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto u = read_u16_le(data_ptr + 2 * i);
            const auto s = static_cast<std::int16_t>(u);
            samples[i] = static_cast<float>(s) / 32768.0f;
        }
        return true;
    }
    if (format_tag == 3 && bits_per_sample == 32) {
        const std::size_t n = data_bytes / 4;
        samples.resize(n);
        std::memcpy(samples.data(), data_ptr, n * sizeof(float));
        return true;
    }

    error = "unsupported WAV format (format_tag=" + std::to_string(format_tag) +
            ", bits_per_sample=" + std::to_string(bits_per_sample) +
            "; expected 16-bit PCM or 32-bit float)";
    return false;
}

inline bool write_mono_wav(const std::filesystem::path& path, std::uint32_t sample_rate,
                    std::span<const float> samples, bool pcm16, std::string& error) {
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "cannot open output WAV for writing: " + path.string();
        return false;
    }
    auto u16 = [&](std::uint32_t v) {
        f.put(static_cast<char>(v & 0xFFu));
        f.put(static_cast<char>((v >> 8) & 0xFFu));
    };
    auto u32 = [&](std::uint32_t v) {
        u16(v & 0xFFFFu);
        u16((v >> 16) & 0xFFFFu);
    };

    const std::uint16_t format_tag = pcm16 ? 1u : 3u;
    const std::uint16_t bytes_per_sample = pcm16 ? 2u : 4u;
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * bytes_per_sample);

    f.write("RIFF", 4);
    u32(36u + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    u32(16u);
    u16(format_tag);
    u16(1u); // mono
    u32(sample_rate);
    u32(sample_rate * bytes_per_sample);
    u16(bytes_per_sample);
    u16(bytes_per_sample * 8u);
    f.write("data", 4);
    u32(data_bytes);

    if (pcm16) {
        for (float s : samples) {
            const float clamped = std::clamp(s, -1.0f, 1.0f);
            const auto q = static_cast<std::int16_t>(std::lround(clamped * 32767.0f));
            u16(static_cast<std::uint16_t>(q));
        }
    } else {
        f.write(reinterpret_cast<const char*>(samples.data()),
                static_cast<std::streamsize>(data_bytes));
    }

    if (!f) {
        error = "failed writing output WAV: " + path.string();
        return false;
    }
    return true;
}

} // namespace fluxvoice::tools
