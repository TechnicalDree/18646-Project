#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace canny::io {

/// Load binary PPM (P6) RGB, convert to float grayscale [0,1]. Returns false on parse error.
inline bool load_ppm_grayscale(const std::string& path, std::vector<float>& out, int& width, int& height) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    char magic[3] = {0};
    if (std::fscanf(f, "%2s", magic) != 1 || std::strcmp(magic, "P6") != 0) {
        std::fclose(f);
        return false;
    }
    int w = 0, h = 0, maxv = 0;
    if (std::fscanf(f, "%d %d %d", &w, &h, &maxv) != 3 || w <= 0 || h <= 0 || maxv > 255) {
        std::fclose(f);
        return false;
    }
    std::fgetc(f);  // consume single whitespace after maxv
    const std::size_t nbytes = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3u;
    std::vector<unsigned char> rgb(nbytes);
    if (std::fread(rgb.data(), 1, nbytes, f) != nbytes) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    width = w;
    height = h;
    out.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    for (std::size_t i = 0; i < out.size(); ++i) {
        const unsigned r = rgb[i * 3u];
        const unsigned g = rgb[i * 3u + 1u];
        const unsigned b = rgb[i * 3u + 2u];
        // ITU-R BT.601 luma
        out[i] = (0.299f * static_cast<float>(r) + 0.587f * static_cast<float>(g) +
                  0.114f * static_cast<float>(b)) /
                 static_cast<float>(maxv);
    }
    return true;
}

}  // namespace canny::io
