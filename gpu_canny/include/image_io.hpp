#pragma once

// Image loading for the GPU Canny pipeline.
// Supports JPEG, PNG, PPM (via stb_image) → float grayscale [0,1], row-major.
// Adrian's image frontend (PDF Sec. 4: "Image frontend, data preprocessing").

#include "ppm_io.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// stb_image: declaration only — implementation compiled in stb_image_impl.cpp.
#ifndef STBI_INCLUDE_STB_IMAGE_H
#include "../third_party/stb_image.h"
#endif

namespace canny::io {

/// Load any image format supported by stb_image (JPEG, PNG, BMP, TGA, PPM …)
/// and convert to single-channel float grayscale [0, 1] in row-major order.
/// Returns false on failure.
inline bool load_image_grayscale(const std::string& path, std::vector<float>& out, int& width,
                                  int& height) {
    // Fast path: native PPM parser (no stb decode overhead for synthetic tests).
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".ppm") {
        return load_ppm_grayscale(path, out, width, height);
    }

    int w = 0, h = 0, channels_ignored = 0;
    // Force 3 channels (RGB) so BT.601 luma works for any source format.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels_ignored, 3);
    if (!data) {
        return false;
    }

    width = w;
    height = h;
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    out.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        const float r = static_cast<float>(data[i * 3u]);
        const float g = static_cast<float>(data[i * 3u + 1u]);
        const float b = static_cast<float>(data[i * 3u + 2u]);
        // ITU-R BT.601 luma — same formula used in ppm_io.hpp.
        out[i] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
    }

    stbi_image_free(data);
    return true;
}

/// Collect all image paths with a given extension (e.g. ".jpg", ".png") from a directory.
/// Returns sorted vector of absolute paths. Requires C++17 filesystem (set in CMakeLists.txt).
inline std::vector<std::string> list_images(const std::string& dir, const std::string& ext) {
    std::vector<std::string> paths;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        const std::string p = entry.path().string();
        if (p.size() >= ext.size() && p.substr(p.size() - ext.size()) == ext) {
            paths.push_back(p);
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

}  // namespace canny::io
