#pragma once

// Buffer layout contract: GPU fused stages (through double thresholding) -> CPU hysteresis.
// Aligned with 18-646 project PDF: row-major storage, explicit pitches, async D2H before CPU graph work.

#include <cstddef>
#include <cstdint>

namespace canny::contract {

/// Pixel classification written by fused GPU stages (before CPU hysteresis).
/// Values match common Canny hysteresis labeling.
enum class EdgeClass : std::uint8_t {
    kNonEdge = 0,
    kWeak = 1,
    kStrong = 2,
};

/// Host-side grayscale input to the fused pipeline: single channel float32, row-major.
struct GrayscaleHostView {
    const float* data = nullptr;
    int width = 0;
    int height = 0;
    /// Bytes between rows (may equal width * sizeof(float) for tight packing).
    std::size_t row_pitch_bytes = 0;
};

/// Device grayscale: pitched allocation from cudaMallocPitch.
struct GrayscaleDeviceView {
    float* ptr = nullptr;
    std::size_t pitch_bytes = 0;
    int width = 0;
    int height = 0;
};

/// Device output: per-pixel EdgeClass, pitched.
struct EdgeMapDeviceView {
    std::uint8_t* ptr = nullptr;
    std::size_t pitch_bytes = 0;
    int width = 0;
    int height = 0;
};

/// Pinned host buffer for async D2H of edge map (Sid / hysteresis reads this after event fires).
struct EdgeMapHostView {
    std::uint8_t* data = nullptr;
    std::size_t row_pitch_bytes = 0;
    int width = 0;
    int height = 0;
};

inline std::size_t grayscale_row_pitch_default(int width) {
    return static_cast<std::size_t>(width) * sizeof(float);
}

inline std::size_t edge_row_pitch_default(int width) {
    return static_cast<std::size_t>(width) * sizeof(std::uint8_t);
}

}  // namespace canny::contract
