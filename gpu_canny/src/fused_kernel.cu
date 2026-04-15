#include <cuda_runtime.h>

#include <cstdint>

#include "canny_buffer_contract.hpp"
#include "canny_pipeline.hpp"

namespace {

using canny::contract::EdgeClass;

/// Placeholder fused pipeline: double-threshold style labeling on raw grayscale.
/// Kushaan replaces this with the real fused blur + Sobel + NMS + threshold kernel.
__global__ void fused_canny_stages_stub_kernel(const float* __restrict__ gray,
                                                 std::size_t pitch_gray_bytes,
                                                 std::uint8_t* __restrict__ edge,
                                                 std::size_t pitch_edge_bytes, int width, int height,
                                                 float weak_thr, float strong_thr) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<unsigned>(width) || y >= static_cast<unsigned>(height)) {
        return;
    }
    const float* row_gray =
        reinterpret_cast<const float*>(reinterpret_cast<const char*>(gray) + y * pitch_gray_bytes);
    float v = row_gray[x];

    std::uint8_t c = static_cast<std::uint8_t>(EdgeClass::kNonEdge);
    if (v >= strong_thr) {
        c = static_cast<std::uint8_t>(EdgeClass::kStrong);
    } else if (v >= weak_thr) {
        c = static_cast<std::uint8_t>(EdgeClass::kWeak);
    }
    std::uint8_t* row_edge =
        reinterpret_cast<std::uint8_t*>(reinterpret_cast<char*>(edge) + y * pitch_edge_bytes);
    row_edge[x] = c;
}

}  // namespace

namespace canny {

cudaError_t launch_fused_canny_stages(cudaStream_t stream, const FusedLaunchConfig& cfg,
                                      const contract::GrayscaleDeviceView& gray,
                                      contract::EdgeMapDeviceView& edge_out, float weak_thr,
                                      float strong_thr) {
    dim3 block = cfg.block_dim;
    if (block.x == 0) {
        block.x = 16;
    }
    if (block.y == 0) {
        block.y = 16;
    }
    dim3 grid = cfg.grid_dim;
    if (grid.x == 0 || grid.y == 0) {
        grid.x = (gray.width + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
        grid.y = (gray.height + static_cast<int>(block.y) - 1) / static_cast<int>(block.y);
    }

    fused_canny_stages_stub_kernel<<<grid, block, cfg.shared_mem_bytes, stream>>>(
        gray.ptr, gray.pitch_bytes, edge_out.ptr, edge_out.pitch_bytes, gray.width, gray.height, weak_thr,
        strong_thr);
    return cudaGetLastError();
}

}  // namespace canny
