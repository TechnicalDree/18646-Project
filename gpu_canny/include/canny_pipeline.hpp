#pragma once

#include "canny_buffer_contract.hpp"
#include "hysteresis_queue.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace canny {

/// Launch parameters for the fused kernel (Kushaan owns internals; Adrian owns grid/block/stream wiring).
/// Block size 32x32 matches Kushaan's fused tile design (PDF Sec. 3.2: 32x32 to hit 1024 thread limit).
/// grid_dim is auto-computed from image size when left zero.
struct FusedLaunchConfig {
    dim3 block_dim{32, 32, 1};
    /// Filled from image size in orchestrator if zero.
    dim3 grid_dim{0, 0, 0};
    /// Set by Kush when real fused kernel lands; stub requires 0.
    std::size_t shared_mem_bytes = 0;
};

/// Single-stream path: one image at a time, async H2D / kernel / D2H + event for CPU handoff.
class GpuCannyOrchestrator {
public:
    GpuCannyOrchestrator() = default;
    GpuCannyOrchestrator(const GpuCannyOrchestrator&) = delete;
    GpuCannyOrchestrator& operator=(const GpuCannyOrchestrator&) = delete;

    ~GpuCannyOrchestrator();

    /// Allocates device + pinned host buffers for given dimensions.
    cudaError_t resize(int width, int height);

    /// Copies grayscale host view to device (async), launches fused stub, D2H edge map, records event.
    /// `weak_thr` / `strong_thr` are placeholders until full fused Canny exists.
    cudaError_t run_async(const contract::GrayscaleHostView& host_gray, float weak_thr, float strong_thr);

    /// CPU memcpy into this slot's pinned grayscale buffer (call before run_from_pinned_async for multi-stream).
    void copy_grayscale_to_pinned_input(const contract::GrayscaleHostView& host_gray);

    /// H2D from pinned gray, fused kernel, D2H edge, record completion event (safe for concurrent slots).
    cudaError_t run_from_pinned_async(float weak_thr, float strong_thr);

    cudaStream_t stream() const { return stream_; }
    cudaEvent_t completion_event() const { return done_event_; }

    const contract::EdgeMapHostView& host_edge_map() const { return host_edge_view_; }

    /// Enqueue handoff to Sid after run_async (does not sync GPU).
    void enqueue_for_hysteresis(HysteresisWorkQueue& queue);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_event_ = nullptr;

    float* d_gray_ = nullptr;
    std::size_t pitch_gray_ = 0;

    std::uint8_t* d_edge_ = nullptr;
    std::size_t pitch_edge_ = 0;

    float* h_gray_pinned_ = nullptr;
    std::uint8_t* h_edge_pinned_ = nullptr;

    contract::EdgeMapHostView host_edge_view_{};

    FusedLaunchConfig launch_cfg_{};

    cudaError_t free_device_and_host_buffers();
};

/// One CUDA stream per image slot (PDF: input-level concurrency, keep GPU fed).
class MultiStreamCannyDispatcher {
public:
    explicit MultiStreamCannyDispatcher(int num_streams);
    ~MultiStreamCannyDispatcher();

    MultiStreamCannyDispatcher(const MultiStreamCannyDispatcher&) = delete;
    MultiStreamCannyDispatcher& operator=(const MultiStreamCannyDispatcher&) = delete;

    int num_slots() const { return static_cast<int>(orchestrators_.size()); }

    GpuCannyOrchestrator& slot(int i) { return *orchestrators_[static_cast<std::size_t>(i)]; }
    const GpuCannyOrchestrator& slot(int i) const { return *orchestrators_[static_cast<std::size_t>(i)]; }

    /// Round-robin slot index for next submission.
    int next_slot();

private:
    std::vector<std::unique_ptr<GpuCannyOrchestrator>> orchestrators_;
    int rr_ = 0;
};

// Implemented in fused_kernel.cu — replace body when real fused Canny lands.
cudaError_t launch_fused_canny_stages(cudaStream_t stream, const FusedLaunchConfig& cfg,
                                      const contract::GrayscaleDeviceView& gray,
                                      contract::EdgeMapDeviceView& edge_out, float weak_thr,
                                      float strong_thr);

}  // namespace canny
