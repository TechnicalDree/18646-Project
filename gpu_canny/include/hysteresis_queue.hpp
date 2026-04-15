#pragma once

// Handoff from Adrian (GPU pipeline) to Sid (CPU hysteresis + work queue).
// After cudaEventSynchronize(gpu_done) or cudaStreamWaitEvent, the pinned buffer is safe to read.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

namespace canny {

struct HysteresisWorkItem {
    int width = 0;
    int height = 0;
    std::size_t row_pitch_bytes = 0;
    /// Pinned host memory: EdgeClass per pixel (see canny_buffer_contract.hpp).
    std::uint8_t* edge_map_host = nullptr;
    /// Recorded after the async D2H into edge_map_host completes on `stream`.
    cudaEvent_t gpu_done = nullptr;
    cudaStream_t stream = nullptr;
};

/// Thread-safe queue; Adrian pushes after scheduling D2H + event; Sid's pool pops and runs hysteresis.
class HysteresisWorkQueue {
public:
    void push(HysteresisWorkItem item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(item);
    }

    /// Non-blocking: returns nullopt if empty.
    std::optional<HysteresisWorkItem> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        HysteresisWorkItem w = queue_.front();
        queue_.pop_front();
        return w;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::deque<HysteresisWorkItem> queue_;
};

}  // namespace canny
