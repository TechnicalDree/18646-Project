#include "canny_pipeline.hpp"

#include <cstring>

namespace canny {

cudaError_t GpuCannyOrchestrator::free_device_and_host_buffers() {
    cudaError_t err = cudaSuccess;
    if (d_gray_) {
        err = cudaFree(d_gray_);
        d_gray_ = nullptr;
    }
    if (d_edge_) {
        cudaError_t e = cudaFree(d_edge_);
        d_edge_ = nullptr;
        if (err == cudaSuccess) {
            err = e;
        }
    }
    if (h_gray_pinned_) {
        cudaError_t e = cudaFreeHost(h_gray_pinned_);
        h_gray_pinned_ = nullptr;
        if (err == cudaSuccess) {
            err = e;
        }
    }
    if (h_edge_pinned_) {
        cudaError_t e = cudaFreeHost(h_edge_pinned_);
        h_edge_pinned_ = nullptr;
        if (err == cudaSuccess) {
            err = e;
        }
    }
    return err;
}

GpuCannyOrchestrator::~GpuCannyOrchestrator() {
    free_device_and_host_buffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (done_event_) {
        cudaEventDestroy(done_event_);
        done_event_ = nullptr;
    }
}

cudaError_t GpuCannyOrchestrator::resize(int width, int height) {
    if (width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    if (width == width_ && height == height_ && d_gray_ != nullptr) {
        return cudaSuccess;
    }

    cudaError_t err = free_device_and_host_buffers();
    if (err != cudaSuccess) {
        return err;
    }

    width_ = width;
    height_ = height;

    if (!stream_) {
        // Non-blocking: allows true H2D/kernel/D2H overlap across independent streams
        // (PDF Sec. 3.3: keep the GPU fed with per-image streams).
        err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            return err;
        }
    }
    if (!done_event_) {
        err = cudaEventCreate(&done_event_);
        if (err != cudaSuccess) {
            return err;
        }
    }

    err = cudaMallocPitch(reinterpret_cast<void**>(&d_gray_), &pitch_gray_,
                          static_cast<std::size_t>(width_) * sizeof(float), static_cast<std::size_t>(height_));
    if (err != cudaSuccess) {
        return err;
    }

    err =
        cudaMallocPitch(reinterpret_cast<void**>(&d_edge_), &pitch_edge_, static_cast<std::size_t>(width_),
                        static_cast<std::size_t>(height_));
    if (err != cudaSuccess) {
        return err;
    }

    const std::size_t gray_bytes = pitch_gray_ * static_cast<std::size_t>(height_);
    const std::size_t edge_bytes = pitch_edge_ * static_cast<std::size_t>(height_);

    err = cudaHostAlloc(reinterpret_cast<void**>(&h_gray_pinned_), gray_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaHostAlloc(reinterpret_cast<void**>(&h_edge_pinned_), edge_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        return err;
    }

    host_edge_view_.data = h_edge_pinned_;
    host_edge_view_.width = width_;
    host_edge_view_.height = height_;
    host_edge_view_.row_pitch_bytes = pitch_edge_;

    return cudaSuccess;
}

void GpuCannyOrchestrator::copy_grayscale_to_pinned_input(const contract::GrayscaleHostView& host_gray) {
    if (!h_gray_pinned_ || host_gray.width != width_ || host_gray.height != height_ || !host_gray.data) {
        return;
    }
    for (int y = 0; y < height_; ++y) {
        const float* src = reinterpret_cast<const float*>(reinterpret_cast<const char*>(host_gray.data) +
                                                           static_cast<std::size_t>(y) * host_gray.row_pitch_bytes);
        float* dst = reinterpret_cast<float*>(reinterpret_cast<char*>(h_gray_pinned_) +
                                                static_cast<std::size_t>(y) * pitch_gray_);
        std::memcpy(dst, src, static_cast<std::size_t>(width_) * sizeof(float));
    }
}

cudaError_t GpuCannyOrchestrator::run_from_pinned_async(float weak_thr, float strong_thr) {
    if (!d_gray_ || !h_gray_pinned_) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err =
        cudaMemcpy2DAsync(d_gray_, pitch_gray_, h_gray_pinned_, pitch_gray_,
                          static_cast<std::size_t>(width_) * sizeof(float), static_cast<std::size_t>(height_),
                          cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        return err;
    }

    contract::GrayscaleDeviceView g{d_gray_, pitch_gray_, width_, height_};
    contract::EdgeMapDeviceView e{d_edge_, pitch_edge_, width_, height_};

    err = launch_fused_canny_stages(stream_, launch_cfg_, g, e, weak_thr, strong_thr);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemcpy2DAsync(h_edge_pinned_, pitch_edge_, d_edge_, pitch_edge_, static_cast<std::size_t>(width_),
                            static_cast<std::size_t>(height_), cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        return err;
    }

    return cudaEventRecord(done_event_, stream_);
}

cudaError_t GpuCannyOrchestrator::run_async(const contract::GrayscaleHostView& host_gray, float weak_thr,
                                            float strong_thr) {
    if (!d_gray_ || host_gray.width != width_ || host_gray.height != height_) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err =
        cudaMemcpy2DAsync(d_gray_, pitch_gray_, host_gray.data, host_gray.row_pitch_bytes,
                          static_cast<std::size_t>(width_) * sizeof(float), static_cast<std::size_t>(height_),
                          cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        return err;
    }

    contract::GrayscaleDeviceView g{d_gray_, pitch_gray_, width_, height_};
    contract::EdgeMapDeviceView e{d_edge_, pitch_edge_, width_, height_};

    err = launch_fused_canny_stages(stream_, launch_cfg_, g, e, weak_thr, strong_thr);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMemcpy2DAsync(h_edge_pinned_, pitch_edge_, d_edge_, pitch_edge_, static_cast<std::size_t>(width_),
                            static_cast<std::size_t>(height_), cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        return err;
    }

    return cudaEventRecord(done_event_, stream_);
}

void GpuCannyOrchestrator::enqueue_for_hysteresis(HysteresisWorkQueue& queue) {
    HysteresisWorkItem item;
    item.width = width_;
    item.height = height_;
    item.row_pitch_bytes = pitch_edge_;
    item.edge_map_host = h_edge_pinned_;
    item.gpu_done = done_event_;
    item.stream = stream_;
    queue.push(item);
}

MultiStreamCannyDispatcher::MultiStreamCannyDispatcher(int num_streams) {
    orchestrators_.reserve(static_cast<std::size_t>(std::max(1, num_streams)));
    for (int i = 0; i < num_streams; ++i) {
        orchestrators_.push_back(std::make_unique<GpuCannyOrchestrator>());
    }
}

MultiStreamCannyDispatcher::~MultiStreamCannyDispatcher() = default;

int MultiStreamCannyDispatcher::next_slot() {
    const int n = static_cast<int>(orchestrators_.size());
    if (n == 0) {
        return 0;
    }
    const int s = rr_;
    rr_ = (rr_ + 1) % n;
    return s;
}

}  // namespace canny
