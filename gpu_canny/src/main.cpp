#include "canny_buffer_contract.hpp"
#include "canny_pipeline.hpp"
#include "image_io.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options] [image]\n"
                 "\n"
                 "Options:\n"
                 "  --multi N          Use N concurrent streams (default: 1).\n"
                 "  --dir-small <dir>  Batch: load all .jpg from dir (BSDS500 style).\n"
                 "  --dir-large <dir>  Batch: load all .png from dir (Div2K style).\n"
                 "  --weak  <f>        Weak threshold  [0,1]  (default: 0.25).\n"
                 "  --strong <f>       Strong threshold [0,1] (default: 0.60).\n"
                 "\n"
                 "Without image/dir: runs a synthetic 64x48 gradient through the pipeline.\n",
                 argv0);
}

/// Minimal stand-in for Sid's CPU hysteresis: waits for GPU+D2H then counts edge classes.
/// Sid replaces this with BFS propagation (PDF Sec. 3.3 / cpp_src/thresholding.cpp).
void consume_work_queue(canny::HysteresisWorkQueue& queue) {
    for (;;) {
        auto item = queue.try_pop();
        if (!item) {
            break;
        }
        const cudaError_t err = cudaEventSynchronize(item->gpu_done);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaEventSynchronize: %s\n", cudaGetErrorString(err));
            continue;
        }
        std::size_t strong = 0, weak = 0;
        for (int y = 0; y < item->height; ++y) {
            const std::uint8_t* row = item->edge_map_host + y * item->row_pitch_bytes;
            for (int x = 0; x < item->width; ++x) {
                if (row[x] == static_cast<std::uint8_t>(canny::contract::EdgeClass::kStrong)) {
                    ++strong;
                } else if (row[x] == static_cast<std::uint8_t>(canny::contract::EdgeClass::kWeak)) {
                    ++weak;
                }
            }
        }
        std::printf("hysteresis_stub: strong=%zu weak=%zu (%dx%d)\n", strong, weak, item->width,
                    item->height);
    }
}

/// Process a batch of image paths with multi-stream dispatch (PDF Sec. 3.3–3.4).
/// Strategy (from PDF): one image → one stream slot.  When we have more images than slots, we
/// reuse slots in round-robin — but only after syncing on that slot's completion event so we
/// never overwrite its pinned buffer while a previous D2H is still in flight.
/// This lets H2D(i+1) overlap with kernel(i) and D2H(i-1) across independent streams.
double run_batch(const std::vector<std::string>& paths, int num_streams, float weak_thr,
                 float strong_thr) {
    if (paths.empty()) {
        return 0.0;
    }

    canny::MultiStreamCannyDispatcher dispatch(num_streams);
    canny::HysteresisWorkQueue queue;

    // Track how many times each slot has been used (to know when a sync is needed on reuse).
    std::vector<int> slot_use_count(static_cast<std::size_t>(num_streams), 0);

    double total_submit_ms = 0.0;
    std::size_t n_processed = 0;

    const auto wall_t0 = std::chrono::steady_clock::now();

    for (const auto& path : paths) {
        std::vector<float> gray;
        int width = 0, height = 0;

        if (!canny::io::load_image_grayscale(path, gray, width, height)) {
            std::fprintf(stderr, "Skipping (load failed): %s\n", path.c_str());
            continue;
        }

        const int slot = dispatch.next_slot();
        canny::GpuCannyOrchestrator& orch = dispatch.slot(slot);

        // Sync before reusing a slot so we don't clobber its pinned buffer.
        if (slot_use_count[static_cast<std::size_t>(slot)] > 0) {
            cudaEventSynchronize(orch.completion_event());
        }

        cudaError_t err = orch.resize(width, height);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "resize %dx%d: %s\n", width, height, cudaGetErrorString(err));
            continue;
        }

        const auto t0 = std::chrono::steady_clock::now();

        canny::contract::GrayscaleHostView view{gray.data(), width, height,
                                                 canny::contract::grayscale_row_pitch_default(width)};

        // CPU→pinned copy (host-side, cheap), then async H2D+kernel+D2H+event on GPU stream.
        orch.copy_grayscale_to_pinned_input(view);
        err = orch.run_from_pinned_async(weak_thr, strong_thr);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "run_from_pinned_async: %s\n", cudaGetErrorString(err));
            continue;
        }
        orch.enqueue_for_hysteresis(queue);

        ++slot_use_count[static_cast<std::size_t>(slot)];

        const auto t1 = std::chrono::steady_clock::now();
        const double submit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_submit_ms += submit_ms;
        ++n_processed;

        std::printf("[%zu/%zu] submitted  slot=%d  %dx%d  %.3f ms  %s\n", n_processed, paths.size(),
                    slot, width, height, submit_ms, path.c_str());
    }

    // Final drain: wait for all in-flight work then hand off to Sid's hysteresis.
    cudaDeviceSynchronize();
    consume_work_queue(queue);

    const auto wall_t1 = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    const double mean_ms =
        n_processed > 0 ? total_submit_ms / static_cast<double>(n_processed) : 0.0;

    std::printf("\nBatch: %zu images  mean_submit=%.3f ms  wall=%.1f ms  streams=%d\n", n_processed,
                mean_ms, wall_ms, num_streams);
    return mean_ms;
}

}  // namespace

int main(int argc, char** argv) {
    int multi = 1;
    float weak_thr = 0.25f;
    float strong_thr = 0.60f;
    const char* ppm_path = nullptr;
    std::string dir_small, dir_large;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--multi") == 0 && i + 1 < argc) {
            multi = std::atoi(argv[++i]);
            if (multi < 1) multi = 1;
        } else if (std::strcmp(argv[i], "--dir-small") == 0 && i + 1 < argc) {
            dir_small = argv[++i];
        } else if (std::strcmp(argv[i], "--dir-large") == 0 && i + 1 < argc) {
            dir_large = argv[++i];
        } else if (std::strcmp(argv[i], "--weak") == 0 && i + 1 < argc) {
            weak_thr = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--strong") == 0 && i + 1 < argc) {
            strong_thr = std::atof(argv[++i]);
        } else if (argv[i][0] != '-') {
            ppm_path = argv[i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---- Batch directory modes (BSDS500 .jpg / Div2K .png) ----
    if (!dir_small.empty() || !dir_large.empty()) {
        if (!dir_small.empty()) {
            std::printf("=== Small images (BSDS500, .jpg): %s  streams=%d ===\n",
                        dir_small.c_str(), multi);
            const auto paths = canny::io::list_images(dir_small, ".jpg");
            run_batch(paths, multi, weak_thr, strong_thr);
        }
        if (!dir_large.empty()) {
            std::printf("=== Large images (Div2K, .png): %s  streams=%d ===\n",
                        dir_large.c_str(), multi);
            const auto paths = canny::io::list_images(dir_large, ".png");
            run_batch(paths, multi, weak_thr, strong_thr);
        }
        return 0;
    }

    // ---- Single image or synthetic mode ----
    std::vector<float> gray;
    int width = 0, height = 0;

    if (ppm_path) {
        if (!canny::io::load_image_grayscale(ppm_path, gray, width, height)) {
            std::fprintf(stderr, "Failed to load image: %s\n", ppm_path);
            return 1;
        }
    } else {
        // Synthetic 64×48 gradient for smoke testing without any image file.
        width = 64;
        height = 48;
        gray.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                gray[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(x)] =
                    static_cast<float>(x + y) / static_cast<float>(width + height);
            }
        }
        std::printf("(No image supplied — using %dx%d synthetic gradient)\n", width, height);
    }

    canny::HysteresisWorkQueue queue;
    canny::contract::GrayscaleHostView view{gray.data(), width, height,
                                             canny::contract::grayscale_row_pitch_default(width)};

    if (multi == 1) {
        // Single-stream path.
        canny::GpuCannyOrchestrator orch;
        cudaError_t err = orch.resize(width, height);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "resize: %s\n", cudaGetErrorString(err));
            return 1;
        }
        err = orch.run_async(view, weak_thr, strong_thr);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "run_async: %s\n", cudaGetErrorString(err));
            return 1;
        }
        orch.enqueue_for_hysteresis(queue);
        consume_work_queue(queue);
        return 0;
    }

    // Multi-stream path: round-robin over N slots (PDF: one stream per image).
    canny::MultiStreamCannyDispatcher dispatch(multi);
    for (int s = 0; s < dispatch.num_slots(); ++s) {
        cudaError_t err = dispatch.slot(s).resize(width, height);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "resize slot %d: %s\n", s, cudaGetErrorString(err));
            return 1;
        }
    }

    for (int i = 0; i < multi; ++i) {
        const int s = dispatch.next_slot();
        canny::GpuCannyOrchestrator& orch = dispatch.slot(s);
        orch.copy_grayscale_to_pinned_input(view);
        const float wt = weak_thr + 0.05f * static_cast<float>(i);
        const float st = strong_thr + 0.05f * static_cast<float>(i);
        const cudaError_t err = orch.run_from_pinned_async(wt, st);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "run_from_pinned_async slot %d: %s\n", s, cudaGetErrorString(err));
            return 1;
        }
        orch.enqueue_for_hysteresis(queue);
    }
    consume_work_queue(queue);
    return 0;
}
