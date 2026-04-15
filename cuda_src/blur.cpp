// cuda_src/blur.cpp — Adrian's early prototype / streaming smoke-test.
// Updated to use float grayscale (single channel, row-major) aligned with
// the buffer contract in gpu_canny/include/canny_buffer_contract.hpp.
//
// Pipeline: read grayscale floats from file → pinned H2D → kernel → D2H → write results.
// Gaussian kernel internals belong to Kushaan's fused kernel; this file tests the
// stream / memcpy / launch infrastructure only.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define NUM_STREAMS 1

// Trivial kernel: normalize grayscale to [0, 1] and write through.
// Replace with Kushaan's fused blur+gradient+NMS+threshold when ready.
__global__ void grayscale_passthrough(const float* __restrict__ input, std::size_t pitch_floats,
                                      float* __restrict__ output, int width, int height) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= static_cast<unsigned>(width) || y >= static_cast<unsigned>(height)) {
        return;
    }
    output[y * pitch_floats + x] = input[y * pitch_floats + x];
}

// Read float grayscale values written by deconstruct_image.py (one float per line, [0,1]).
static bool read_grayscale_file(const std::string& filename, std::vector<float>& pixels,
                                 int width, int height) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open: " << filename << "\n";
        return false;
    }
    pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    for (auto& v : pixels) {
        if (!(f >> v)) {
            std::cerr << "Unexpected end of file in: " << filename << "\n";
            return false;
        }
    }
    return true;
}

static void write_grayscale_file(const std::string& filename, const float* pixels, int width,
                                  int height) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot write: " << filename << "\n";
        return;
    }
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            f << pixels[i * width + j] << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>\n";
        std::cerr << "  Reads out/0001_gray.txt (floats), runs pipeline, writes out/output_gray.txt\n";
        return 1;
    }

    const int img_width  = std::atoi(argv[1]);
    const int img_height = std::atoi(argv[2]);
    if (img_width <= 0 || img_height <= 0) {
        std::cerr << "Width and height must be positive.\n";
        return 1;
    }

    const std::size_t npix = static_cast<std::size_t>(img_width) * static_cast<std::size_t>(img_height);

    // --- Load grayscale float pixels from file ---
    std::vector<float> h_pixels;
    if (!read_grayscale_file("out/0001_gray.txt", h_pixels, img_width, img_height)) {
        return 1;
    }

    // --- Pinned host buffers ---
    float* h_input_pinned  = nullptr;
    float* h_output_pinned = nullptr;
    cudaHostAlloc(&h_input_pinned,  npix * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_output_pinned, npix * sizeof(float), cudaHostAllocDefault);
    std::copy(h_pixels.begin(), h_pixels.end(), h_input_pinned);

    // --- Pitched device buffers (row-major, aligned; matches canny_buffer_contract.hpp) ---
    float* d_input  = nullptr;
    float* d_output = nullptr;
    std::size_t pitch_bytes = 0;
    cudaMallocPitch(reinterpret_cast<void**>(&d_input),  &pitch_bytes,
                    static_cast<std::size_t>(img_width) * sizeof(float),
                    static_cast<std::size_t>(img_height));
    cudaMallocPitch(reinterpret_cast<void**>(&d_output), &pitch_bytes,
                    static_cast<std::size_t>(img_width) * sizeof(float),
                    static_cast<std::size_t>(img_height));
    const std::size_t pitch_floats = pitch_bytes / sizeof(float);

    // --- CUDA streams (non-blocking for overlap; PDF Sec. 3.3–3.4) ---
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // --- Async H2D ---
    cudaMemcpy2DAsync(d_input, pitch_bytes,
                      h_input_pinned, static_cast<std::size_t>(img_width) * sizeof(float),
                      static_cast<std::size_t>(img_width) * sizeof(float),
                      static_cast<std::size_t>(img_height),
                      cudaMemcpyHostToDevice, streams[0]);

    // --- Kernel launch ---
    const dim3 block(32, 32);  // 32x32 matches fused.cpp tile design.
    const dim3 grid((img_width  + 31) / 32, (img_height + 31) / 32);
    grayscale_passthrough<<<grid, block, 0, streams[0]>>>(
        d_input, pitch_floats, d_output, img_width, img_height);

    // --- Async D2H ---
    cudaMemcpy2DAsync(h_output_pinned, static_cast<std::size_t>(img_width) * sizeof(float),
                      d_output, pitch_bytes,
                      static_cast<std::size_t>(img_width) * sizeof(float),
                      static_cast<std::size_t>(img_height),
                      cudaMemcpyDeviceToHost, streams[0]);

    // --- Sync and clean up ---
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    std::cout << "Pipeline complete. Writing output.\n";
    write_grayscale_file("out/output_gray.txt", h_output_pinned, img_width, img_height);
    std::cout << "Done: out/output_gray.txt\n";

    cudaFreeHost(h_input_pinned);
    cudaFreeHost(h_output_pinned);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
