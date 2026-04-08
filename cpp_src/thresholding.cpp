#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <sys/types.h>
#include <vector>

const int WIDTH = 2000;
const int HEIGHT = 1000;

enum pixel_strengths_t { STRONG, WEAK, MID };

static const char *EnumStrings[] = {"strong", "weak", "mid"};

const char *getTextForEnum(int enumVal) { return EnumStrings[enumVal]; }

void run_thresholding(
    std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> &pixels,
    std::vector<std::tuple<int, int>> &strong_indices) {
  int accumulator = 0;

  const auto start{std::chrono::steady_clock::now()};
#pragma omp parallel num_threads(8)
  {
    int id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    int num_indices = strong_indices.size();
    std::queue<std::tuple<int, int>> curr_processing_indices;
    for (int i = num_indices / num_threads * id;
         i < std::max(num_indices, num_indices / num_threads * (id + 1)); i++) {
      // std::cout << "Pushing strong indices in thread " << id << std::endl;
      curr_processing_indices.push(strong_indices[i]);
    }
    while (!curr_processing_indices.empty()) {
      auto curr_pixel = curr_processing_indices.front();
      curr_processing_indices.pop();
      auto x = std::get<0>(curr_pixel);
      auto y = std::get<1>(curr_pixel);

      pixels[x][y] = STRONG;
      // printf("Setting pixel (%d, %d) to STRONG!\n", x, y);
      for (int j = std::max(0, x - 1); j < std::min(WIDTH, x + 2); j++) {
        for (int k = std::max(0, y - 1); k < std::min(HEIGHT, y + 2); k++) {
          // printf("Checking pixel (%d, %d)!\n", j, k);
          if (pixels[j][k] == MID) {
            // printf("Found pixel (%d, %d) to be MID!", j, k);
            curr_processing_indices.push(std::tuple<int, int>(j, k));
          }
        }
      }
    }
  }
  const auto finish{std::chrono::steady_clock::now()};
  const std::chrono::duration<double> elapsed_seconds{finish - start};
  std::cout << "Elapsed time for thresholding is: " << elapsed_seconds << "\n";
  return;
}

void initialize_pixels(
    std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> &pixels,
    std::vector<std::tuple<int, int>> &strong_indices) {
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      pixels[i][j] = WEAK;
    }
  }

  // For now, do a striped pattern.
  for (int i = 0; i < WIDTH; i += 100) {
    pixels[i][0] = STRONG;
    strong_indices.push_back(std::tuple<int, int>(i, 0));
    for (int j = 1; j < HEIGHT; j++) {
      pixels[i][j] = MID;
    }
  }

  return;
}

void write_image_to_file(
    std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> &pixels,
    std::string filename) {
  std::ofstream img(filename);
  if (img.is_open()) {
    for (int i = 0; i < WIDTH; i++) {
      for (int j = 0; j < HEIGHT; j++) {
        img << getTextForEnum(pixels[i][j]) << std::endl;
      }
    }
  }
}

int main() {
  std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> pixels;
  std::vector<std::tuple<int, int>> strong_indices;

  initialize_pixels(pixels, strong_indices);
  std::cout << "Currently writing pixels to file!\n";
  write_image_to_file(pixels, "img.txt");
  std::cout << "Finished writing pixels, running thresholding!\n";
  run_thresholding(pixels, strong_indices);
  std::cout << "Finished thresholding, writing pixels to file!\n";
  write_image_to_file(pixels, "thresholded.txt");
  std::cout << "Done writing pixels to file!\n";
  return 0;
}
