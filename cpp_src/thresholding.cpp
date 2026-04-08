#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
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
#pragma omp parallel num_threads(4)
  {
  }
  return;
}

void initialize_pixels(
    std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> &pixels) {
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < HEIGHT; j++) {
      pixels[i][j] = WEAK;
    }
  }

  // For now, do a striped pattern.
  for (int i = 0; i < WIDTH; i += 100) {
    pixels[i][0] = STRONG;
    for (int j = 1; j < HEIGHT; j++) {
      pixels[i][j] = MID;
    }
  }

  return;
}

void write_image_to_file(
    std::array<std::array<pixel_strengths_t, HEIGHT>, WIDTH> &pixels) {
  std::ofstream img("img.txt");
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
  std::vector<std::tuple<int, int>> indices;

  initialize_pixels(pixels);
  write_image_to_file(pixels);
  run_thresholding(pixels, indices);
  return 0;
}
