#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sys/types.h>
#include <vector>

using namespace cv;

Mat src, src_gray;
Mat dst, detected_edges;

int lowThreshold = 0;
const int NUM_LARGE_IMGS = 200;
const int NUM_SMALL_IMGS = 200;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char *window_name = "Edge Map";
double large_img_average = 0;
double small_img_average = 0;

enum img_size { SMALL, LARGE };

static void CannyThreshold(int, void *, img_size size) {
  blur(src_gray, detected_edges, Size(3, 3));

  const auto start{std::chrono::steady_clock::now()};
  Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio,
        kernel_size);
  const auto finish{std::chrono::steady_clock::now()};

  dst = Scalar::all(0);

  src.copyTo(dst, detected_edges);

  imshow(window_name, dst);

  const std::chrono::duration<double> elapsed_seconds{finish - start};
  std::cout << elapsed_seconds << "\n";
  switch (size) {
  case LARGE: {
    large_img_average += elapsed_seconds.count();
    break;
  }
  case SMALL: {
    small_img_average += elapsed_seconds.count();
    break;
  }
  }
}

void run_canny_imgs(img_size size, std::string dir_name) {
  for (const auto &entry : std::filesystem::directory_iterator(dir_name)) {
    src = imread(entry.path(), IMREAD_COLOR); // Load an image
    if (src.empty()) {
      std::cout << "Could not open or find the image!\n" << std::endl;
      std::cout << "Usage: " << entry.path() << " <Input image>" << std::endl;
      return;
    }

    dst.create(src.size(), src.type());

    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    CannyThreshold(0, 0, size);
  }
}

int main(int argc, char **argv) {

  std::string large_img_dir_name = "2k_img/";
  std::string small_img_dir_name = "berkeley_img/";

  std::cout << "Edge detection on large images:\n";
  run_canny_imgs(LARGE, large_img_dir_name);
  std::cout << "Edge detection on small images:\n";
  run_canny_imgs(SMALL, small_img_dir_name);

  std::cout << "Average Elapsed Time for Large Images: "
            << large_img_average / NUM_LARGE_IMGS << std::endl;

  std::cout << "Average Elapsed Time for Small Images: "
            << small_img_average / NUM_SMALL_IMGS << std::endl;

  return 0;
}
