#include <iostream>
#include <array>
#include <omp.h>
#include <string> 
#include <cstdlib>
#include <vector>
#include <fstream>

#define NUM_IMAGES  1
#define NUM_STREAMS 1 

using namespace std;

__global__ void blur(int *img_arr, int width, int height, int *output_buff) {   
    int r, g, b;
    int pixel_value;

    pixel_value = 0;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            r = (img_arr[i * width + j] & 0xFF0000) >> 16;
            g = (img_arr[i * width + j] & 0xFF00) >> 8;
            b = (img_arr[i * width + j] & 0xFF);

            // Dummy operation to prove rest of code works.
            b = (b + 20) & 0xFF;
            pixel_value = r << 16 | g << 8 | b; 
            output_buff[i * width + j] = pixel_value;
            // printf("Pixel: %d\n", output_buff[i * width + j]);
        }
    }
}

void write_image_to_file(std::string filename, int* pixel_arr, int width, int height) {
  std::ofstream img(filename);
  if (img.is_open()) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        std::cout << i * width + j << std::endl;
        img << pixel_arr[i * width + j] << "|";
      }
    }
  }
}

int main(int argc, char *argv[]) {
    int *dev_buff, *conv_buff, *output_buff;

    if (argc != 3) {
        std::cout << "Number of arguments incorrect, try again.\n";
        exit(-1);
    }

    int img_width = atoi(argv[1]);
    int img_height = atoi(argv[2]);

    // Set up streams for concurrent image blurring.
    cudaStream_t streams[NUM_STREAMS];
    
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    
    std::vector<int> pixels;
    std::array<std::string, 1> imgs{ "out/0001.txt" };
    std::string line;

    int* arr;

    for (int i = 0; i < imgs.size(); i++) {
        std::cout << "Opening and reading img: " << i << std::endl;
        std::ifstream img(imgs[i]);
        if (img.is_open()) {
            for (int i = 0; i < img_height; i++) {
                for (int j = 0; j < img_width; j++) {
                    std::getline(img, line);
                    pixels.push_back(stoi(line));
                }
            }
            std::cout << "Done loading pixel data, starting blurring!" << std::endl;
            arr = pixels.data();

            cudaMalloc(&dev_buff, sizeof(int) * img_width * img_height);
            cudaMalloc(&conv_buff, sizeof(int) * img_width * img_height);
            output_buff = (int *)malloc(sizeof(int) * img_width * img_height);

            cudaMemcpyAsync(dev_buff, arr, sizeof(int) * img_width * img_height,
                cudaMemcpyHostToDevice, streams[0]);

            blur<<<4, 256, 0, streams[0]>>>(dev_buff, img_width, img_height, conv_buff);

            cudaMemcpyAsync(output_buff, conv_buff, sizeof(int) * img_width * img_height,
                cudaMemcpyDeviceToHost, streams[0]);
        }
        img.close();
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    std::cout << "Done destroying streams!" << std::endl;

    cudaDeviceSynchronize();

    std::cout << "Done blurring, starting image writing!" << std::endl;
    write_image_to_file("out/modified_img.txt", output_buff, img_width, img_height);
    std::cout << "Done writing the image!" << std::endl;

    return 0;
}
