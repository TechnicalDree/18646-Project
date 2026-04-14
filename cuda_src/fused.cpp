#include <iostream>
#include <array>
#include <omp.h>
#include <string> 
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cuda_fp16.h>

#define NUM_IMAGES  1
#define NUM_STREAMS 1

#define GAUSSIAN_K 5
#define SOBEL_K 3
#define SUPERTILE_X 100
#define SUPERTILE_Y 100
#define TILE_X 92
#define TILE_Y 92
#define PADDING 4

#define UPPER_THRESHOLD = 1
#define LOWER_THRESHOLD = 0
#define PI 3.1415
using namespace std;

__constant__ float gaussian_kernel_d[GAUSSIAN_K][GAUSSIAN_K];
__constant__ float sobel_kernel_d[SOBEL_K][SOBEL_K];
/*
| - x- 
y
|
*/

inline int flat_idx(int x, int y, int width) {
    return y * width + x;
}

inline int angle_direction(half angle) {
    return 0;
}

__global__ void fused(int *img_arr, int width, int height, int *output_buff) {
    __shared__ int buf1[SUPERTILE_X][SUPERTILE_Y];
    __shared__ float buf2[SUPERTILE_X][SUPERTILE_Y];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int img_x_start = bx * TILE_X;
    int img_y_start = by * TILE_Y;

    // Load from the image into shared memory buf1
    for(int y = (img_y_start - PADDING) + ty; y < (img_y_start + TILE_Y + PADDING); y += blockDim.y){
        for(int x = (img_x_start - PADDING) + tx; x < (img_x_start + TILE_X + PADDING); x += blockDim.x){
            if(x >= 0 && x < width && y >= 0 && y < height){
                buf1[PADDING + y - img_y_start][PADDING + x - img_x_start] = img_arr[flat_idx(x, y, width)];
            }
            else {
                buf1[PADDING + y - img_y_start][PADDING + x - img_x_start] = 0;
            }
        }
    }


    __syncthreads();
    // gaussian blur from buf1 to buf2
    for(int y = PADDING - 2 + ty; y < (TILE_Y + PADDING + 2); y += blockDim.y){
        for(int x = PADDING - 2 + tx; x < (TILE_X + PADDING + 2); x += blockDim.x){
            float sum = 0.0f;
            #pragma unroll
            for(int ky = -GAUSSIAN_K/2; ky <= GAUSSIAN_K/2; ky++){
                #pragma unroll
                for(int kx = -GAUSSIAN_K/2; kx <= GAUSSIAN_K/2; kx++){
                    sum += gaussian_kernel_d[ky + GAUSSIAN_K/2][kx + GAUSSIAN_K/2] * buf1[y + ky][x + kx];
                }
            }
            buf2[y][x] = sum;
        }
    }

    __syncthreads();

    // sobel from buf2 to buf1
    //change buf1 to 2 buffers
    half (*g_mag)[TILE_X+2] = reinterpret_cast<half (*)[TILE_X+2]>(buf1);
    half (*g_angle)[TILE_X+2] = g_mag + TILE_Y + 2;
    for(int y = PADDING - 1 + ty; y < TILE_Y + PADDING + 1; y += blockDim.y){
        for(int x = PADDING - 1 + tx; x < TILE_X + PADDING + 1; x += blockDim.x){
            float sum_x = 0.0f; float sum_y = 0.0f;
            #pragma unroll
            for(int ky = -SOBEL_K/2; ky <= SOBEL_K/2; ky++){
                #pragma unroll 
                for(int kx = -SOBEL_K/2; kx <= SOBEL_K/2; kx++){
                    sum_x += sobel_kernel_d[ky + SOBEL_K/2][kx + SOBEL_K/2] * buf2[y + ky][x + kx];
                    sum_y += sobel_kernel_d[kx + SOBEL_K/2][ky + SOBEL_K/2] * buf2[y+ ky][x + kx];
                }
            }
            g_mag[y - (PADDING - 1)][x - (PADDING - 1)] = sqrtf(sum_x * sum_x + sum_y * sum_y);
            g_angle[y - (PADDING - 1)][x - (PADDING - 1)] = atan2f(sum_y, sum_x) * 180.0 / PI;
        }
    }

    __syncthreads();

    // do nms in place
    for(int y = 1 + ty; y < TILE_Y + 1; y += blockDim.y){
        for(int x = 1 + tx; x < TILE_X + 1; x += blockDim.x){
            int dir = angle_direction(g_angle[y][x]);
            int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
            if (dir == HORIZONTAL_DIR) {
                dx1 = 1; dy1 = 0; dx2 = -1; dy2= 0;
            }
            else {
                if (dir == VERTICAL_DIR) {
                    dx1 = 0; dy1 = 1; dx2 = 0; dy2 = -1;
                }
                if (dir == TL_BR_DIR) {
                    dx1 = -1; dy1 = 1; dx2 = 1; dy2 = -1;
                }
                if (dir == TR_BL_DIR) {
                    dx1 = 1; dy1 = 1; dx2= -1; dy2 = -1;
                }
            }
            half mag = g_mag[y][x];
            if (g_mag[y+dy1][x+dx1] > mag || g_mag[y+dy2][x+dx2] > mag) {
                output_buff[flat_idx(img_x_start - 1, img_y_start + y - 1, width)] = 0;
            }
            else {
                if (mag > UPPER_THRESHOLD) {
                    output_buff[flat_idx(img_x_start + x - 1, img_y_start + y - 1, width)] = 2;
                }
                else if (mag < LOWER_THRESHOLD){
                    output_buff[flat_idx(img_x_start + x - 1, img_y_start + y - 1, width)] = 0;
                }
                else {
                    output_buff[flat_idx(img_x_start + x - 1, img_y_start + y - 1, width)] = 1;
                }
            }
        }
    }
}