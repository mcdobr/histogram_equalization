#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// For __syncthreads to not be not found anymore
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <device_functions.h>

using namespace std;
using namespace cv;

const int max_threads_per_block = 1024;
const int number_of_bins = 256;

#define gpu_error_check(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(const cudaError_t code, const char* file, const int line, const bool abort = true)
{
    if (code != cudaSuccess)
    {
        cerr << "GPU_assert: " << cudaGetErrorString(code) << " " << file << " " << line << ".\n";
        if (abort)
        {
            exit(code);
        }
    }
}

Mat read_image(const string image_path)
{
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if (image.empty())
    {
        cerr << "The provided image at path " << image_path << " could not be read\n";
        exit(-2);
    }
    return image;
}

__global__ void histogram(const uint8_t* image, uint32_t* histogram)
{
    const uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
    atomicAdd(&histogram[image[index]], 1);
}

// Naive implementation of inclusive scan algorithm based on the exclusive scan presented in
// https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
// It uses a continuous double buffer, so a single memory location with twice the number of slots as the input.
// The complexity of this is O(N*logN) whereas the trivial CPU single threaded version is O(N).
// It handles only arrays smaller than max number of threads per 1 block.
// Doesn't work
__global__ void scan(uint32_t* prefix_sums, uint32_t* arr, const int n)
{
    volatile __shared__ uint32_t temp[512]; // todo: is there any way to parametrize this in CUDA?
    // Load input into shared memory.    
    // This is exclusive scan, so shift right by one    
    // and set first element to 0
    if (threadIdx.x < n)
    {
        temp[threadIdx.x] = arr[threadIdx.x];
    }
    __syncthreads();

    int pout = 1, pin = 1;
    for (int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        if (threadIdx.x >= offset)
        {
            temp[pout * n + threadIdx.x] += temp[pin * n + threadIdx.x - offset];
        }
        else
        {
            temp[pout * n + threadIdx.x] = temp[pin * n + threadIdx.x];
        }
        __syncthreads();
    }

    prefix_sums[threadIdx.x] = temp[pout * n + threadIdx.x]; // write output
}

__global__ void equalize_image(const uint8_t* original_image, const size_t number_of_pixels, const uint32_t* cdf,
                               uint8_t* equalized_image)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    equalized_image[index] = cdf[original_image[index]] * (number_of_bins - 1) / number_of_pixels;
}

// Also see https://www.mygreatlearning.com/blog/histogram-equalization-explained/#Algorithm
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "An image file path is needed!\n");
        exit(-1);
    }

    string image_path(argv[1]);
    Mat image = read_image(image_path);

    if (!image.isContinuous())
    {
        std::cerr << "Image is not read but stitched together so it is not continuous\n";
        exit(-3);
    }

    size_t number_of_pixels = image.total();
    uint8_t* host_image = (uint8_t*)malloc(number_of_pixels * sizeof(uint8_t));
    memcpy_s(host_image, number_of_pixels * sizeof(uint8_t), image.data, number_of_pixels * sizeof(uint8_t));

    uint8_t* dev_image = nullptr;
    gpu_error_check(cudaMalloc(&dev_image, number_of_pixels * sizeof(uint8_t)));
    gpu_error_check(cudaMemcpy(dev_image, host_image, number_of_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    uint32_t* dev_histogram = nullptr;
    gpu_error_check(cudaMalloc(&dev_histogram, number_of_bins * sizeof(uint32_t)));
    gpu_error_check(cudaMemset(dev_histogram, 0, number_of_bins * sizeof(uint32_t)));

    // Compute histogram of image using a naive kernel
    histogram<<<number_of_pixels / max_threads_per_block, max_threads_per_block>>>(dev_image, dev_histogram);
    gpu_error_check(cudaGetLastError());
    gpu_error_check(cudaDeviceSynchronize());

    uint32_t* host_histogram = (uint32_t*)malloc(number_of_bins * sizeof(uint32_t));
    cudaMemcpy(host_histogram, dev_histogram, number_of_bins * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // free(host_histogram);


    // todo: remove this because scan is broken
    uint32_t* debug_cdf = (uint32_t*)malloc(number_of_bins * sizeof(uint32_t));
    debug_cdf[0] = host_histogram[0];
    for (int i = 1; i < number_of_bins; ++i)
    {
        debug_cdf[i] = debug_cdf[i - 1] + host_histogram[i];
    }


    // Compute the cumulative distribution function using a naive
    uint32_t* dev_cdf = nullptr;
    gpu_error_check(cudaMalloc(&dev_cdf, number_of_bins * sizeof(uint32_t)));
    gpu_error_check(cudaMemcpy(dev_cdf, debug_cdf, number_of_bins * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // gpu_error_check(cudaMemset(dev_cdf, 0, number_of_bins * sizeof(uint32_t)));
    // scan<<<1, number_of_bins>>>(dev_cdf, dev_histogram, number_of_bins);
    // gpu_error_check(cudaGetLastError());
    // gpu_error_check(cudaDeviceSynchronize());

    uint32_t *host_cdf = (uint32_t*)malloc(number_of_bins * sizeof(uint32_t));
    cudaMemcpy(host_cdf, dev_cdf, number_of_bins * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // free(host_cdf);

    // Compute the new image values
    uint8_t* dev_equalized_image = nullptr;
    gpu_error_check(cudaMalloc(&dev_equalized_image, number_of_pixels * sizeof(uint8_t)));
    equalize_image<<<number_of_pixels / max_threads_per_block, max_threads_per_block>>>(
        dev_image, number_of_pixels, dev_cdf, dev_equalized_image);
    gpu_error_check(cudaGetLastError());
    gpu_error_check(cudaDeviceSynchronize());

    // Copy the equalized image back to the CPU
    uint8_t* host_equalized_image = nullptr;
    host_equalized_image = (uint8_t*)malloc(number_of_pixels * sizeof(uint8_t));
    gpu_error_check(
        cudaMemcpy(host_equalized_image, dev_equalized_image, number_of_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost
        ));

    // Create and image for displaying
    Mat equalized_image = Mat(image.rows, image.cols, CV_8UC1, host_equalized_image);

    imshow("Original image", image);
    imshow("Equalized image", equalized_image);
    waitKey(0);

    // Free all memory
    free(host_image);
    free(host_equalized_image);
    cudaFree(dev_image);
    cudaFree(dev_histogram);
    cudaFree(dev_cdf);
    cudaFree(dev_equalized_image);
    return 0;
}
