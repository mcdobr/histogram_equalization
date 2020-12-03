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
// Doesn't work
// It uses a continuous double buffer, so a single memory location with twice the number of slots as the input.
// The complexity of this is O(N*logN) whereas the trivial CPU single threaded version is O(N).
// It handles only arrays smaller than max number of threads per 1 block.
__global__ void scan(uint32_t* output, const uint32_t* const input, const uint32_t n, const uint32_t offset)
{
    const unsigned thread_index = threadIdx.x;
    if (thread_index >= offset)
    {
        output[thread_index] = input[thread_index] + input[thread_index - offset];
    }
    else
    {
        output[thread_index] = input[thread_index];
    }
}

__global__ void equalize_image(const uint8_t* original_image, const size_t number_of_pixels, const uint32_t* cdf,
                               uint8_t* equalized_image)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    equalized_image[index] = cdf[original_image[index]] * (number_of_bins - 1) / number_of_pixels;
}

int get_elapsed_time(float time_in_miliseconds) {
    return (int)(time_in_miliseconds * 1000);
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

    // Time transfer to GPU
    cudaEvent_t start_transfer, end_transfer;
    cudaEventCreate(&start_transfer);
    cudaEventCreate(&end_transfer);
    cudaEventRecord(start_transfer);

    // Copy image to GPU
    uint8_t* dev_image = nullptr;
    gpu_error_check(cudaMalloc(&dev_image, number_of_pixels * sizeof(uint8_t)));
    gpu_error_check(cudaMemcpy(dev_image, host_image, number_of_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice));

    cudaEventRecord(end_transfer, 0);
    cudaEventSynchronize(end_transfer);

    float transfer_elapsed_time;
    cudaEventElapsedTime(&transfer_elapsed_time, start_transfer, end_transfer);
    cudaEventDestroy(start_transfer);
    cudaEventDestroy(end_transfer);
    cout << "The time to transfer image to GPU: " << get_elapsed_time(transfer_elapsed_time) << " microseconds\n";

    // Time histogram equalization
    cudaEvent_t start_histogram_equalization, end_histogram_equalization;
    cudaEventCreate(&start_histogram_equalization);
    cudaEventCreate(&end_histogram_equalization);
    cudaEventRecord(start_histogram_equalization);

    // Initialize histogram on device
    uint32_t* dev_histogram = nullptr;
    gpu_error_check(cudaMalloc(&dev_histogram, number_of_bins * sizeof(uint32_t)));
    gpu_error_check(cudaMemset(dev_histogram, 0, number_of_bins * sizeof(uint32_t)));

    // Compute histogram of image using a naive kernel
    histogram<<<number_of_pixels / max_threads_per_block, max_threads_per_block>>>(dev_image, dev_histogram);
    gpu_error_check(cudaGetLastError());
    gpu_error_check(cudaDeviceSynchronize());

    // Compute the cumulative distribution function using a naive
    uint32_t* dev_cdf = nullptr;
    gpu_error_check(cudaMalloc(&dev_cdf, number_of_bins * sizeof(uint32_t)));
    gpu_error_check(cudaMemset(dev_cdf, 0, number_of_bins * sizeof(uint32_t)));

    // Compute the cumulative distribution function
    uint32_t* temp = nullptr;
    gpu_error_check(cudaMalloc(&temp, number_of_bins * sizeof(uint32_t)));
    gpu_error_check(cudaMemcpy(temp, dev_histogram, number_of_bins * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
    for (uint32_t offset = 1; offset < number_of_bins; offset *= 2)
    {
        scan<<<1, number_of_bins, 2 * number_of_bins * sizeof(int32_t)>>>(dev_cdf, temp, number_of_bins, offset);
        gpu_error_check(cudaGetLastError());
        gpu_error_check(cudaDeviceSynchronize());
        if (offset * 2 < number_of_bins)
        {
            std::swap(temp, dev_cdf);
        }
    }

    // Compute the new image values
    uint8_t* dev_equalized_image = nullptr;
    gpu_error_check(cudaMalloc(&dev_equalized_image, number_of_pixels * sizeof(uint8_t)));
    equalize_image<<<number_of_pixels / max_threads_per_block, max_threads_per_block>>>(
        dev_image, number_of_pixels, dev_cdf, dev_equalized_image);
    gpu_error_check(cudaGetLastError());
    gpu_error_check(cudaDeviceSynchronize());

    cudaEventRecord(end_histogram_equalization, 0);
    cudaEventSynchronize(end_histogram_equalization);

    float histogram_equalization_elapsed_time;
    cudaEventElapsedTime(&histogram_equalization_elapsed_time, start_histogram_equalization,
                         end_histogram_equalization);
    cudaEventDestroy(start_histogram_equalization);
    cudaEventDestroy(end_histogram_equalization);
    cout << "The time to equalize the histogram on the GPU for the input image: " <<
        get_elapsed_time(histogram_equalization_elapsed_time) << " microseconds\n";


    // Time the transfer back to the CPU
    cudaEvent_t start_transfer_back, end_transfer_back;
    cudaEventCreate(&start_transfer_back);
    cudaEventCreate(&end_transfer_back);
    cudaEventRecord(start_transfer_back);

    // Copy the equalized image back to the CPU
    uint8_t* host_equalized_image = nullptr;
    host_equalized_image = (uint8_t*)malloc(number_of_pixels * sizeof(uint8_t));
    gpu_error_check(
        cudaMemcpy(host_equalized_image, dev_equalized_image, number_of_pixels * sizeof(uint8_t), cudaMemcpyDeviceToHost
        ));

    cudaEventRecord(end_transfer_back, 0);
    cudaEventSynchronize(end_transfer_back);

    float transfer_back_elapsed_time;
    cudaEventElapsedTime(&transfer_back_elapsed_time, start_transfer_back, end_transfer_back);
    cudaEventDestroy(start_transfer_back);
    cudaEventDestroy(end_transfer_back);
    cout << "The time to transfer the equalized image back to CPU: " <<
        get_elapsed_time(transfer_back_elapsed_time) << " microseconds\n";


    cout << "The total time (without loading initial image from filesystem and displaying them at the end): "
        << get_elapsed_time(transfer_elapsed_time + histogram_equalization_elapsed_time + transfer_back_elapsed_time) << " microseconds\n";

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
