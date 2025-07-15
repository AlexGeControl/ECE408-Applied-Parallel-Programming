#include "kernel.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

using Byte = unsigned char;

inline size_t getImageSize(const cv::Mat& image) {
    return static_cast<size_t>(image.rows * image.cols * sizeof(Byte));
}   

inline size_t getFilterSize(const cv::Mat& filter) {
    return static_cast<size_t>(filter.rows * filter.cols * sizeof(float));
}   

#define INPUT_TILE_SIZE 32
#define FILTER_RADIUS 1
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - 2 * FILTER_RADIUS)

// Declare device constant memory for filter
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)

__constant__ float filterDevice[FILTER_SIZE * FILTER_SIZE];

__global__
void naiveConvolutionKernel(
    Byte* outputImage, 
    const Byte* inputImage, 
    const unsigned int height, 
    const unsigned int width
) {
    const unsigned int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int outputCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip out-of-bounds pixels
    if (outputRow >= height || outputCol >= width) {
        return;
    }   

    // Do convolution
    float pixel{0.0f};
    for (int rowOffset{-FILTER_RADIUS}; rowOffset <= FILTER_RADIUS; ++rowOffset) {
        for (int colOffset{-FILTER_RADIUS}; colOffset <= FILTER_RADIUS; ++colOffset) {
            const int inputRow = static_cast<int>(outputRow) + rowOffset;
            const int inputCol = static_cast<int>(outputCol) + colOffset;

            // Skip out-of-bounds pixels
            if (inputRow < 0 || inputRow >= static_cast<int>(height) || 
                inputCol < 0 || inputCol >= static_cast<int>(width)) {
                continue;
            }

            // Compute the offset for the current pixel
            const unsigned int inputOffset = inputRow * width + inputCol;

            const int filterRow = rowOffset + FILTER_RADIUS;
            const int filterCol = colOffset + FILTER_RADIUS;
            const unsigned int filterOffset = filterRow * FILTER_SIZE + filterCol;

            pixel += static_cast<float>(inputImage[inputOffset]) * filterDevice[filterOffset];
        }
    }

    const unsigned int outputOffset = outputRow * width + outputCol;
    outputImage[outputOffset] = static_cast<Byte>(std::fmaxf(0.0f, fminf(255.0f, pixel)));
}

int64_t convolutionDevice(cv::Mat& outputImage, const cv::Mat& inputImage, const cv::Mat& filter) {
    if (inputImage.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    // Allocate memory on the device
    const int height{inputImage.rows};
    const int width{inputImage.cols};
    const size_t imageSize{getImageSize(inputImage)};

    Byte* inputImageDevice{nullptr};
    Byte* outputImageDevice{nullptr};
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&inputImageDevice), imageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&outputImageDevice), imageSize));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(inputImageDevice, inputImage.data, imageSize, cudaMemcpyHostToDevice));

    // Cache filter on device constant memory
    const size_t filterSize{getFilterSize(filter)};
    CUDA_CHECK(cudaMemcpyToSymbol(filterDevice, filter.data, filterSize));

    auto start = std::chrono::high_resolution_clock::now();

    // Compute 2D grayscale image convolution using device kernel
    {
        dim3 blockDim(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
        dim3 gridDim(
            (static_cast<unsigned int>(std::ceil(static_cast<float>(width) / blockDim.x))),
            (static_cast<unsigned int>(std::ceil(static_cast<float>(height) / blockDim.y)))
        );
        naiveConvolutionKernel<<<gridDim, blockDim>>>(outputImageDevice, inputImageDevice, height, width);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(outputImage.data, outputImageDevice, imageSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(outputImageDevice));
    CUDA_CHECK(cudaFree(inputImageDevice));

    return duration.count();
}
