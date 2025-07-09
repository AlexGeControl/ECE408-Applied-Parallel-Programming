#include "kernel.hpp"
#include <cuda_runtime.h>
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

__global__
void blurKernel(
    Byte* outputImage, 
    const Byte* inputImage, 
    const unsigned int height, 
    const unsigned int width,
    const int halfKernelSize
) {
    const unsigned int targetRow = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int targetCol = blockIdx.x * blockDim.x + threadIdx.x;

    // Accumulate the input pixel values within the kernel window
    unsigned int total{0U};
    unsigned int count{0U};
    for (int rowOffset{-halfKernelSize}; rowOffset <= halfKernelSize; ++rowOffset) {
        for (int colOffset{-halfKernelSize}; colOffset <= halfKernelSize; ++colOffset) {
            const int row = static_cast<int>(targetRow) + rowOffset;
            const int col = static_cast<int>(targetCol) + colOffset;

            // Skip out-of-bounds pixels
            if (row < 0 || row >= static_cast<int>(height) || 
                col < 0 || col >= static_cast<int>(width)) {
                continue;
            }

            // Compute the offset for the current pixel
            const unsigned int offset = row * width + col;

            total += inputImage[offset];
            count += 1U;
        }
    }

    // Set the output pixel value to the average
    if (count < 1U) {
        return;
    }
    const unsigned int targetOffset = targetRow * width + targetCol;
    outputImage[targetOffset] = static_cast<Byte>(total / count);
}

void blurDevice(cv::Mat& outputImage, const cv::Mat& inputImage, const int halfKernelSize) {
    if (inputImage.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    // Allocate memory on the device
    const int height{inputImage.rows};
    const int width{inputImage.cols};

    const size_t imageSize{height * width * sizeof(Byte)};

    Byte* inputImageDevice{nullptr};
    Byte* outputImageDevice{nullptr};
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&inputImageDevice), imageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&outputImageDevice), imageSize));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(inputImageDevice, inputImage.data, imageSize, cudaMemcpyHostToDevice));

    // Compute vector addition using device kernel
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (static_cast<unsigned int>(std::ceil(static_cast<float>(width) / blockDim.x))),
        (static_cast<unsigned int>(std::ceil(static_cast<float>(height) / blockDim.y)))
    );
    blurKernel<<<gridDim, blockDim>>>(outputImageDevice, inputImageDevice, height, width, halfKernelSize);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(outputImage.data, outputImageDevice, imageSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(outputImageDevice));
    CUDA_CHECK(cudaFree(inputImageDevice));
}
