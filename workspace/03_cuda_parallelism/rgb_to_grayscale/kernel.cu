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
void cvtColorKernel(Byte* outputImage, const Byte* inputImage, const size_t height, const size_t width) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip threads that are out of bounds
    if (x >= width || y >= height) {
        return; 
    }

    const unsigned int offset = y * width + x;

    // Get RGB values from input image
    const unsigned int inputOffset = offset * 3;

    const Byte b = inputImage[inputOffset + 0U];
    const Byte g = inputImage[inputOffset + 1U];
    const Byte r = inputImage[inputOffset + 2U];

    // Convert to grayscale using the luminosity method
    const Byte i = static_cast<Byte>(0.299f * r + 0.587f * g + 0.114f * b);

    // Set the grayscale value in the output image
    const unsigned int outputOffset = offset;
    outputImage[outputOffset] = i;
}

void cvtColorDevice(cv::Mat& outputImage, const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    
    // Allocate memory on the device
    const int height{inputImage.rows};
    const int width{inputImage.cols};
    const int channels{inputImage.channels()};
    if (channels != 3) {
        throw std::runtime_error("Input image must have 3 channels (BGR format)");
    }

    const size_t inputImageSize{height * width * channels * sizeof(Byte)};
    const size_t outputImageSize{height * width * sizeof(Byte)};

    Byte* inputImageDevice{nullptr};
    Byte* outputImageDevice{nullptr};
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&inputImageDevice), inputImageSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&outputImageDevice), outputImageSize));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(inputImageDevice, inputImage.data, inputImageSize, cudaMemcpyHostToDevice));

    // Compute vector addition using device kernel
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (static_cast<unsigned int>(std::ceil(static_cast<float>(width) / blockDim.x))),
        (static_cast<unsigned int>(std::ceil(static_cast<float>(height) / blockDim.y)))
    );
    cvtColorKernel<<<gridDim, blockDim>>>(outputImageDevice, inputImageDevice, height, width);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(outputImage.data, outputImageDevice, outputImageSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(outputImageDevice));
    CUDA_CHECK(cudaFree(inputImageDevice));
}
