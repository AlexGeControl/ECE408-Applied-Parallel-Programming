#include "kernel.hpp"
#include <cuda_runtime.h>
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

__global__
void vectorAddKernel(ElementType* result, const ElementType* x, const ElementType* y, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = x[i] + y[i];
    }
}

void vectorAddDevice(Vector& result, const Vector& x, const Vector& y) { 
    if (x.size() != y.size()) {
        throw std::runtime_error("Vector sizes must be equal for addition");
    }

    // Allocate memory on the device
    const size_t elementCount{x.size()};
    const size_t vectorSize{elementCount * sizeof(ElementType)};
    ElementType* resultDevice;
    ElementType* xDevice;
    ElementType* yDevice;
    
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&resultDevice), vectorSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&xDevice), vectorSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&yDevice), vectorSize));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(xDevice, x.data(), vectorSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(yDevice, y.data(), vectorSize, cudaMemcpyHostToDevice));

    // Compute vector addition using device kernel
    const int threadsPerBlock{1024};
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((elementCount + threadsPerBlock - 1) / threadsPerBlock);
    vectorAddKernel<<<gridDim, blockDim>>>(resultDevice, xDevice, yDevice, elementCount);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output from device to host
    result.resize(elementCount);
    CUDA_CHECK(cudaMemcpy(result.data(), resultDevice, vectorSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(resultDevice));
    CUDA_CHECK(cudaFree(xDevice));
    CUDA_CHECK(cudaFree(yDevice));
}
