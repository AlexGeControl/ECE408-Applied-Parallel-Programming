#include "kernel.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

#define TILE_SIZE 32
#define BLOCK_SIZE TILE_SIZE

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
void naiveMatMulKernel(
    float* CDevice, const float* ADevice, const float* BDevice, 
    const int M, const int K, const int N
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Skip threads that are out of bounds
    if (row >= M || col >= N) {
        return; 
    }

    // Eigen uses column-major storage
    float CValue{0.0f};
    for (int k = 0; k < K; ++k) {
        const int AOffset = k * M + row; // A(row, k)
        const int BOffset = col * K + k; // B(k, col)
        
        CValue += ADevice[AOffset] * BDevice[BOffset];
    }

    const int COffset = col * M + row; // C(row, col) 
    CDevice[COffset] = CValue;
}

__global__
void tiledMatMulKernel(
    float* CDevice, const float* ADevice, const float* BDevice, 
    const int M, const int K, const int N
) {
    // Init on-chip shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int tileCol = threadIdx.x;
    const int tileRow = threadIdx.y;

    float CValue{0.0f};
    for (int k = 0; k < static_cast<int>(std::ceil(static_cast<float>(K) / TILE_SIZE)); ++k) {
        // Each thread loads or sets one element of A and one element of B into shared memory
        // By setting out-of-bounds elements to zero, control divergence in later computation is avoided
        // Load tile from A
        const int ACol = k * TILE_SIZE + tileCol;
        if (ACol < K && row < M) {
            const int AOffset = ACol * M + row; // A(row, ACol)
            As[tileRow][tileCol] = ADevice[AOffset]; // A(row, ACol)
        } else {
            As[tileRow][tileCol] = 0.0f;
        }

        // Load tile from B
        const int BRow = k * TILE_SIZE + tileRow;
        if (BRow < K && col < N) {
            const int BOffset = col * K + BRow; // B(BRow, col)
            Bs[tileRow][tileCol] = BDevice[BOffset]; // B(BRow, col)
        } else {
            Bs[tileRow][tileCol] = 0.0f;    
        }

        __syncthreads(); // Ensure all the tiles are loaded before computation

        for (int t = 0; t < TILE_SIZE; ++t) {
            CValue += As[tileRow][t] * Bs[t][tileCol]; // C(row, col) += A(row, k) * B(k, col)
        }
        __syncthreads(); // Ensure all threads are done using the tiles before loading new
    }

    // The boundary check cannot be moved to the beginning of the kernel
    // Otherwise, some elements in the shared memory will be left uninitialized, causing undefined behavior
    if (row >= M || col >= N) {
        return; 
    }
    
    const int COffset = col * M + row; // C(row, col)
    CDevice[COffset] = CValue;
}

int64_t matrixMultiplicationDevice(Eigen::MatrixXf& C, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B, const bool useTiledKernel) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Inner dimensions of A and B must match for multiplication.");
    }
    
    if (C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument("Output matrix C has incorrect dimensions.");
    }

    const size_t ASize{A.size() * sizeof(float)};
    const size_t BSize{B.size() * sizeof(float)};
    const size_t CSize(C.size() * sizeof(float));

    float* ADevice{nullptr};
    float* BDevice{nullptr};
    float* CDevice{nullptr};

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ADevice), ASize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&BDevice), BSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&CDevice), CSize));

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(ADevice, A.data(), ASize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(BDevice, B.data(), BSize, cudaMemcpyHostToDevice));

    // Do compute
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    auto start = std::chrono::high_resolution_clock::now();

    if (useTiledKernel) {
        dim3 gridDim(
            (static_cast<unsigned int>(std::ceil(static_cast<float>(N) / blockDim.x))),
            (static_cast<unsigned int>(std::ceil(static_cast<float>(M) / blockDim.y)))
        );
        tiledMatMulKernel<<<gridDim, blockDim>>>(CDevice, ADevice, BDevice, M, K, N);
    } else {
        dim3 gridDim(
            (static_cast<unsigned int>(std::ceil(static_cast<float>(M) / blockDim.x))),
            (static_cast<unsigned int>(std::ceil(static_cast<float>(N) / blockDim.y)))
        );
        naiveMatMulKernel<<<gridDim, blockDim>>>(CDevice, ADevice, BDevice, M, K, N);
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Copy output from device to host
    CUDA_CHECK(cudaMemcpy(C.data(), CDevice, CSize, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(ADevice));
    CUDA_CHECK(cudaFree(BDevice));
    CUDA_CHECK(cudaFree(CDevice));

    return duration.count();
}
