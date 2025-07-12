#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <Eigen/Dense>

/**
 * @brief Computes the multiplication of two Eigen matrices using CUDA.
 * 
 * Performs matrix multiplication C = A * B using CUDA kernels with optional
 * tiled memory optimization.
 * 
 * @param C Output matrix to store the result of A * B
 * @param A First input matrix (left operand)
 * @param B Second input matrix (right operand) 
 * @param useTiledKernel Optional flag to use tiled kernel implementation (default: false)
 * @return Time in microseconds taken for the matrix multiplication on the device
 */
int64_t matrixMultiplicationDevice(Eigen::MatrixXf& C, const Eigen::MatrixXf& A, const Eigen::MatrixXf& B, const bool useTiledKernel = false);

#endif // KERNEL_HPP
