#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <string>

namespace utils {
    /**
     * Computes the MD5 hash of the underlying flattened data of an Eigen::MatrixXf
     * @param matrix The input matrix
     * @return MD5 hash as a hexadecimal string
     */
    std::string computeEigenMatrixMD5(const Eigen::MatrixXf& matrix);
    
    /**
     * Prints an Eigen::MatrixXf to the console
     * @param matrix The input matrix to print
     */
    void showEigenMatrix(const Eigen::MatrixXf& matrix);

    /**
     * Computes the achieved GFLOPS for matrix multiplication given dimensions and duration.
     * @param M Number of rows in matrix A and C
     * @param K Number of columns in matrix A and rows in matrix B
     * @param N Number of columns in matrix B and C
     * @param duration_microseconds Time taken for computation in microseconds
     * @return Achieved GFLOPS (double)
     */
    double computeGflops(int M, int K, int N, int64_t duration_microseconds);
};

#endif // UTILS_HPP