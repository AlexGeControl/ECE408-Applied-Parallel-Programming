#include <boost/program_options.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <cassert>
#include "kernel.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

int64_t matrixMultiplicationHost(
    Eigen::MatrixXf& C,
    const Eigen::MatrixXf& A,
    const Eigen::MatrixXf& B
) {
    auto start = std::chrono::high_resolution_clock::now();

    C = A * B;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count();
}

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Matrix multiplication C = A × B benchmarking. Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("M,m", po::value<int>()->required(), "number of rows in matrix A and matrix C")
            ("K,k", po::value<int>()->required(), "number of columns in matrix A and rows in matrix B")
            ("N,n", po::value<int>()->required(), "number of columns in matrix B and matrix C")
            ("sequential,s", "Do matrix multiplication on CPU")
            ("parallel,p", "Do matrix multiplication on GPU");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return EXIT_SUCCESS;
        }

        // Check for mutually exclusive options
        if (vm.count("sequential") && vm.count("parallel")) {
            std::cerr << "Error: --sequential and --parallel are mutually exclusive" << std::endl;
            return EXIT_FAILURE;
        }

        if (!vm.count("sequential") && !vm.count("parallel")) {
            std::cerr << "Error: Either --sequential or --parallel must be specified" << std::endl;
            return EXIT_FAILURE;
        }

        po::notify(vm);

        // Parse matrix multiplication dimensions
        int M = vm["M"].as<int>();
        int K = vm["K"].as<int>();
        int N = vm["N"].as<int>();

        std::cout << "Matrix multiplication: C(" << M << "-by-" << N << ") = A(" << M << "-by-" << K << ") * B(" << K << "-by-" << N << ")" << std::endl;

        // Initialize matrices A, B, C
        Eigen::MatrixXf A = Eigen::MatrixXf::Random(M, K);
        Eigen::MatrixXf B = Eigen::MatrixXf::Random(K, N);

        // Get baseline result using Eigen on host
        Eigen::MatrixXf CHost = Eigen::MatrixXf::Zero(M, N);
        int64_t durationHost = matrixMultiplicationHost(CHost, A, B);
        std::cout << "Matrix multiplication on host completed in: " << durationHost << " microseconds. MD5 signature: " << utils::computeEigenMatrixMD5(CHost) << std::endl;

        if (vm.count("parallel")) {
            // Do naive matrix multiplication using CUDA C++
            Eigen::MatrixXf CDeviceNaive = Eigen::MatrixXf::Zero(M, N);
            int64_t durationDeviceNaive = matrixMultiplicationDevice(CDeviceNaive, A, B, false);
            std::cout << "Naive matrix multiplication on device completed in: " << durationDeviceNaive << " microseconds. GFLOPS: " << utils::computeGflops(M, N, K, durationDeviceNaive) << std::endl;

            assert(CDeviceNaive.isApprox(CHost, 1e-4f) && "Result from device does not match host result!");

            // Do tiled matrix multiplication using CUDA C++
            Eigen::MatrixXf CDeviceTiled = Eigen::MatrixXf::Zero(M, N);
            int64_t durationDeviceTiled = matrixMultiplicationDevice(CDeviceTiled, A, B, true);
            std::cout << "Tiled matrix multiplication on device completed in: " << durationDeviceTiled << " microseconds. GFLOPS: " << utils::computeGflops(M, N, K, durationDeviceTiled) << std::endl;

            assert(CDeviceTiled.isApprox(CHost, 1e-4f) && "Result from device does not match host result!");
        }

    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
