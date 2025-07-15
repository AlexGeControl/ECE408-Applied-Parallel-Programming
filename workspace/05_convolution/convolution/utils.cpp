#include "utils.hpp"

namespace utils {
    double getMaxPixelDifference(const cv::Mat& imageSource, const cv::Mat& imageTarget) {
        return cv::norm(imageSource, imageTarget, cv::NORM_INF);
    }

    double computeGflops(const cv::Mat& image, const cv::Mat& kernel, int64_t duration_microseconds) {
        const int M{image.rows};
        const int N{image.cols};
        const int K{kernel.rows >> 1}; // For a 3x3 kernel, K (radius) is 1

        // Number of floating point operations for GEMM: 2*M*K*N
        double kernelSize = static_cast<double>(2 * K + 1);
        double flops = 2.0 * kernelSize * kernelSize * static_cast<double>(M) * static_cast<double>(N);
        double seconds = static_cast<double>(duration_microseconds);
        return (flops / 1e3) / seconds;
    }
}
