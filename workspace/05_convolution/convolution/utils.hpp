#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

namespace utils {
    /**
     * Computes the maximum absolute pixel difference between two images.
     * @param imageSource The first image matrix for comparison (reference image)
     * @param imageTarget The second image matrix for comparison (test image)
     * @return The maximum absolute difference between corresponding pixels as a double
     */
    double getMaxPixelDifference(const cv::Mat& imageSource, const cv::Mat& imageTarget);

    /**
     * Computes the achieved GFLOPS for 2D grayscale image convolution given dimensions and duration.
     * @param image Input image (cv::Mat)
     * @param kernel Convolution kernel (cv::Mat)
     * @param duration_microseconds Time taken for computation in microseconds
     * @return Achieved GFLOPS (double)
     */
    double computeGflops(const cv::Mat& image, const cv::Mat& kernel, int64_t duration_microseconds);
};

#endif // UTILS_HPP