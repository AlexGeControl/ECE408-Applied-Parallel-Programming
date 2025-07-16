#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <opencv2/opencv.hpp>

// CUDA 2D grayscale image convolution kernel 
int64_t convolutionDevice(cv::Mat& outputImage, const cv::Mat& inputImage, const cv::Mat& filter, const bool useTiledKernel = true);

#endif // KERNEL_HPP
