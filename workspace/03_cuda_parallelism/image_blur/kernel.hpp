#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <opencv2/opencv.hpp>

// CUDA RGB to grayscale conversion 
void blurDevice(cv::Mat& outputImage, const cv::Mat& inputImage, const int halfKernelSize);

#endif // KERNEL_HPP
