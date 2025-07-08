#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <opencv2/opencv.hpp>

// CUDA RGB to grayscale conversion 
void cvtColorDevice(cv::Mat& outputImage, const cv::Mat& inputImage);

#endif // KERNEL_HPP
