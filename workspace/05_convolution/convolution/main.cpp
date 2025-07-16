#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <cassert>
#include <iostream>
#include <string>
#include <chrono>
#include "kernel.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

int64_t convolutionHost(
    cv::Mat& outputImage,
    const cv::Mat& inputImage,
    const cv::Mat& filter
) {
    auto start = std::chrono::high_resolution_clock::now();

    cv::filter2D(inputImage, outputImage, -1, filter, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    return duration.count();
}

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Image convolution benchmarking. Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>()->required(), "input RGB image filepath")
            ("kernel,k", po::value<std::string>()->required(), "kernel type: 'sharpening' or 'blurring'")
            ("output,o", po::value<std::string>()->required(), "output grayscale image filepath")
            ("sequential,s", "Do conversion on CPU")
            ("parallel,p", "Do conversion on GPU");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        // Check for mutually exclusive options
        if (vm.count("sequential") && vm.count("parallel")) {
            std::cerr << "Error: --sequential and --parallel are mutually exclusive" << std::endl;
            return 1;
        }

        if (!vm.count("sequential") && !vm.count("parallel")) {
            std::cerr << "Error: Either --sequential or --parallel must be specified" << std::endl;
            return 1;
        }

        po::notify(vm);

        // Read input RGB image:
        const std::string inputFilePath{vm["input"].as<std::string>()};
        cv::Mat inputImage = cv::imread(inputFilePath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not read input image from " << inputFilePath << std::endl;
            return 1;
        }

        // Do RGB to grayscale conversion:
        cv::Mat grayscacleImage(inputImage.size(), CV_8UC1);
        cv::cvtColor(inputImage, grayscacleImage, cv::COLOR_BGR2GRAY);

        // Instantiate convolution kernel:
        const std::string kernelType = vm["kernel"].as<std::string>();
        if (kernelType != "sharpening" && kernelType != "blurring") {
            std::cerr << "Error: Filter type must be either 'sharpening' or 'blurring'" << std::endl;
            return 1;
        }

        // Create 3x3 convolution kernel based on kernel type
        cv::Mat kernel;
        if (kernelType == "sharpening") {
            // Sharpening kernel
            kernel = (cv::Mat_<float>(3, 3) <<
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0);
        } else { // blurring
            // Box blur kernel (3x3 averaging)
            kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
        }
    
        // For baseline, compute convolution using OpenCV on CPU and report GFLOPS
        cv::Mat outputImage(grayscacleImage.size(), CV_8UC1);
        int64_t durationHost = convolutionHost(outputImage, grayscacleImage, kernel);
        std::cout << "Image convolution (" << kernelType << ") on CPU completed in: " << durationHost << " microseconds. GFLOPS: " << utils::computeGflops(outputImage, kernel, durationHost) << std::endl;

        if (vm.count("parallel")) {
            // Do convolution using CUDA
            cv::Mat outputImageHost = outputImage.clone();
            int64_t durationDevice = convolutionDevice(outputImage, grayscacleImage, kernel);
            std::cout << "Image convolution (" << kernelType << ") on GPU completed in: " << durationDevice << " microseconds. GFLOPS: " << utils::computeGflops(outputImage, kernel, durationDevice) << std::endl;

            // Benchmark outputImage from GPU against CPU result for correctness
            //assert((utils::getMaxPixelDifference(outputImageHost, outputImage) <= 1.0) && "Result from device does not match host result!");
            std::cout << utils::getMaxPixelDifference(outputImageHost, outputImage) << std::endl;
        }

        // Write output grayscale image:
        const std::string outputFilePath{vm["output"].as<std::string>()};
        if (!cv::imwrite(outputFilePath, outputImage)) {
            std::cerr << "Error: Could not write output image to " << outputFilePath << std::endl;
            return 1;
        }
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
