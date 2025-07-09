#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "kernel.hpp"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Image blurring benchmarking. Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>()->required(), "input RGB image filepath")
            ("kernel,k", po::value<int>()->required(), "kernel size for blurring, must be odd integer")
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

        // Do conversion:
        cv::Mat grayscacleImage(inputImage.size(), CV_8UC1);
        cv::cvtColor(inputImage, grayscacleImage, cv::COLOR_BGR2GRAY);

        // Do blurring:
        const int kernelSize = vm["kernel"].as<int>();
        if (kernelSize % 2 == 0 || kernelSize <= 0) {
            std::cerr << "Error: Kernel size must be a positive odd integer" << std::endl;
            return 1;
        }
        cv::Mat outputImage(grayscacleImage.size(), CV_8UC1);
        auto start = std::chrono::high_resolution_clock::now();

        if (vm.count("sequential")) {
            // Do conversion using OpenCV
            cv::Size kSize(kernelSize, kernelSize);
            cv::blur(grayscacleImage, outputImage, kSize);
        } else if (vm.count("parallel")) {
            // Do conversion using CUDA
            const int halfKernelSize{kernelSize >> 1};
            blurDevice(outputImage, grayscacleImage, halfKernelSize);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "RGB to grayscale conversion completed in: " << duration.count() << " microseconds. " << std::endl;

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
