#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "types.hpp"
#include "kernel.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

void vectorAddHost(Vector& result, const Vector& x, const Vector& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("Vector sizes must be equal for addition");
    }
    
    result.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    try {
        po::options_description desc("Vector addition benchmarking. Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("input,i", po::value<std::string>()->required(), "input JSON filepath")
            ("output,o", po::value<std::string>()->required(), "output JSON filepath")
            ("sequential,s", "Do addition on CPU")
            ("parallel,p", "Do addition on GPU");

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

        // Read input data:
        Vector x, y;
        readInputJSON(x, y, vm["input"].as<std::string>());

        // Compute vector addition:
        auto start = std::chrono::high_resolution_clock::now();

        Vector result;
        if (vm.count("sequential")) {
            vectorAddHost(result, x, y);
        } else if (vm.count("parallel")) {
            vectorAddDevice(result, x, y);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Vector addition completed in: " << duration.count() << " microseconds. " << std::endl;
        std::cout << "Output sample: " << result.size() << std::endl;
        for (size_t i = 0; i < std::min(result.size(), static_cast<size_t>(10)); ++i) {
            ElementType value = result[i];
            std::cout << "\t" << i << ": " << value << std::endl;
        }
        std::cout << "MD5 hash of result: " << computeMD5(result) << std::endl;

        // Write output data:
        writeOutputJSON(result, vm["output"].as<std::string>());
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
