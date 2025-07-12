#include "utils.hpp"
#include <boost/uuid/detail/md5.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace utils {
    std::string computeEigenMatrixMD5(const Eigen::MatrixXf& matrix) {
        // Get pointer to the underlying data
        const auto* data = matrix.data();
        size_t size = matrix.size() * sizeof(*data);
        
        // Compute MD5 using Boost
        boost::uuids::detail::md5 hash;
        boost::uuids::detail::md5::digest_type digest;
        
        hash.process_bytes(data, size);
        hash.get_digest(digest);
        
        // Convert digest to hex string
        std::ostringstream oss;
        for (int i = 0; i < 4; ++i) {
            oss << std::hex << std::setfill('0') << std::setw(8) << digest[i];
        }
        
        return oss.str();
    }
    
    void showEigenMatrix(const Eigen::MatrixXf& matrix) {
        std::cout << matrix << std::endl;
    }

    double computeGflops(int M, int K, int N, int64_t duration_microseconds) {
        // Number of floating point operations for GEMM: 2*M*K*N
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
        double seconds = static_cast<double>(duration_microseconds);
        return (flops / 1e3) / seconds;
    }
}
