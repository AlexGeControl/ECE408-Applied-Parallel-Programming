#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Get the number of CUDA devices available on the system
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    for (int device = 0; device < deviceCount; ++device) {
        // Get device properties for current device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        
        // Get device name
        std::cout << "\nDevice " << device << ": " << deviceProp.name << std::endl;
        // Get Compute Capability, defined as 'major.minor', which indicates the features supported by the device
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        // Get grid dimensions
        std::cout << "  Max grid dimensions: ("
                  << deviceProp.maxGridSize[0] << ", "
                  << deviceProp.maxGridSize[1] << ", "
                  << deviceProp.maxGridSize[2] << ")" << std::endl;
        // Get block dimensions
        std::cout << "  Max block dimensions: ("
                  << deviceProp.maxThreadsDim[0] << ", "
                  << deviceProp.maxThreadsDim[1] << ", "
                  << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        // Get the maximum number of threads per block, which is the maximum number of threads that can be launched in a single block
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        // Get global memory size
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1 << 20) << " MB" << std::endl;
        // Get const memory size, the ideal storage for constants that are read-only and shared by all threads
        std::cout << "  Total constant memory: " << deviceProp.totalConstMem / (1 << 10) << " KB" << std::endl;
        // Get shared memory size, the memory that is shared among threads in a block
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / (1 << 10) << " KB" << std::endl;
        // Get the number of Stream Multiprocessors (SMs), which are the basic units of parallel execution in a CUDA device
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        // Get the warp size, which is the number of threads in a warp (a group of threads that execute instructions in lockstep)
        std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
    }
    
    return EXIT_SUCCESS;
}
