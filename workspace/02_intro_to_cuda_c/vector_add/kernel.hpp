#ifndef KERNEL_HPP
#define KERNEL_HPP

#include "types.hpp"

// CUDA vector addition function
void vectorAddDevice(Vector& result, const Vector& x, const Vector& y);

#endif // KERNEL_HPP
