#ifndef UTILS_HPP
#define UTILS_HPP

#include "types.hpp"
#include <string>

// JSON I/O functions
void readInputJSON(Vector& x, Vector& y, const std::string& inputJSON);
void writeOutputJSON(const Vector& result, const std::string& outputJSON);

// Utility functions
std::string computeMD5(const Vector& result);

#endif // UTILS_HPP
