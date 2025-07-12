# Module 04 - Memory Architecture and Data Locality

---

## Matrix Multiplication

```bash
# Eigen
bazel run //04_memory_architecture_and_data_locality/matrix_multiplication:matrix_multiplication -- --M 1000 --K 800 --N 600 --sequential
# CUDA
bazel run //04_memory_architecture_and_data_locality/matrix_multiplication:matrix_multiplication -- --M 1000 --K 800 --N 600 --parallel
```