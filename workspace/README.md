# Programming Massively Parallel Processors, 4th Edition

---

## Module 02 - Introduction to CUDA C

### Device Query

```bash
# Build
bazel build //02_intro_to_cuda_c/device_query:device_query
# Run
bazel run //02_intro_to_cuda_c/device_query:device_query
```

### Vector Addition

```bash
# Sequential
bazel run //02_intro_to_cuda_c/vector_add:vector_add -- --input $PWD/02_intro_to_cuda_c/vector_add/input.json --output $PWD/02_intro_to_cuda_c/vector_add/output.json --sequential
# Parallel
bazel run //02_intro_to_cuda_c/vector_add:vector_add -- --input $PWD/02_intro_to_cuda_c/vector_add/input.json --output $PWD/02_intro_to_cuda_c/vector_add/output.json --parallel
```