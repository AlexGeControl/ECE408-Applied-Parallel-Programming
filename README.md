# ECE 408: Applied Parallel Programming

Pre-prep. course for CMU LTI 11-868: Large Language Model Systems

---

## Resources

[Illinois-NVIDIA GPU Teaching Kit](http://gputeachingkit.hwu.crhc.illinois.edu/)

---

## Development Setup

---

### Bazel + CUDA

The `workspace/` directory contains a reference Bazel setup for CUDA parallel programming in C++. This setup includes:

- CUDA rules configuration
- Sample CUDA kernels and host code
- Build configurations for different GPU architectures

To build and run:
```bash
cd workspace
bazel build //src:vector_add
bazel run //src:vector_add
```