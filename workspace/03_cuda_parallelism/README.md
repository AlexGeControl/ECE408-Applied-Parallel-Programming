# Module 03 - CUDA Parallelism Model

---

## RGB To Grayscale

```bash
# OpenCV
bazel run //03_cuda_parallelism/rgb_to_grayscale:rgb_to_grayscale -- --input $PWD/03_cuda_parallelism/rgb_to_grayscale/data/rgb.png --output $PWD/03_cuda_parallelism/rgb_to_grayscale/data/grayscale-cpu.png --sequential
# Parallel
bazel run //03_cuda_parallelism/rgb_to_grayscale:rgb_to_grayscale -- --input $PWD/03_cuda_parallelism/rgb_to_grayscale/data/rgb.png --output $PWD/03_cuda_parallelism/rgb_to_grayscale/data/grayscale-gpu.png --parallel
```