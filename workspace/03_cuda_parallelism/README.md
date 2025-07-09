# Module 03 - CUDA Parallelism Model

---

## RGB To Grayscale

```bash
# OpenCV
bazel run //03_cuda_parallelism/rgb_to_grayscale:rgb_to_grayscale -- --input $PWD/03_cuda_parallelism/rgb_to_grayscale/data/rgb.png --output $PWD/03_cuda_parallelism/rgb_to_grayscale/data/grayscale-cpu.png --sequential
# Parallel
bazel run //03_cuda_parallelism/rgb_to_grayscale:rgb_to_grayscale -- --input $PWD/03_cuda_parallelism/rgb_to_grayscale/data/rgb.png --output $PWD/03_cuda_parallelism/rgb_to_grayscale/data/grayscale-gpu.png --parallel
```

---

## Image Blurring

```bash
# OpenCV
bazel run //03_cuda_parallelism/image_blur:image_blur -- --input $PWD/03_cuda_parallelism/image_blur/data/input.png --kernel 5 --output $PWD/03_cuda_parallelism/image_blur/data/output-cpu.png --sequential
# Parallel
bazel run //03_cuda_parallelism/image_blur:image_blur -- --input $PWD/03_cuda_parallelism/image_blur/data/input.png --kernel 5 --output $PWD/03_cuda_parallelism/image_blur/data/output-gpu.png --parallel
```