# Module 05 - Convolution

---

## Convolution

```bash
# OpenCV on CPU
bazel run //05_convolution/convolution:convolution -- --input $PWD/05_convolution/convolution/data/input.png --kernel sharpening --output $PWD/05_convolution/convolution/data/output-cpu-sharpening.png --sequential
bazel run //05_convolution/convolution:convolution -- --input $PWD/05_convolution/convolution/data/input.png --kernel blurring --output $PWD/05_convolution/convolution/data/output-cpu-blurring.png --sequential
# CUDA
bazel run //05_convolution/convolution:convolution -- --input $PWD/05_convolution/convolution/data/input.png --kernel sharpening --output $PWD/05_convolution/convolution/data/output-gpu-sharpening.png --parallel
bazel run //05_convolution/convolution:convolution -- --input $PWD/05_convolution/convolution/data/input.png --kernel blurring --output $PWD/05_convolution/convolution/data/output-gpu-blurring.png --parallel
```