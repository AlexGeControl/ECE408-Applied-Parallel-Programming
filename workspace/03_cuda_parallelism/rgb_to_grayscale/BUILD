load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

cuda_binary(
    name = "rgb_to_grayscale",
    srcs = ["main.cpp"],
    deps = [
        "@boost.program_options",
        "@opencv//:opencv",
        ":kernel",
    ],
)

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
    hdrs = ["kernel.hpp"],
    deps = [
        "@opencv//:opencv",      
    ]
)