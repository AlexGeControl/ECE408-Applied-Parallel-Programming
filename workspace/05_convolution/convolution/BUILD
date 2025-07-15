load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

cuda_binary(
    name = "convolution",
    srcs = ["main.cpp"],
    deps = [
        "@boost.program_options",
        "@opencv//:opencv",
        ":utils",
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

cuda_library(
    name = "utils",
    srcs = [
        "utils.cpp",
    ],
    hdrs = ["utils.hpp"],
    deps = [
        "@opencv//:opencv",      
    ]
)