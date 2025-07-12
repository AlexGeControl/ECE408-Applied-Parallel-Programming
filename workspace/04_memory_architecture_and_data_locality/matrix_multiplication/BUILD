load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

cuda_binary(
    name = "matrix_multiplication",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "@boost.program_options",
        "@eigen",
        ":kernel",
        ":utils"
    ],
)

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
    hdrs = ["kernel.hpp"],
    deps = [
        "@eigen",     
    ]
)

cuda_library(
    name = "utils",
    srcs = [
        "utils.cpp",
    ],
    hdrs = ["utils.hpp"],
    deps = [
        "@boost.uuid",
        "@eigen",
    ],
)