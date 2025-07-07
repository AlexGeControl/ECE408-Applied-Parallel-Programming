load("@rules_cuda//cuda:defs.bzl", "cuda_library", "cuda_binary")

cuda_binary(
    name = "vector_add",
    srcs = ["main.cpp"],
    deps = [
        ":types",
        ":kernel",
        ":utils",
        "@boost.program_options",
    ]
)

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
    hdrs = ["kernel.hpp"],
    deps = [
        ":types",       
    ]
)

cc_library(
    name = "utils",
    srcs = ["utils.cpp"],
    hdrs = ["utils.hpp"],
    deps = [
        ":types",
        "@boost.property_tree",
        "@boost.uuid"
    ]
)

cc_library(
    name = "types",
    hdrs = ["types.hpp"],
)
