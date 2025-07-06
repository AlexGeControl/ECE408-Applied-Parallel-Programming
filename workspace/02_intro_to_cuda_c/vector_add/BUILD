load("@rules_cuda//cuda:defs.bzl", "cuda_binary")

cuda_binary(
    name = "vector_add",
    srcs = ["main.cpp"],
    deps = [
        "@boost.program_options",
        "@boost.property_tree"
    ]
)
