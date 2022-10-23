"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class TransformerBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_TRANSFORMER"
    NAME = "transformer"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def extra_ldflags(self):
        return []

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/sycl/onednn_wrappers.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/ds_transformer_sycl.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/onemkl_wrappers.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/transform_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/gelu_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/dropout_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/normalize_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/softmax_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/general_kernels.dp.cpp')
        ]

    def include_paths(self):
        includes = [sycl_kernel_include('csrc/includes')]
        return includes
