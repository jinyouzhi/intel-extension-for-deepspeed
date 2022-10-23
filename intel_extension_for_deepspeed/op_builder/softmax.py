from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class SoftmaxBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_SOFTMAX"
    NAME = "softmax"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/sycl/ds_softmax_sycl.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/softmax_kernels.dp.cpp'),
        ]

    def include_paths(self):
        return [sycl_kernel_include('csrc/includes'), 'csrc/includes']
