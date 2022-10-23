from .builder import SYCLOpBuilder, sycl_kernel_path, sycl_kernel_include


class LayerReorderBuilder(SYCLOpBuilder):
    BUILD_VAR = "DS_BUILD_LAYERREORDER"
    NAME = "layer_reorder"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.transformer.{self.NAME}_op'

    def sources(self):
        return [
            sycl_kernel_path('csrc/transformer/sycl/ds_layer_reorder_sycl.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/transform_kernels.dp.cpp'),
            sycl_kernel_path('csrc/transformer/sycl/general_kernels.dp.cpp'),
        ]

    def include_paths(self):
        return [sycl_kernel_include('csrc/includes'), 'csrc/includes']
