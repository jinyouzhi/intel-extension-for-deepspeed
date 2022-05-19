#include "sycl/context.hpp"
#include "sycl/dropout.hpp"
#include "sycl/common.hpp"

template<typename T>
std::vector<torch::Tensor>
dropout_forward(float ratio,
                uint32_t dim,
                int bsz,
                const torch::Tensor &vals) {
    CHECK_INPUT(vals);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .layout(torch::kStrided)
        .device(torch::kXPU)
        .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T *input_ptr = (const T *)vals.data_ptr();
    T *output_ptr = (T *)output.data_ptr();
    uint8_t *mask_ptr = (uint8_t *)mask.data_ptr();

    sycl::queue *q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.Forward(bsz, output_ptr, input_ptr, q);
    return {output, mask};
}

template<typename T>
std::vector<torch::Tensor>
dropout_forward_with_bias(float ratio,
                          uint32_t dim,
                          int bsz,
                          const torch::Tensor &vals,
                          const torch::Tensor &bias,
                          const torch::Tensor &residual) {
    CHECK_INPUT(vals);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    auto output = torch::empty_like(vals);

    auto uint8_options = torch::TensorOptions()
        .dtype(torch::kInt8)
        .layout(torch::kStrided)
        .device(torch::kXPU)
        .requires_grad(false);

    auto mask = torch::empty({bsz, dim}, uint8_options);

    const T *input_ptr = (const T *)vals.data_ptr();
    const T *bias_ptr = (const T *)bias.data_ptr();
    const T *residual_ptr = (const T *)residual.data_ptr();
    T *output_ptr = (T *)output.data_ptr();
    uint8_t *mask_ptr = (uint8_t *)mask.data_ptr();

    sycl::queue *q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    _dropout.SetMask(mask_ptr);
    _dropout.ForwardWithBias(bsz, output_ptr, input_ptr, residual_ptr, bias_ptr, q);
    return {output, mask};
}



template<typename T>
std::vector<torch::Tensor>
dropout_backward(float ratio,
                 uint32_t dim,
                 int bsz,
                 torch::Tensor &vals,
                 torch::Tensor &mask,
                 bool in_place) {
    CHECK_INPUT(vals);
    CHECK_INPUT(mask);
    sycl::queue *q = ::SyclContext::Instance().GetCurrentStream();
    Dropout<T> _dropout = Dropout<T>(typename Dropout<T>::Config(ratio, dim));
    uint8_t *mask_ptr = (uint8_t *)mask.data_ptr();
    _dropout.SetMask(mask_ptr);
    if(in_place){
        T *d_input_ptr = (T *)vals.data_ptr();
        _dropout.Backward(bsz, d_input_ptr, q);
        return {vals};
    } else {
        auto output = torch::empty_like(vals);
        const T *d_input_ptr = (const T *)vals.data_ptr();
        T *d_output_ptr = (T *)output.data_ptr();
        _dropout.Backward(bsz, d_output_ptr, d_input_ptr, q);
        return {output};
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_fp32", &dropout_forward<float>,
          "DROPOUT forward with fp32 (DPCPP)");
    m.def("forward_fp16", &dropout_forward<sycl::half>,
          "DROPOUT forward with fp16 (DPCPP)");
    m.def("forward_bf16", &dropout_forward<bf16>,
          "DROPOUT forward with bf16 (DPCPP)");
    m.def("forward_with_bias_fp32", &dropout_forward_with_bias<float>,
          "DROPOUT forward with bias with fp32 (DPCPP)");
    m.def("forward_with_bias_fp16", &dropout_forward_with_bias<sycl::half>,
          "DROPOUT forward with bias with fp16 (DPCPP)");
    m.def("forward_with_bias_bf16", &dropout_forward_with_bias<bf16>,
          "DROPOUT forward with bias with bf16 (DPCPP)");
    m.def("backward_fp32", &dropout_backward<float>,
          "DROPOUT backward with fp32 (DPCPP)");
    m.def("backward_fp16", &dropout_backward<sycl::half>,
          "DROPOUT backward with fp16 (DPCPP)");
    m.def("backward_bf16", &dropout_backward<bf16>,
          "DROPOUT backward with bf16 (DPCPP)");
}
