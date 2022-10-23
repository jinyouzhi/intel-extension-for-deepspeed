#include <CL/sycl.hpp>
#include "sycl/general_kernels.hpp"

using namespace cl::sycl;

constexpr int MAX_SG_NUM = 32;
constexpr int MAX_SG_NUM1 = MAX_SG_NUM + 1;
template <typename T>
void column_sum_reduce(const T* inp,
                       T* out,
                       int rows,
                       int width,
                       nd_item<3> item_ct1,
                       float *tile)
{
    group<3> b = item_ct1.get_group();
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    int y_stride = width * MAX_SG_NUM;

    float localSum = 0;

    // Loop across matrix height
    if (idx < width) {
        int offset = item_ct1.get_local_id(1) * width + idx;
        for (int r = item_ct1.get_local_id(1); r < rows; r += MAX_SG_NUM) {
            localSum += (float)inp[offset];
            offset += y_stride;
        }
    }

    tile[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = localSum;

    /*
    DPCT1065:0: Consider replacing nd_item::barrier() with
     * nd_item::barrier(access::fence_space::local_space) for better performance, if
     * there is no access to global memory.
    */
    item_ct1.barrier();

    // Sum the shared buffer.
    float sum = tile[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__
    /*
    DPCT1065:1: Consider replacing nd_item::barrier() with
     * nd_item::barrier(access::fence_space::local_space) for better performance, if
     * there is no access to global memory.
    */
    item_ct1.barrier();
#endif

    for (int i = 1; i < MAX_SG_NUM; i <<= 1) {
        sum += sg.shuffle_down(sum, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        int pos = item_ct1.get_group(2) * MAX_SG_NUM + item_ct1.get_local_id(1);
        if (pos < width) out[pos] = sum;
    }
}

template<>
void column_sum_reduce(const bf16* inp,
                       bf16* out,
                       int rows,
                       int width,
                       nd_item<3> item_ct1,
                       float *tile)
{
    group<3> b = item_ct1.get_group();
    sub_group sg = item_ct1.get_sub_group();

    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    int y_stride = width * MAX_SG_NUM;

    float localSum = 0;
    ushort *inp_cast = (ushort *)inp;
    ushort *out_cast = (ushort *)out;
    // Loop across matrix height
    if (idx < width) {
        int offset = item_ct1.get_local_id(1) * width + idx;
        for (int r = item_ct1.get_local_id(1); r < rows; r += MAX_SG_NUM) {
            localSum += bf16::to_float(inp_cast[offset]);
            offset += y_stride;
        }
    }

    tile[item_ct1.get_local_id(2) * MAX_SG_NUM1 + item_ct1.get_local_id(1)] = localSum;

    /*
    DPCT1065:0: Consider replacing nd_item::barrier() with
     * nd_item::barrier(access::fence_space::local_space) for better performance, if
     * there is no access to global memory.
    */
    item_ct1.barrier();

    // Sum the shared buffer.
    float sum = tile[item_ct1.get_local_id(1) * MAX_SG_NUM1 + item_ct1.get_local_id(2)];

#ifndef __STOCHASTIC_MODE__
    /*
    DPCT1065:1: Consider replacing nd_item::barrier() with
     * nd_item::barrier(access::fence_space::local_space) for better performance, if
     * there is no access to global memory.
    */
    item_ct1.barrier();
#endif

    for (int i = 1; i < MAX_SG_NUM; i <<= 1) {
        sum += sg.shuffle_down(sum, i);
    }

    if (item_ct1.get_local_id(2) == 0) {
        int pos = item_ct1.get_group(2) * MAX_SG_NUM + item_ct1.get_local_id(1);
        if (pos < width) out_cast[pos] = bf16::from_float(sum);
    }
}

template <typename T>
void launch_fuse_transpose_bias_kernel(const T* inp,
                                       T* out,
                                       int rows,
                                       int cols,
                                       queue* stream) {
    range<3> grid_dim(1, 1, (cols - 1) / MAX_SG_NUM + 1);
    range<3> block_dim(1, MAX_SG_NUM, MAX_SG_NUM);

    stream->submit([&](handler& cgh) {
        accessor<float, 2, access_mode::read_write, access::target::local>
            tile(range<2>(MAX_SG_NUM, MAX_SG_NUM1), cgh);
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) [[intel::reqd_sub_group_size(MAX_SG_NUM)]] {
                             column_sum_reduce<T>(inp,
                                                  out,
                                                  rows,
                                                  cols,
                                                  item_ct1,
                                                  tile.get_pointer());
                         });
    });
}

template
void launch_fuse_transpose_bias_kernel<float>(const float* inp,
                                              float* out,
                                              int rows,
                                              int cols,
                                              queue* stream);
template
void launch_fuse_transpose_bias_kernel<bf16>(const bf16* inp,
                                             bf16* out,
                                             int rows,
                                             int cols,
                                             queue* stream);
template
void launch_fuse_transpose_bias_kernel<half>(const half* inp,
                                             half* out,
                                             int rows,
                                             int cols,
                                             queue* stream);

void fused_add2_kernel(const int N,
                       float* out,
                       const float* inp1,
                       const float* inp2,
                       nd_item<3> item_ct1)
{
    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    float4* out_4 = reinterpret_cast<float4*>(out);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 val;
        float4 inp1_reg = inp1_4[j];
        float4 inp2_reg = inp2_4[j];

        val.x() = inp1_reg.x() + inp2_reg.x();
        val.y() = inp1_reg.y() + inp2_reg.y();
        val.z() = inp1_reg.z() + inp2_reg.z();
        val.w() = inp1_reg.w() + inp2_reg.w();

        out_4[j] = val;
    }
}

void fused_add2_kernel(const int N,
                       bf16* out,
                       const bf16* inp1,
                       const bf16* inp2,
                       nd_item<3> item_ct1)
{
    const ushort4* inp1_cast = reinterpret_cast<const ushort4*>(inp1);
    const ushort4* inp2_cast = reinterpret_cast<const ushort4*>(inp2);
    ushort4* out_cast = reinterpret_cast<ushort4*>(out);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        float4 val;
        float4 inp1_reg = {
            bf16::to_float(inp1_cast[j].x()),
            bf16::to_float(inp1_cast[j].y()),
            bf16::to_float(inp1_cast[j].z()),
            bf16::to_float(inp1_cast[j].w())
        };
        float4 inp2_reg = {
            bf16::to_float(inp2_cast[j].x()),
            bf16::to_float(inp2_cast[j].y()),
            bf16::to_float(inp2_cast[j].z()),
            bf16::to_float(inp2_cast[j].w())
        };

        val.x() = inp1_reg.x() + inp2_reg.x();
        val.y() = inp1_reg.y() + inp2_reg.y();
        val.z() = inp1_reg.z() + inp2_reg.z();
        val.w() = inp1_reg.w() + inp2_reg.w();

        out_cast[j] = {
            bf16::from_float(val.x()),
            bf16::from_float(val.y()),
            bf16::from_float(val.z()),
            bf16::from_float(val.w())
        };
    }
}

void fused_add2_kernel(const int N,
                       half* out,
                       const half* inp1,
                       const half* inp2,
                       nd_item<3> item_ct1)
{
    float2 inp1_4;
    float2 inp2_4;

    half2* inp1_h = reinterpret_cast<half2*>(&inp1_4);
    half2* inp2_h = reinterpret_cast<half2*>(&inp2_4);

    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);

    DPCPP_1D_KERNEL_LOOP(j, N)
    {
        inp1_4 = inp1_arr[j];
        inp2_4 = inp2_arr[j];

        float2 inp1_h_f_0 = inp1_h[0].convert<float, rounding_mode::automatic>();
        float2 inp1_h_f_1 = inp1_h[1].convert<float, rounding_mode::automatic>();

        float2 inp2_h_f_0 = inp2_h[0].convert<float, rounding_mode::automatic>();
        float2 inp2_h_f_1 = inp2_h[1].convert<float, rounding_mode::automatic>();

        inp1_h_f_0.x() += inp2_h_f_0.x();
        inp1_h_f_0.y() += inp2_h_f_0.y();
        inp1_h_f_1.x() += inp2_h_f_1.x();
        inp1_h_f_1.y() += inp2_h_f_1.y();

        float2 val_f;
        half2* val_h = reinterpret_cast<half2*>(&val_f);

        val_h[0] = inp1_h_f_0.convert<half, rounding_mode::rte>();
        val_h[1] = inp1_h_f_1.convert<half, rounding_mode::rte>();

        float2* out_4 = reinterpret_cast<float2*>(out);
        out_4[j] = val_f;
    }
}

template <typename T>
void launch_fused_add2(T* out,
                       const T* inp1,
                       const T* inp2,
                       int batch_size,
                       int seq_length,
                       int hidden_dim,
                       queue* stream)
{
    int total_count = batch_size * seq_length * hidden_dim / 4;
    range<3> grid_dim =
        range<3>(1, 1, DS_GET_BLOCKS(total_count));  //(batch_size * seq_length);

    range<3> block_dim = range<3>(1, 1, DS_CUDA_NUM_THREADS);  //(hidden_dim / 4);
    /*
    DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) {
                             fused_add2_kernel(total_count, out, inp1, inp2, item_ct1);
                         });
    });
}

template
void launch_fused_add2<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              int batch_size,
                              int seq_length,
                              int hidden_dim,
                              queue* stream);
template
void launch_fused_add2<bf16>(bf16* out,
                             const bf16* inp1,
                             const bf16* inp2,
                             int batch_size,
                             int seq_length,
                             int hidden_dim,
                             queue* stream);
template
void launch_fused_add2<half>(half* out,
                             const half* inp1,
                             const half* inp2,
                             int batch_size,
                             int seq_length,
                             int hidden_dim,
                             queue* stream);

void fused_add3_kernel(float* out,
                       const float* inp1,
                       const float* inp2,
                       const float* inp3,
                       int size,
                       int row_stride,
                       nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);

    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    const float4* inp3_4 = reinterpret_cast<const float4*>(inp3);

    float4* out_4 = reinterpret_cast<float4*>(out);

    float4 val;
    float4 inp1_reg = inp1_4[row * row_stride + id];
    float4 inp2_reg = inp2_4[row * row_stride + id];
    float4 inp3_reg = inp3_4[row * row_stride + id];

    val.x() = inp1_reg.x() + inp2_reg.x() + inp3_reg.x();
    val.y() = inp1_reg.y() + inp2_reg.y() + inp3_reg.y();
    val.z() = inp1_reg.z() + inp2_reg.z() + inp3_reg.z();
    val.w() = inp1_reg.w() + inp2_reg.w() + inp3_reg.w();

    out_4[row * row_stride + id] = val;
}

void fused_add3_kernel(half* out,
                       const half* inp1,
                       const half* inp2,
                       const half* inp3,
                       int size,
                       int row_stride,
                       nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);
    const float2* inp3_arr = reinterpret_cast<const float2*>(inp3);

    float2 inp1_4 = inp1_arr[row * row_stride + id];
    float2 inp2_4 = inp2_arr[row * row_stride + id];
    float2 inp3_4 = inp3_arr[row * row_stride + id];

    half2* inp1_h = reinterpret_cast<half2*>(&inp1_4);
    half2* inp2_h = reinterpret_cast<half2*>(&inp2_4);
    half2* inp3_h = reinterpret_cast<half2*>(&inp3_4);

    float2 inp1_h_f_0 = inp1_h[0].convert<float, rounding_mode::automatic>();
    float2 inp1_h_f_1 = inp1_h[1].convert<float, rounding_mode::automatic>();

    float2 inp2_h_f_0 = inp2_h[0].convert<float, rounding_mode::automatic>();
    float2 inp2_h_f_1 = inp2_h[1].convert<float, rounding_mode::automatic>();

    float2 inp3_h_f_0 = inp3_h[0].convert<float, rounding_mode::automatic>();
    float2 inp3_h_f_1 = inp3_h[1].convert<float, rounding_mode::automatic>();

    inp1_h_f_0.x() += (inp2_h_f_0.x() + inp3_h_f_0.x());
    inp1_h_f_0.y() += (inp2_h_f_0.y() + inp3_h_f_0.y());
    inp1_h_f_1.x() += (inp2_h_f_1.x() + inp3_h_f_1.x());
    inp1_h_f_1.y() += (inp2_h_f_1.y() + inp3_h_f_1.y());

    float2 val_f;
    half2* val_h = reinterpret_cast<half2*>(&val_f);

    val_h[0] = inp1_h_f_0.convert<half, rounding_mode::rte>();
    val_h[1] = inp1_h_f_1.convert<half, rounding_mode::rte>();

    float2* out_4 = reinterpret_cast<float2*>(out);
    out_4[row * row_stride + id] = val_f;
}

template <>
void launch_fused_add3<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              const float* inp3,
                              int batch_size,
                              int seq_length,
                              int hidden_size,
                              queue* stream)
{
    range<3> grid_dim(1, 1, batch_size * seq_length);
    range<3> block_dim(1, 1, hidden_size / 4);

    /*
    DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) {
                             fused_add3_kernel(out,
                                               inp1,
                                               inp2,
                                               inp3,
                                               (batch_size * seq_length * hidden_size),
                                               hidden_size / 4,
                                               item_ct1);
                         });
    });
}

template <>
void launch_fused_add3<half>(half* out,
                             const half* inp1,
                             const half* inp2,
                             const half* inp3,
                             int batch_size,
                             int seq_length,
                             int hidden_size,
                             queue* stream)
{
    range<3> grid_dim(1, 1, batch_size * seq_length);

    range<3> block_dim(1, 1, hidden_size / 4);

    /*
    DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) {
                             fused_add3_kernel(out,
                                               inp1,
                                               inp2,
                                               inp3,
                                               (batch_size * seq_length * hidden_size),
                                               hidden_size / 4,
                                               item_ct1);
                         });
    });
}

void fused_add4_kernel(float* out,
                       const float* inp1,
                       const float* inp2,
                       const float* inp3,
                       const float* inp4,
                       int size,
                       int row_stride,
                       nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);

    const float4* inp1_4 = reinterpret_cast<const float4*>(inp1);
    const float4* inp2_4 = reinterpret_cast<const float4*>(inp2);
    const float4* inp3_4 = reinterpret_cast<const float4*>(inp3);
    const float4* inp4_4 = reinterpret_cast<const float4*>(inp4);
    float4* out_4 = reinterpret_cast<float4*>(out);

    float4 val;
    float4 inp1_reg = inp1_4[row * row_stride + id];
    float4 inp2_reg = inp2_4[row * row_stride + id];
    float4 inp3_reg = inp3_4[row * row_stride + id];
    float4 inp4_reg = inp4_4[row * row_stride + id];

    val.x() = inp1_reg.x() + inp2_reg.x() + inp3_reg.x() + inp4_reg.x();
    val.y() = inp1_reg.y() + inp2_reg.y() + inp3_reg.y() + inp4_reg.y();
    val.z() = inp1_reg.z() + inp2_reg.z() + inp3_reg.z() + inp4_reg.z();
    val.w() = inp1_reg.w() + inp2_reg.w() + inp3_reg.w() + inp4_reg.w();

    out_4[row * row_stride + id] = val;
}

void fused_add4_kernel(half* out,
                       const half* inp1,
                       const half* inp2,
                       const half* inp3,
                       const half* inp4,
                       int size,
                       int row_stride,
                       nd_item<3> item_ct1)
{
    int row = item_ct1.get_group(2);
    int id = item_ct1.get_local_id(2);
    const float2* inp1_arr = reinterpret_cast<const float2*>(inp1);
    const float2* inp2_arr = reinterpret_cast<const float2*>(inp2);
    const float2* inp3_arr = reinterpret_cast<const float2*>(inp3);
    const float2* inp4_arr = reinterpret_cast<const float2*>(inp4);

    float2 inp1_4 = inp1_arr[row * row_stride + id];
    float2 inp2_4 = inp2_arr[row * row_stride + id];
    float2 inp3_4 = inp3_arr[row * row_stride + id];
    float2 inp4_4 = inp4_arr[row * row_stride + id];

    half2* inp1_h = reinterpret_cast<half2*>(&inp1_4);
    half2* inp2_h = reinterpret_cast<half2*>(&inp2_4);
    half2* inp3_h = reinterpret_cast<half2*>(&inp3_4);
    half2* inp4_h = reinterpret_cast<half2*>(&inp4_4);

    float2 inp1_h_f_0 = inp1_h[0].convert<float, rounding_mode::automatic>();
    float2 inp1_h_f_1 = inp1_h[1].convert<float, rounding_mode::automatic>();

    float2 inp2_h_f_0 = inp2_h[0].convert<float, rounding_mode::automatic>();
    float2 inp2_h_f_1 = inp2_h[1].convert<float, rounding_mode::automatic>();

    float2 inp3_h_f_0 = inp3_h[0].convert<float, rounding_mode::automatic>();
    float2 inp3_h_f_1 = inp3_h[1].convert<float, rounding_mode::automatic>();

    float2 inp4_h_f_0 = inp4_h[0].convert<float, rounding_mode::automatic>();
    float2 inp4_h_f_1 = inp4_h[1].convert<float, rounding_mode::automatic>();

    inp1_h_f_0.x() += (inp2_h_f_0.x() + inp3_h_f_0.x() + inp4_h_f_0.x());
    inp1_h_f_0.y() += (inp2_h_f_0.y() + inp3_h_f_0.y() + inp4_h_f_0.y());
    inp1_h_f_1.x() += (inp2_h_f_1.x() + inp3_h_f_1.x() + inp4_h_f_1.x());
    inp1_h_f_1.y() += (inp2_h_f_1.y() + inp3_h_f_1.y() + inp4_h_f_1.y());

    float2 val_f;
    half2* val_h = reinterpret_cast<half2*>(&val_f);

    val_h[0] = inp1_h_f_0.convert<half, rounding_mode::rte>();
    val_h[1] = inp1_h_f_1.convert<half, rounding_mode::rte>();

    float2* out_4 = reinterpret_cast<float2*>(out);
    out_4[row * row_stride + id] = val_f;
}

template <>
void launch_fused_add4<float>(float* out,
                              const float* inp1,
                              const float* inp2,
                              const float* inp3,
                              const float* inp4,
                              int batch_size,
                              int seq_length,
                              int hidden_size,
                              queue* stream)
{
    range<3> grid_dim(1, 1, batch_size * seq_length);

    range<3> block_dim(1, 1, hidden_size / 4);

    /*
    DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) {
                             fused_add4_kernel(out,
                                               inp1,
                                               inp2,
                                               inp3,
                                               inp4,
                                               (batch_size * seq_length * hidden_size),
                                               hidden_size / 4,
                                               item_ct1);
                         });
    });
}

template <>
void launch_fused_add4<half>(half* out,
                             const half* inp1,
                             const half* inp2,
                             const half* inp3,
                             const half* inp4,
                             int batch_size,
                             int seq_length,
                             int hidden_size,
                             queue* stream)
{
    range<3> grid_dim(1, 1, batch_size * seq_length);

    range<3> block_dim(1, 1, hidden_size / 4);

    /*
    DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](handler& cgh) {
        cgh.parallel_for(nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](nd_item<3> item_ct1) {
                             fused_add4_kernel(out,
                                               inp1,
                                               inp2,
                                               inp3,
                                               inp4,
                                               (batch_size * seq_length * hidden_size),
                                               hidden_size / 4,
                                               item_ct1);
                         });
    });
}
