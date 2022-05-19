#include <CL/sycl.hpp>
#include "sycl/custom_sycl_layers.hpp"

void param_update_kernel(const float* input,
                         sycl::half* output,
                         int size,
                         sycl::nd_item<3> item_ct1)
{
    int id = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

    if (id < size) { output[id] = (sycl::half)input[id]; }
}

void launch_param_update(const float* input, sycl::half* output, int size, sycl::queue *stream)
{
    int threads = 1024;

    sycl::range<3> grid_dim(1, 1, (size - 1) / threads + 1);
    sycl::range<3> block_dim(1, 1, threads);

    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit. To get the
     * device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.

     */
    stream->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<3>(grid_dim * block_dim, block_dim),
                         [=](sycl::nd_item<3> item_ct1) {
                             param_update_kernel(input, output, size, item_ct1);
                         });
    });
}
