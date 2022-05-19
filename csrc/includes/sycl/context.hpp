#pragma once

#include <CL/sycl.hpp>
#include <core/Stream.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>
#include <ext/intel/experimental/bfloat16.hpp>

using bf16 = sycl::ext::intel::experimental::bfloat16;

#define WARP_SIZE 32
#define ONEMKL_OP_T oneapi::mkl::transpose::trans
#define ONEMKL_OP_N oneapi::mkl::transpose::nontrans

#define DPCPP_1D_KERNEL_LOOP(i, n)                                      \
    for (size_t (i) = item_ct1.get_global_id(2);                        \
         (i) < (n);                                                     \
         (i) += item_ct1.get_global_range(2))

#define DPCPP_2D_KERNEL_LOOP(i, n, j, m)                                \
    for (size_t i = item_ct1.get_global_id(2);                          \
         (i) < (n); (i) += item_ct1.get_global_range(2))                \
        for (size_t j = item_ct1.get_global_id(1);                      \
             (j) < (m); (j) += item_ct1.get_global_range(1))

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 262144

inline int DS_GET_BLOCKS(const int N)
{
    return (std::max)(
        (std::min)((N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS, DS_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
}

class SyclContext {
public:
    SyclContext() try : _workspace(nullptr), _seed(42), _curr_offset(0) {
        _gen = new oneapi::mkl::rng::philox4x32x10(
            xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue(), 123);
        /*
        DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You may need to
        rewrite this code.
        */
        if ((_onemklQ = &xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue(), 0) != 0) {
            auto message = std::string("Fail to create onemkl queue.");
            std::cerr << message << std::endl;
            throw std::runtime_error(message);
        }
    }
    catch (sycl::exception const& exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                << std::endl;
      std::exit(1);
    }

    virtual ~SyclContext()
    {
        _onemklQ = nullptr;
        free(_gen);
        sycl::free(_workspace, xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue());
    }

    static SyclContext& Instance()
    {
        static SyclContext _ctx;
        return _ctx;
    }

    void SetWorkSpace(void* workspace)
    {
        if (!workspace) { throw std::runtime_error("Workspace is null."); }
        _workspace = workspace;
    }

    void* GetWorkSpace() { return _workspace; }

    /*
    DPCT1050:13: The template argument of the RNG engine could not be deduced. You need to update
    this code.
    */
    //TODO:guokai random generator dpct_placeholder /*Fix the engine type manually*/*& GetRandGenerator() { return _gen; }

    sycl::queue* GetCurrentStream()
    {
        // get current pytorch stream.
        return &xpu::dpcpp::getCurrentDPCPPStream().dpcpp_queue();
    }

    sycl::queue* GetNewStream() {
        return &(xpu::dpcpp::getDPCPPStreamFromPool(true, -1).dpcpp_queue());
    }

    sycl::queue* GetOneMKLQ() { return _onemklQ; }

    std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t offset_inc)
    {
        uint64_t offset = _curr_offset;
        _curr_offset += offset_inc;
        // set _GPT_DEBUG_ and fix seed to avoid randomness
#ifdef _GPT_DEBUG_
        return std::pair<uint64_t, uint64_t>(_seed, 0);
#else
        return std::pair<uint64_t, uint64_t>(_seed, offset);
#endif
    }

    void SetSeed(uint64_t new_seed) { _seed = new_seed; }

    void TestGemmFP16(bool test_gemm, int batch_size, int seq_len, int head_num, int size_per_head)
    {
        // avoid rerun.
        if (_gemm_algos.size() > 0) return;

#if 0
        TODO Yuankun Shi
        if (test_gemm) {
            sycl::queue* Q = GetOneMKLQ();

            std::unique_ptr<GemmTest<half>> test_qkv_fw(
                new GemmTest<half>(batch_size * seq_len,      // M
                                   head_num * size_per_head,  // N
                                   head_num * size_per_head,  // K
                                   ONEMKL_OP_T,
                                   ONEMKL_OP_N,
                                   Q));

            std::unique_ptr<GemmTest<half>> test_inter(
                new GemmTest<half>(batch_size * seq_len,          // M
                                   4 * head_num * size_per_head,  // N
                                   head_num * size_per_head,      // K
                                   ONEMKL_OP_T,
                                   ONEMKL_OP_N,
                                   Q));

            std::unique_ptr<GemmTest<half>> test_output(
                new GemmTest<half>(batch_size * seq_len,          // M
                                   head_num * size_per_head,      // N
                                   4 * head_num * size_per_head,  // K
                                   ONEMKL_OP_T,
                                   ONEMKL_OP_N,
                                   Q));

            std::unique_ptr<StridedGemmTest<half>> test_attn_scores(
                new StridedGemmTest<half>(batch_size * head_num,  // batch
                                          seq_len,                // M
                                          seq_len,                // N
                                          size_per_head,          // K
                                          ONEMKL_OP_T,
                                          ONEMKL_OP_N,
                                          Q));

            std::unique_ptr<StridedGemmTest<half>> test_attn_context(
                new StridedGemmTest<half>(batch_size * head_num,  // batch
                                            size_per_head,          // M
                                            seq_len,                // N
                                            seq_len,                // K
                                            ONEMKL_OP_N,
                                            ONEMKL_OP_N,
                                            Q));

            _gemm_algos.push_back(test_qkv_fw->TestAlgo(100));
            _gemm_algos.push_back(test_inter->TestAlgo(100));
            _gemm_algos.push_back(test_output->TestAlgo(100));
            _gemm_algos.push_back(test_attn_scores->TestAlgo(100));
            _gemm_algos.push_back(test_attn_context->TestAlgo(100));
        } else {
#endif
            // Use default algo.
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
            _gemm_algos.push_back(std::array<int, 3>({99, 99, 99}));
        //}
    }

    const std::vector<std::array<int, 3>>& GetGemmAlgos() const { return _gemm_algos; }

private:
    oneapi::mkl::rng::philox4x32x10 *_gen;
    sycl::queue* _onemklQ;
    void* _workspace;
    uint64_t _seed;
    uint64_t _curr_offset;
    std::vector<std::array<int, 3>> _gemm_algos;
};
