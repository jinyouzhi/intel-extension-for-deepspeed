#include <CL/sycl.hpp>
#include "sycl/onemkl_wrappers.hpp"

int onemkl_gemm_ex(sycl::queue* handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const float alpha,
                   const float beta,
                   const float* A,
                   const float* B,
                   float* C) {
    try {
        int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        oneapi::mkl::blas::gemm(*handle,
                                transa,
                                transb,
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                lda,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:"
                  << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

int onemkl_gemm_ex(sycl::queue* handle,
                   oneapi::mkl::transpose transa,
                   oneapi::mkl::transpose transb,
                   int m,
                   int n,
                   int k,
                   const sycl::half alpha,
                   const sycl::half beta,
                   const sycl::half* A,
                   const sycl::half* B,
                   sycl::half* C) {
    try {
        int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        #if 1  // TODO gma disable temporary
        //std::cout << "transa = " << (transa == oneapi::mkl::transpose::nontrans) << std::endl
                  //<< "transb = " << (transb == oneapi::mkl::transpose::nontrans) << std::endl
                  //<< "m = " << m << std::endl
                  //<< "n = " << n << std::endl
                  //<< "k = " << k << std::endl
                  //<< "alpha = " << alpha << std::endl
                  //<< "beta = " << beta << std::endl
                  //<< "lda = " << lda << std::endl
                  //<< "ldb = " << ldb << std::endl
                  //<< "ldc = " << ldc << std::endl;
        oneapi::mkl::blas::gemm(*handle,
                                transa,
                                transb,
                                m,
                                n,
                                k,
                                alpha,
                                A,
                                lda,
                                B,
                                ldb,
                                beta,
                                C,
                                ldc);
        #endif
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:"
                  << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

int onemkl_strided_batched_gemm(sycl::queue* handle,
                                int m,
                                int n,
                                int k,
                                const float alpha,
                                const float beta,
                                const float* A,
                                const float* B,
                                float* C,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo)
{
    try {
        int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        oneapi::mkl::blas::gemm_batch(*handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      stride_A,
                                      B,
                                      ldb,
                                      stride_B,
                                      beta,
                                      C,
                                      ldc,
                                      stride_C,
                                      batch);
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << " (batch, m, n, k)" << batch
                  << " " << m << " " << n << " " << k
                  << std::endl;
        std::exit(1);
    }
}

int onemkl_strided_batched_gemm(sycl::queue* handle,
                                int m,
                                int n,
                                int k,
                                const sycl::half alpha,
                                const sycl::half beta,
                                const sycl::half* A,
                                const sycl::half* B,
                                sycl::half* C,
                                oneapi::mkl::transpose transa,
                                oneapi::mkl::transpose transb,
                                int stride_A,
                                int stride_B,
                                int stride_C,
                                int batch,
                                int algo) {
    try {
        int lda = (transa == oneapi::mkl::transpose::nontrans) ? m : k;
        int ldb = (transb == oneapi::mkl::transpose::nontrans) ? k : n;
        int ldc = m;
        #if 1   // TODO gma
        //std::cout << "transa = " << (transa == oneapi::mkl::transpose::nontrans) << std::endl
                  //<< "transb = " << (transb == oneapi::mkl::transpose::nontrans) << std::endl
                  //<< "m = " << m << std::endl
                  //<< "n = " << n << std::endl
                  //<< "k = " << k << std::endl
                  //<< "alpha = " << alpha << std::endl
                  //<< "beta = " << beta << std::endl
                  //<< "lda = " << lda << std::endl
                  //<< "ldb = " << ldb << std::endl
                  //<< "ldc = " << ldc << std::endl
                  //<< "stride_A = " << stride_A << std::endl
                  //<< "stride_B = " << stride_B << std::endl
                  //<< "stride_C = " << stride_C << std::endl
                  //<< "batch = " << batch << std::endl;
        oneapi::mkl::blas::gemm_batch(*handle,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      stride_A,
                                      B,
                                      ldb,
                                      stride_B,
                                      beta,
                                      C,
                                      ldc,
                                      stride_C,
                                      batch);
          #endif
    } catch (sycl::exception const& exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << " (batch, m, n, k)" << batch
                  << " " << m << " " << n << " " << k
                  << std::endl;
        std::exit(1);
    }
}
