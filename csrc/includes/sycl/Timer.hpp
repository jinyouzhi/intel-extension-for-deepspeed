
#ifndef __TIMER_H__
#define __TIMER_H__

#include <CL/sycl.hpp>
#include <chrono>
#include <chrono>

class GPUTimer {
    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

public:
    GPUTimer()
    {
        /*
        DPCT1026:0: The call to cudaEventCreate was removed, because this call is
         * redundant in DPC++.
        */
        /*
        DPCT1026:1: The call to cudaEventCreate was removed, because this call is
         * redundant in DPC++.
        */
    }
    ~GPUTimer()
    {
        /*
        DPCT1026:2: The call to cudaEventDestroy was removed, because this call is
         * redundant in DPC++.
        */
        /*
        DPCT1026:3: The call to cudaEventDestroy was removed, because this call is
         * redundant in DPC++.
        */
    }
    /*
    DPCT1012:4: Detected kernel execution time measurement pattern and generated an initial
     * code for time measurements in SYCL. You can change the way time is measured depending on your
     * goals.
    */
    inline void Record() { start_ct1 = std::chrono::steady_clock::now(); }
    inline void Elapsed(float& time_elapsed)
    {
        /*
        DPCT1012:5: Detected kernel execution time measurement pattern and generated an
         * initial code for time measurements in SYCL. You can change the way time is measured
         * depending on your goals.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        stop.wait_and_throw();
        time_elapsed = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    }
};

class CPUTimer {
    std::chrono::high_resolution_clock::time_point start;

public:
    CPUTimer() : start(std::chrono::high_resolution_clock::now()) {}
    inline void Reset() { start = std::chrono::high_resolution_clock::now(); }
    inline float Elapsed()
    {
        auto temp = start;
        start = std::chrono::high_resolution_clock::now();
        return (float)(std::chrono::duration_cast<std::chrono::microseconds>(start - temp).count() /
                       1e3);
    }
};

#endif
