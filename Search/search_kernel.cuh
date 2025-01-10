#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
//for atomicAdd
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__

__global__ void searchKernel(const char* text, size_t textLength,
    const char* pattern, size_t patternLength,
    unsigned long long* results, size_t* numResults);