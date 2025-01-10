#include "search_kernel.cuh"
#include <device_functions.h>

__global__ void searchKernel(const char* text, size_t textLength,
    const char* pattern, size_t patternLength,
    unsigned long long* results, size_t* numResults,
    size_t totalResults,
    size_t offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < textLength - patternLength + 1; i += stride) {
        bool match = true;
        for (size_t j = 0; j < patternLength; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }

        if (match) {
            size_t resultIdx = atomicAdd(numResults, 1);
            if (resultIdx < totalResults) {
                results[resultIdx] = i + offset;
            }
        }
    }
}

extern "C" bool LaunchSearchKernel(
    const char* d_fileData, size_t chunkSize,
    const char* d_pattern, size_t patternLength,
    unsigned long long* d_results, size_t totalResults,
    size_t * d_numResults,
    size_t chunkOffset
) {
    int threadsPerBlock = 256;
    int blocks = (int)((chunkSize + threadsPerBlock - 1) / threadsPerBlock);

    size_t zero = 0;
    cudaError_t cudaStatus = cudaMemcpy(d_numResults, &zero, sizeof(size_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    searchKernel << <blocks, threadsPerBlock >> > (
        d_fileData, chunkSize,
        d_pattern, patternLength,
        d_results, d_numResults,
        totalResults,
        chunkOffset
        );

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    return true;
}
