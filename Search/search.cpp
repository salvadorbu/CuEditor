#include "search.h"
#include "search_kernel.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <Windows.h>

namespace {
    constexpr size_t CHUNK_SIZE = 1024 * 1024 * 32; // 32 MB
    HANDLE g_hFile = INVALID_HANDLE_VALUE;
    HANDLE g_hMapping = NULL;
    char* g_fileData = nullptr;
    size_t g_fileSize = 0;

    char* g_d_fileData = nullptr;
    char* g_d_pattern = nullptr;
    unsigned long long* g_d_results = nullptr;
    size_t g_maxResults = 1000000;
}

extern "C" {
    bool LaunchSearchKernel(
        const char* d_fileData, size_t chunkSize,
        const char* d_pattern, size_t patternLength,
        unsigned long long* d_results, size_t totalResults,
        size_t* d_numResults,
        size_t chunkOffset
    );
}

bool Search_Initialize() {
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    cudaStatus = cudaMalloc(&g_d_results, g_maxResults * sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    return true;
}

bool Search_LoadFile(const wchar_t* filepath) {
    Search_UnloadFile();

    g_hFile = CreateFileW(filepath, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER fileSize64;
    if (!GetFileSizeEx(g_hFile, &fileSize64)) {
        CloseHandle(g_hFile);
        return false;
    }
    g_fileSize = static_cast<size_t>(fileSize64.QuadPart);

    g_hMapping = CreateFileMappingW(g_hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!g_hMapping) {
        CloseHandle(g_hFile);
        return false;
    }

    g_fileData = static_cast<char*>(MapViewOfFile(g_hMapping, FILE_MAP_READ, 0, 0, 0));
    if (!g_fileData) {
        CloseHandle(g_hMapping);
        CloseHandle(g_hFile);
        return false;
    }

    return true;
}

bool Search_SearchPattern(const char* pattern, size_t patternLength,
    unsigned long long* results, size_t* numResults) {
    if (!g_fileData || !pattern || patternLength == 0) {
        return false;
    }

    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&g_d_pattern, patternLength);
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    cudaStatus = cudaMemcpy(g_d_pattern, pattern, patternLength, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cudaFree(g_d_pattern);
        return false;
    }

    size_t totalResults = 0;
    size_t* d_numResults;
    cudaStatus = cudaMalloc(&d_numResults, sizeof(size_t));
    if (cudaStatus != cudaSuccess) {
        cudaFree(g_d_pattern);
        return false;
    }

    constexpr size_t CHUNK_SIZE = 1024 * 1024 * 32; // 32MB
    for (size_t offset = 0; offset < g_fileSize; offset += CHUNK_SIZE) {
        size_t chunkSize = min(CHUNK_SIZE, g_fileSize - offset);

        cudaStatus = cudaMalloc(&g_d_fileData, chunkSize);
        if (cudaStatus != cudaSuccess) {
            cudaFree(g_d_pattern);
            cudaFree(d_numResults);
            return false;
        }

        cudaStatus = cudaMemcpy(g_d_fileData, g_fileData + offset, chunkSize, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            cudaFree(g_d_fileData);
            cudaFree(g_d_pattern);
            cudaFree(d_numResults);
            return false;
        }

        size_t remainingResults = g_maxResults - totalResults;
        if (!LaunchSearchKernel(
            g_d_fileData, chunkSize,
            g_d_pattern, patternLength,
            g_d_results + totalResults, remainingResults,
            d_numResults,
            offset
        )) {
            cudaFree(g_d_fileData);
            cudaFree(g_d_pattern);
            cudaFree(d_numResults);
            return false;
        }

        size_t chunkResults;
        cudaStatus = cudaMemcpy(&chunkResults, d_numResults, sizeof(size_t), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            cudaFree(g_d_fileData);
            cudaFree(g_d_pattern);
            cudaFree(d_numResults);
            return false;
        }

        if (chunkResults > g_maxResults - totalResults) {
            totalResults = g_maxResults;
            break;
        }

        totalResults += chunkResults;

        cudaFree(g_d_fileData);

        if (totalResults >= g_maxResults) {
            break;
        }
    }

    cudaStatus = cudaMemcpy(results, g_d_results,
        totalResults * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);
    cudaFree(g_d_pattern);
    cudaFree(d_numResults);

    if (cudaStatus != cudaSuccess) {
        return false;
    }

    *numResults = totalResults;
    return true;
}

void Search_UnloadFile() {
    if (g_fileData) {
        UnmapViewOfFile(g_fileData);
        g_fileData = nullptr;
    }
    if (g_hMapping) {
        CloseHandle(g_hMapping);
        g_hMapping = NULL;
    }
    if (g_hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(g_hFile);
        g_hFile = INVALID_HANDLE_VALUE;
    }
}

void Search_Cleanup() {
    Search_UnloadFile();
    if (g_d_results) {
        cudaFree(g_d_results);
        g_d_results = nullptr;
    }
}

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
