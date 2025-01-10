#pragma once

#ifdef SEARCH_EXPORTS
#define SEARCH_API __declspec(dllexport)
#else
#define SEARCH_API __declspec(dllimport)
#endif

typedef unsigned long long uint64_t;

#ifdef __cplusplus
extern "C" {
#endif

    SEARCH_API bool Search_Initialize();
    SEARCH_API bool Search_LoadFile(const wchar_t* filepath);
    SEARCH_API void Search_UnloadFile();
    SEARCH_API bool Search_SearchPattern(const char* pattern, size_t patternLength,
        uint64_t* results, size_t* numResults);
    SEARCH_API void Search_Cleanup();

#ifdef __cplusplus
}
#endif
