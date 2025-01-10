#include "gtest/gtest.h"
#include "search.h"
#include <Windows.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <random>

void CreateZeroedFile(const std::wstring& filename, size_t fileSize) {
    HANDLE hFile = CreateFileW(filename.c_str(), GENERIC_WRITE, 0, NULL,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    ASSERT_NE(hFile, INVALID_HANDLE_VALUE) << "Failed to create file: " << filename;

    LARGE_INTEGER li;
    li.QuadPart = fileSize;
    ASSERT_TRUE(SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) << "Failed to set file pointer";
    ASSERT_TRUE(SetEndOfFile(hFile)) << "Failed to set end of file";

    const size_t chunkSize = 1024 * 1024; // 1MB chunks
    std::vector<char> buffer(chunkSize, 0);
    DWORD bytesWritten;
    for (size_t i = 0; i < fileSize / chunkSize; ++i) {
        ASSERT_TRUE(WriteFile(hFile, buffer.data(), chunkSize, &bytesWritten, NULL))
            << "Failed to write to file";
    }
    size_t remainingBytes = fileSize % chunkSize;
    if (remainingBytes > 0) {
        ASSERT_TRUE(WriteFile(hFile, buffer.data(), remainingBytes, &bytesWritten, NULL))
            << "Failed to write to file";
    }

    CloseHandle(hFile);
}

TEST(SearchTest, DllLoad) {
    HMODULE hDll = LoadLibrary(L"Search.dll");

    ASSERT_NE(hDll, (HMODULE)NULL) << "Failed to load Search.dll";

    FARPROC pFunction = GetProcAddress(hDll, "Search_Initialize");
    ASSERT_NE(pFunction, (FARPROC)NULL) << "Failed to get address of Search_Initialize";

    FreeLibrary(hDll);
}

TEST(SearchTest, SearchPatternIn32MBFile) {
    // 1. Create a 32MB file filled with zeros
    std::wstring testFilename = L"testfile_32mb.bin";
    size_t fileSize = 32 * 1024 * 1024; // 32MB
    CreateZeroedFile(testFilename, fileSize);

    // 2. Choose a random offset to insert the pattern
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(0, fileSize - 10); // Ensure pattern fits
    size_t patternOffset = distrib(gen);

    // 3. Define the pattern
    const char* pattern = "MyPattern";
    size_t patternLength = strlen(pattern);

    // 4. Insert the pattern into the file
    HANDLE hFile = CreateFileW(testFilename.c_str(), GENERIC_WRITE, 0, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    ASSERT_NE(hFile, INVALID_HANDLE_VALUE) << "Failed to open file for writing";

    LARGE_INTEGER li;
    li.QuadPart = patternOffset;
    ASSERT_TRUE(SetFilePointerEx(hFile, li, NULL, FILE_BEGIN)) << "Failed to set file pointer";

    DWORD bytesWritten;
    ASSERT_TRUE(WriteFile(hFile, pattern, patternLength, &bytesWritten, NULL))
        << "Failed to write pattern to file";

    CloseHandle(hFile);

    // 5. Initialize Search and load the file
    ASSERT_TRUE(Search_Initialize());
    ASSERT_TRUE(Search_LoadFile(testFilename.c_str()));

    // 6. Search for the pattern
    unsigned long long results[1]; // We expect at most one result
    size_t numResults;
    ASSERT_TRUE(Search_SearchPattern(pattern, patternLength, results, &numResults));

    // 7. Verify the results
    ASSERT_EQ(numResults, 1) << "Pattern not found or found multiple times";
    ASSERT_EQ(results[0], patternOffset) << "Incorrect pattern offset";

    // 8. Clean up
    Search_UnloadFile();
    DeleteFileW(testFilename.c_str());
    Search_Cleanup();
}
