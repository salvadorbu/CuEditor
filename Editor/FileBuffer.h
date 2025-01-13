#pragma once
#include <qstring.h>
#include <Windows.h>
#include <vector>

class FileBuffer {
public:
    static constexpr size_t CHUNK_SIZE = 1024 * 1024;

    FileBuffer();
    ~FileBuffer();

    bool load(const QString& filePath);
    QString readChunk(size_t offset, size_t size);
    size_t size() const { return m_fileSize; }
    void unload();

private:
    bool loadSmallFile();
    bool loadLargeFile();

    HANDLE m_hFile;
    HANDLE m_hMapping;
    char* m_fileData;
    std::vector<char> m_data;
    size_t m_fileSize;
    bool m_isLarge;
};
