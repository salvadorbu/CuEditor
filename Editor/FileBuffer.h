#pragma once
#include <qstring.h>
#include <Windows.h>
#include <vector>

class FileBuffer {
public:
    static constexpr size_t CHUNK_SIZE = 1024 * 1024;

    struct ChunkData {
        QByteArray data;
        size_t startOffset;
        size_t size;
    };

    FileBuffer();
    ~FileBuffer();

    bool load(const QString& filePath);
    QString readChunk(size_t offset, size_t size);
    ChunkData readChunkData(size_t offset, size_t size) const;
    size_t size() const { return m_fileSize; }
    void unload();

private:
    bool loadSmallFile();
    bool loadLargeFile();

    HANDLE m_hFile;
    HANDLE m_hMapping;
    std::vector<char> m_data;
    size_t m_fileSize;
    bool m_isLarge;
};
