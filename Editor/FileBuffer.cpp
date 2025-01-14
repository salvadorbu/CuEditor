#define NOMINMAX
#include "FileBuffer.h"

FileBuffer::FileBuffer()
    : m_hFile(INVALID_HANDLE_VALUE)
    , m_hMapping(NULL)
    , m_fileSize(0)
    , m_isLarge(false)
{
}

FileBuffer::~FileBuffer()
{
    unload();
}

bool FileBuffer::load(const QString& filePath)
{
    unload();

    m_hFile = CreateFileW(
        (LPCWSTR)filePath.utf16(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (m_hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(m_hFile, &fileSize)) {
        CloseHandle(m_hFile);
        m_hFile = INVALID_HANDLE_VALUE;
        return false;
    }
    m_fileSize = fileSize.QuadPart;
    m_isLarge = m_fileSize > CHUNK_SIZE;

    if (!m_isLarge) {
        return loadSmallFile();
    }

    // For large files, just create the file mapping object
    m_hMapping = CreateFileMappingW(
        m_hFile,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    return m_hMapping != NULL;
}

bool FileBuffer::loadSmallFile()
{
    m_data.resize(m_fileSize);

    DWORD bytesRead;
    if (!ReadFile(m_hFile, m_data.data(), static_cast<DWORD>(m_fileSize), &bytesRead, NULL) ||
        bytesRead != m_fileSize) {
        return false;
    }

    CloseHandle(m_hFile);
    m_hFile = INVALID_HANDLE_VALUE;

    return true;
}

QString FileBuffer::readChunk(size_t offset, size_t size)
{
    if (offset >= m_fileSize) {
        return QString();
    }

    size = std::min(size, m_fileSize - offset);
    if (size == 0) {
        return QString();
    }

    if (!m_isLarge) {
        return QString::fromUtf8(m_data.data() + offset, size);
    }

    DWORD offsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
    DWORD offsetLow = static_cast<DWORD>(offset & 0xFFFFFFFF);

    char* data = static_cast<char*>(MapViewOfFile(
        m_hMapping,
        FILE_MAP_READ,
        offsetHigh,
        offsetLow,
        size
    ));

    if (!data) {
        return QString();
    }

    QString text = QString::fromUtf8(data, size);
    UnmapViewOfFile(data);
    return text;
}

void FileBuffer::unload()
{
    m_data.clear();

    if (m_hMapping) {
        CloseHandle(m_hMapping);
        m_hMapping = NULL;
    }

    if (m_hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(m_hFile);
        m_hFile = INVALID_HANDLE_VALUE;
    }

    m_fileSize = 0;
    m_isLarge = false;
}
