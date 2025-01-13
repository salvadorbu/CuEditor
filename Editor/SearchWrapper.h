#pragma once
#include <qstring.h>
#include <qvector.h>
#include "search.h"

class SearchWrapper {
public:
    SearchWrapper() {
        m_initialized = Search_Initialize();
    }

    ~SearchWrapper() {
        Search_Cleanup();
    }

    bool isInitialized() const { return m_initialized; }

    bool loadFile(const QString& filePath) {
        if (!m_initialized) return false;
        return Search_LoadFile(reinterpret_cast<const wchar_t*>(filePath.utf16()));
    }

    void unloadFile() {
        Search_UnloadFile();
    }

    bool search(const QString& pattern, QVector<uint64_t>& results) {
        if (!m_initialized) return false;

        QByteArray patternUtf8 = pattern.toUtf8();
        size_t numResults = 0;

        results.resize(1000);

        bool success = Search_SearchPattern(
            patternUtf8.constData(),
            patternUtf8.size(),
            results.data(),
            &numResults
        );

        if (success) {
            results.resize(numResults);
            return true;
        }

        results.clear();
        return false;
    }

private:
    bool m_initialized;
};
