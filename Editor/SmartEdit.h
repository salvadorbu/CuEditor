#pragma once
#include <QPlainTextEdit>
#include "FileBuffer.h"
#include <memory>

class SmartEdit : public QPlainTextEdit {
    Q_OBJECT

public:
    explicit SmartEdit(QWidget* parent = nullptr);

    void setBuffer(std::shared_ptr<FileBuffer> buffer);
    std::shared_ptr<FileBuffer> buffer() const;

private:
    std::shared_ptr<FileBuffer> m_buffer;
};
