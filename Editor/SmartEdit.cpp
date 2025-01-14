#include "SmartEdit.h"

SmartEdit::SmartEdit(QWidget* parent)
    : QPlainTextEdit(parent)
    , m_buffer(nullptr)
{
    setLineWrapMode(QPlainTextEdit::WidgetWidth);
    setUndoRedoEnabled(true);
}

void SmartEdit::setBuffer(std::shared_ptr<FileBuffer> buffer) {
    m_buffer = buffer;
    if (!m_buffer) {
        clear();
        return;
    }

    QString content = m_buffer->readChunk(0, m_buffer->size());
    setPlainText(content);
}

std::shared_ptr<FileBuffer> SmartEdit::buffer() const {
    return m_buffer;
}
