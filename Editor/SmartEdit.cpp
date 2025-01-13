#define NOMINMAX
#include "SmartEdit.h"
#include <qscrollbar.h>

/*
    -- Lazy Mode Logic --

    Text Loading:
    Dynamic 1MB chunk split into 4 subchunks of ~262KB. If user navigates to portion of text in one of outer chunks furthest
    subchunk is released and new subchunk is loaded adjacent to current subchunk. Exceptions: Viewing start/end of files.
    Challenges: vertical and horizontal lazy loading?

    Modification List:
    Text modifications should save the offset as well as changes to a modification list. If user decides to save file, the
    contents of the modification list will be written to the file. Additional feature: store modification list in revision json
    as user modifies the file in case editor exits unexpectedly.
*/

namespace {
    size_t estimateLineCount(size_t fileSize, size_t avgLineLength = 80) {
        return fileSize / avgLineLength;
    }

    size_t estimateOffsetForLine(size_t lineNumber, size_t avgLineLength = 80) {
        return lineNumber * avgLineLength;
    }
}

SmartEdit::SmartEdit(QWidget* parent)
    : QPlainTextEdit(parent)
    , m_updatingContent(false)
    , m_pendingUpdate(false)
    , m_isLazyMode(false)
{
    setUndoRedoEnabled(false);
    setLineWrapMode(QPlainTextEdit::WidgetWidth);

    connect(verticalScrollBar(), &QScrollBar::valueChanged,
        this, &SmartEdit::loadVisibleContent);
}

void SmartEdit::setBuffer(std::shared_ptr<FileBuffer> buffer)
{
    m_buffer = buffer;
    m_currentContent.clear();
    m_updatingContent = false;
    m_pendingUpdate = false;

    if (!m_buffer) {
        clear();
        m_isLazyMode = false;
        return;
    }

    m_isLazyMode = m_buffer->size() > FileBuffer::CHUNK_SIZE;

    if (m_isLazyMode) {
        size_t estimatedLines = estimateLineCount(m_buffer->size());
        verticalScrollBar()->setRange(0, estimatedLines);
        verticalScrollBar()->setValue(0);
        loadVisibleContent();
    }
    else {
        QString content = m_buffer->readChunk(0, m_buffer->size());
        setPlainText(content);
    }
}

void SmartEdit::resizeEvent(QResizeEvent* event)
{
    QPlainTextEdit::resizeEvent(event);

    // Only handle lazy loading in lazy mode
    if (!m_isLazyMode) {
        return;
    }

    if (m_updatingContent) {
        m_pendingUpdate = true;
        return;
    }

    loadVisibleContent();
}

void SmartEdit::scrollContentsBy(int dx, int dy)
{
    QPlainTextEdit::scrollContentsBy(dx, dy);
    if (dy != 0) {
        loadVisibleContent();
    }
}

void SmartEdit::updateScrollbar()
{
    if (!m_buffer) return;

    int documentHeight = document()->size().height();
    int viewportHeight = viewport()->height();

    if (documentHeight > viewportHeight) {
        verticalScrollBar()->setPageStep(viewportHeight);
        verticalScrollBar()->setRange(0, documentHeight - viewportHeight);
    }
}

void SmartEdit::loadVisibleContent()
{
    if (!m_buffer || !m_isLazyMode || m_updatingContent) {
        return;
    }

    m_updatingContent = true;

    int scrollValue = verticalScrollBar()->value();
    int linesInViewport = viewport()->height() / fontMetrics().height();

    const int BUFFER_LINES = 100;
    int startLine = std::max(0, scrollValue - BUFFER_LINES);
    int endLine = scrollValue + linesInViewport + BUFFER_LINES;

    size_t startOffset = estimateOffsetForLine(startLine);
    size_t endOffset = estimateOffsetForLine(endLine);

    QString newContent = m_buffer->readChunk(startOffset, endOffset - startOffset);

    if (newContent != m_currentContent) {
        m_currentContent = newContent;

        setUpdatesEnabled(false);

        QTextCursor cursor = textCursor();
        int savedPosition = cursor.position();
        int savedAnchor = cursor.anchor();

        setPlainText(newContent);

        cursor.setPosition(savedAnchor);
        cursor.setPosition(savedPosition, QTextCursor::KeepAnchor);
        setTextCursor(cursor);

        setUpdatesEnabled(true);
    }

    updateScrollbar();

    m_updatingContent = false;

    if (m_pendingUpdate) {
        m_pendingUpdate = false;
        QMetaObject::invokeMethod(this, &SmartEdit::loadVisibleContent, Qt::QueuedConnection);
    }
}
