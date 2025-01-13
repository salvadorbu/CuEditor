#pragma once
#include <qplaintextedit.h>
#include <memory>
#include "FileBuffer.h"

class SmartEdit : public QPlainTextEdit {
	Q_OBJECT

public:
	SmartEdit(QWidget* parent = nullptr);
	void setBuffer(std::shared_ptr<FileBuffer> buffer);

protected:
	void resizeEvent(QResizeEvent* event) override;
	void scrollContentsBy(int dx, int dy) override;

private slots:
	void updateScrollbar();

private:
	void loadVisibleContent();
	std::shared_ptr<FileBuffer> m_buffer;
	bool m_updatingContent;
	QString m_currentContent;
	bool m_pendingUpdate;
	bool m_isLazyMode;
};