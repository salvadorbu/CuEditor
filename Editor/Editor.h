#pragma once

#include <QtWidgets/QMainWindow>
#include <qtextedit.h>
#include <qstring.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include "ui_Editor.h"
#include "SmartEdit.h"
#include "SearchWrapper.h"

class Editor : public QMainWindow
{
    Q_OBJECT

public:
    Editor(QWidget* parent = nullptr);
    ~Editor();

private slots:
    void newFile();
    void openFile();
    void saveFile();
    void saveFileAs();
    void toggleWordWrap(bool wrap);
    void showFindDialog();
    void performSearch();
    void navigateToResult(uint64_t offset);
    void nextSearchResult();
    void previousSearchResult();

private:
    void setupMenus();
    void setupSearchUI();
    bool shouldUseHardwareSearch(size_t fileSize) const;
    void collectSoftwareSearchResults(const QString& pattern);

    Ui::EditorClass ui;
    SmartEdit* textEdit;
    QString currentFile;
    std::shared_ptr<FileBuffer> fileBuffer;
    std::unique_ptr<SearchWrapper> search;

    QDialog* findDialog;
    QLineEdit* searchInput;
    QPushButton* findButton;
    QLabel* searchStatus;
    QVector<uint64_t> searchResults;
    size_t currentSearchResult;
    QPushButton* nextButton;
    QPushButton* prevButton;
};
