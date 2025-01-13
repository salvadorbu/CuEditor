#include "Editor.h"
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qboxlayout.h>

Editor::Editor(QWidget* parent)
    : QMainWindow(parent),
    textEdit(new SmartEdit(this)),
    search(std::make_unique<SearchWrapper>())
{
    ui.setupUi(this);
    setCentralWidget(textEdit);
    setupMenus();
    setupSearchUI();

    if (!search->isInitialized()) {
        QMessageBox::warning(this, tr("Warning"),
            tr("Hardware-accelerated search is not available."));
    }
}

Editor::~Editor()
{
}

void Editor::setupMenus() {
    QMenu* fileMenu = menuBar()->addMenu(tr("File"));

    QAction* newAction = new QAction(tr("New"), this);
    QAction* openAction = new QAction(tr("Open"), this);
    QAction* saveAction = new QAction(tr("Save"), this);
    QAction* saveAsAction = new QAction(tr("Save As"), this);
    QAction* exitAction = new QAction(tr("Exit"), this);

    newAction->setShortcut(QKeySequence::New);
    openAction->setShortcut(QKeySequence::Open);
    saveAction->setShortcut(QKeySequence::Save);

    fileMenu->addAction(newAction);
    fileMenu->addAction(openAction);
    fileMenu->addAction(saveAction);
    fileMenu->addAction(saveAsAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    connect(newAction, &QAction::triggered, this, &Editor::newFile);
    connect(openAction, &QAction::triggered, this, &Editor::openFile);
    connect(saveAction, &QAction::triggered, this, &Editor::saveFile);
    connect(saveAsAction, &QAction::triggered, this, &Editor::saveFileAs);
    connect(exitAction, &QAction::triggered, this, &Editor::close);

    QMenu* editMenu = menuBar()->addMenu(tr("Edit"));
    QAction* cutAction = new QAction(tr("Cut"), this);
    QAction* copyAction = new QAction(tr("Copy"), this);
    QAction* pasteAction = new QAction(tr("Paste"), this);
    QAction* findAction = new QAction(tr("Find"), this);

    cutAction->setShortcut(QKeySequence::Cut);
    copyAction->setShortcut(QKeySequence::Copy);
    pasteAction->setShortcut(QKeySequence::Paste);
    findAction->setShortcut(QKeySequence::Find);

    editMenu->addAction(cutAction);
    editMenu->addAction(copyAction);
    editMenu->addAction(pasteAction);
    editMenu->addSeparator();
    editMenu->addAction(findAction);

    connect(cutAction, &QAction::triggered, textEdit, &QPlainTextEdit::cut);
    connect(copyAction, &QAction::triggered, textEdit, &QPlainTextEdit::copy);
    connect(pasteAction, &QAction::triggered, textEdit, &QPlainTextEdit::paste);
    connect(findAction, &QAction::triggered, this, &Editor::showFindDialog);

    QMenu* viewMenu = menuBar()->addMenu(tr("View"));
    QAction* wordWrapAction = new QAction(tr("Word Wrap"), this);
    wordWrapAction->setCheckable(true);
    wordWrapAction->setChecked(true);
    viewMenu->addAction(wordWrapAction);

    connect(wordWrapAction, &QAction::toggled, this, &Editor::toggleWordWrap);
}

void Editor::setupSearchUI() {
    findDialog = new QDialog(this);
    findDialog->setWindowTitle(tr("Find"));

    QVBoxLayout* layout = new QVBoxLayout(findDialog);

    searchInput = new QLineEdit(findDialog);
    findButton = new QPushButton(tr("Find"), findDialog);
    searchStatus = new QLabel(findDialog);
    nextButton = new QPushButton(tr("Next"), findDialog);
    prevButton = new QPushButton(tr("Previous"), findDialog);
    nextButton->setEnabled(false);
    prevButton->setEnabled(false);

    layout->addWidget(new QLabel(tr("Search for:")));
    layout->addWidget(searchInput);
    layout->addWidget(findButton);
    layout->addWidget(searchStatus);
    layout->addWidget(nextButton);
    layout->addWidget(prevButton);

    connect(findButton, &QPushButton::clicked, this, &Editor::performSearch);
    connect(searchInput, &QLineEdit::returnPressed, this, &Editor::performSearch);
    connect(nextButton, &QPushButton::clicked, this, &Editor::nextSearchResult);
    connect(prevButton, &QPushButton::clicked, this, &Editor::previousSearchResult);
}

void Editor::newFile() {
    currentFile.clear();
    fileBuffer.reset();
    textEdit->clear();
    setWindowTitle(tr("Editor"));
}

void Editor::openFile() {
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
        tr("Text Files (*.txt);;All Files (*)"));

    if (fileName.isEmpty())
        return;

    auto buffer = std::make_shared<FileBuffer>();
    if (!buffer->load(fileName)) {
        QMessageBox::critical(this, tr("Error"), tr("Could not open file."));
        return;
    }

    fileBuffer = buffer;
    textEdit->setBuffer(buffer);
    currentFile = fileName;
    setWindowTitle(fileName);

    if (search->isInitialized() && shouldUseHardwareSearch(buffer->size())) {
        search->loadFile(fileName);
    }
}

void Editor::saveFile() {
    if (currentFile.isEmpty()) {
        saveFileAs();
        return;
    }

    QFile file(currentFile);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::critical(this, tr("Error"), tr("Could not save file."));
        return;
    }

    QTextStream out(&file);
    out << textEdit->toPlainText();
}

void Editor::saveFileAs() {
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), QString(),
        tr("Text Files (*.txt);;All Files (*)"));

    if (fileName.isEmpty())
        return;

    currentFile = fileName;
    saveFile();
    setWindowTitle(fileName);
}

void Editor::toggleWordWrap(bool wrap) {
    textEdit->setWordWrapMode(wrap ? QTextOption::WrapAtWordBoundaryOrAnywhere
        : QTextOption::NoWrap);
}

void Editor::showFindDialog() {
    searchStatus->clear();
    searchInput->clear();
    findDialog->show();
    findDialog->raise();
    findDialog->activateWindow();
    searchInput->setFocus();
}

void Editor::performSearch() {
    QString pattern = searchInput->text();
    if (pattern.isEmpty()) {
        searchStatus->setText(tr("Please enter a search pattern."));
        return;
    }

    if (!fileBuffer) {
        searchStatus->setText(tr("No file is currently open."));
        return;
    }

    if (search->isInitialized() && shouldUseHardwareSearch(fileBuffer->size())) {
        if (search->search(pattern, searchResults)) {
            if (searchResults.isEmpty()) {
                searchStatus->setText(tr("Pattern not found."));
                nextButton->setEnabled(false);
                prevButton->setEnabled(false);
            }
            else {
                searchStatus->setText(tr("Found %1 matches.").arg(searchResults.size()));
                currentSearchResult = 0;
                nextButton->setEnabled(true);
                prevButton->setEnabled(true);
                navigateToResult(searchResults[0]);
            }
        }
        else {
            searchStatus->setText(tr("Search failed."));
            nextButton->setEnabled(false);
            prevButton->setEnabled(false);
        }
    }
    else {
        collectSoftwareSearchResults(pattern);

        if (searchResults.isEmpty()) {
            searchStatus->setText(tr("Pattern not found."));
            nextButton->setEnabled(false);
            prevButton->setEnabled(false);
        }
        else {
            searchStatus->setText(tr("Found %1 matches.").arg(searchResults.size()));
            currentSearchResult = 0;
            nextButton->setEnabled(true);
            prevButton->setEnabled(true);

            QTextCursor cursor = textEdit->textCursor();
            cursor.setPosition(searchResults[0]);
            cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor,
                pattern.length());
            textEdit->setTextCursor(cursor);
        }
    }
}

void Editor::navigateToResult(uint64_t offset)
{
    if (!fileBuffer) return;

    if (search->isInitialized() && shouldUseHardwareSearch(fileBuffer->size())) {
        const size_t contextSize = 1024;
        size_t start = offset > contextSize ? offset - contextSize : 0;
        QString chunk = fileBuffer->readChunk(start, contextSize * 2);

        textEdit->setPlainText(chunk);

        QTextCursor cursor = textEdit->textCursor();
        cursor.setPosition(offset - start);
        cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor,
            searchInput->text().length());
        textEdit->setTextCursor(cursor);
        textEdit->centerCursor();
    }
    else {
        QTextCursor cursor = textEdit->textCursor();
        cursor.setPosition(offset);
        cursor.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor,
            searchInput->text().length());
        textEdit->setTextCursor(cursor);
        textEdit->centerCursor();
    }
}

void Editor::nextSearchResult()
{
    if (searchResults.isEmpty()) return;

    currentSearchResult = (currentSearchResult + 1) % searchResults.size();
    navigateToResult(searchResults[currentSearchResult]);

    searchStatus->setText(tr("Match %1 of %2")
        .arg(currentSearchResult + 1)
        .arg(searchResults.size()));
}

void Editor::previousSearchResult()
{
    if (searchResults.isEmpty()) return;

    if (currentSearchResult == 0)
        currentSearchResult = searchResults.size() - 1;
    else
        currentSearchResult--;

    navigateToResult(searchResults[currentSearchResult]);

    searchStatus->setText(tr("Match %1 of %2")
        .arg(currentSearchResult + 1)
        .arg(searchResults.size()));
}

bool Editor::shouldUseHardwareSearch(size_t fileSize) const {
    return fileSize > 1024 * 1024;
}

void Editor::collectSoftwareSearchResults(const QString& pattern) {
    searchResults.clear();
    currentSearchResult = 0;

    QTextCursor cursor = textEdit->textCursor();
    cursor.movePosition(QTextCursor::Start);
    textEdit->setTextCursor(cursor);

    QTextDocument::FindFlags flags;
    while (textEdit->find(pattern, flags)) {
        cursor = textEdit->textCursor();
        searchResults.append(cursor.position() - pattern.length());
    }

    cursor.movePosition(QTextCursor::Start);
    textEdit->setTextCursor(cursor);
}
