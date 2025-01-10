#include "Editor.h"
#include <qfiledialog.h>

Editor::Editor(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    textEdit = new QTextEdit(this);
    setCentralWidget(textEdit);

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

    cutAction->setShortcut(QKeySequence::Cut);
    copyAction->setShortcut(QKeySequence::Copy);
    pasteAction->setShortcut(QKeySequence::Paste);

    editMenu->addAction(cutAction);
    editMenu->addAction(copyAction);
    editMenu->addAction(pasteAction);

    connect(cutAction, &QAction::triggered, textEdit, &QTextEdit::cut);
    connect(copyAction, &QAction::triggered, textEdit, &QTextEdit::copy);
    connect(pasteAction, &QAction::triggered, textEdit, &QTextEdit::paste);

    QMenu* viewMenu = menuBar()->addMenu(tr("View"));
    QAction* wordWrapAction = new QAction(tr("Word Wrap"), this);
    wordWrapAction->setCheckable(true);
    wordWrapAction->setChecked(true);
    viewMenu->addAction(wordWrapAction);

    connect(wordWrapAction, &QAction::toggled, this, &Editor::toggleWordWrap);

    textEdit->setWordWrapMode(QTextOption::WrapAtWordBoundaryOrAnywhere);

    setWindowTitle("Editor");
}

Editor::~Editor()
{
}

void Editor::newFile()
{
    currentFile.clear();
    textEdit->clear();
    setWindowTitle("Qt Text Editor");
}

void Editor::openFile()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), QString(),
        tr("Text Files (*.txt);;All Files (*)"));
    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (file.open(QFile::ReadOnly | QFile::Text)) {
            currentFile = fileName;
            QTextStream in(&file);
            textEdit->setText(in.readAll());
            file.close();
            setWindowTitle(fileName);
        }
    }
}

void Editor::saveFile()
{
    if (currentFile.isEmpty()) {
        saveFileAs();
    }
    else {
        QFile file(currentFile);
        if (file.open(QFile::WriteOnly | QFile::Text)) {
            QTextStream out(&file);
            out << textEdit->toPlainText();
            file.close();
        }
    }
}

void Editor::saveFileAs()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save As"), QString(),
        tr("Text Files (*.txt);;All Files (*)"));
    if (!fileName.isEmpty()) {
        currentFile = fileName;
        saveFile();
        setWindowTitle(fileName);
    }
}

void Editor::toggleWordWrap(bool wrap)
{
    textEdit->setWordWrapMode(wrap ? QTextOption::WrapAtWordBoundaryOrAnywhere
        : QTextOption::NoWrap);
}
