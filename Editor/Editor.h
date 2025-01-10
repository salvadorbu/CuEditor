#pragma once

#include <QtWidgets/QMainWindow>
#include <qtextedit.h>
#include <qstring.h>
#include "ui_Editor.h"

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

private:
    Ui::EditorClass ui;
    QTextEdit* textEdit;
    QString currentFile;
};
