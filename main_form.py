# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_form.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1303, 987)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("../../../../../../../usr/share/icons/gnome/32x32/apps/accessories-text-editor.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 20, 131, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 170, 151, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.listWidget_selected_file_contents = QtGui.QListWidget(self.centralwidget)
        self.listWidget_selected_file_contents.setGeometry(QtCore.QRect(10, 200, 781, 192))
        self.listWidget_selected_file_contents.setObjectName(_fromUtf8("listWidget_selected_file_contents"))
        self.listWidget_selected_row = QtGui.QListWidget(self.centralwidget)
        self.listWidget_selected_row.setGeometry(QtCore.QRect(810, 270, 471, 121))
        self.listWidget_selected_row.setObjectName(_fromUtf8("listWidget_selected_row"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(810, 170, 151, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 450, 261, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(810, 190, 171, 17))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.textEdit_golden_morph_analysis = QtGui.QTextEdit(self.centralwidget)
        self.textEdit_golden_morph_analysis.setGeometry(QtCore.QRect(810, 220, 471, 41))
        self.textEdit_golden_morph_analysis.setObjectName(_fromUtf8("textEdit_golden_morph_analysis"))
        self.label_6 = QtGui.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 680, 151, 17))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.label_7 = QtGui.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(610, 410, 181, 20))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.textEdit_2 = QtGui.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(810, 400, 471, 41))
        self.textEdit_2.setReadOnly(True)
        self.textEdit_2.setObjectName(_fromUtf8("textEdit_2"))
        self.addRuleToTheListButton = QtGui.QPushButton(self.centralwidget)
        self.addRuleToTheListButton.setGeometry(QtCore.QRect(810, 450, 471, 27))
        self.addRuleToTheListButton.setObjectName(_fromUtf8("addRuleToTheListButton"))
        self.treeView_Xoutput_files = QtGui.QTreeView(self.centralwidget)
        self.treeView_Xoutput_files.setGeometry(QtCore.QRect(10, 40, 1271, 121))
        self.treeView_Xoutput_files.setObjectName(_fromUtf8("treeView_Xoutput_files"))
        self.treeView_Xoutput_files.header().setMinimumSectionSize(100)
        self.tableWidget_samples_from_train_and_dev = QtGui.QTableWidget(self.centralwidget)
        self.tableWidget_samples_from_train_and_dev.setGeometry(QtCore.QRect(10, 480, 1271, 192))
        self.tableWidget_samples_from_train_and_dev.setObjectName(_fromUtf8("tableWidget_samples_from_train_and_dev"))
        self.tableWidget_samples_from_train_and_dev.setColumnCount(0)
        self.tableWidget_samples_from_train_and_dev.setRowCount(0)
        self.tableWidget_output_file_contents = QtGui.QTableWidget(self.centralwidget)
        self.tableWidget_output_file_contents.setGeometry(QtCore.QRect(10, 710, 1271, 192))
        self.tableWidget_output_file_contents.setObjectName(_fromUtf8("tableWidget_output_file_contents"))
        self.tableWidget_output_file_contents.setColumnCount(0)
        self.tableWidget_output_file_contents.setRowCount(0)
        self.output_file_load_status = QtGui.QLabel(self.centralwidget)
        self.output_file_load_status.setGeometry(QtCore.QRect(20, 910, 261, 17))
        self.output_file_load_status.setText(_fromUtf8(""))
        self.output_file_load_status.setObjectName(_fromUtf8("output_file_load_status"))
        self.sort_and_save_button = QtGui.QPushButton(self.centralwidget)
        self.sort_and_save_button.setGeometry(QtCore.QRect(850, 910, 431, 27))
        self.sort_and_save_button.setObjectName(_fromUtf8("sort_and_save_button"))
        self.label_8 = QtGui.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(970, 170, 68, 17))
        self.label_8.setText(_fromUtf8(""))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.special_button_for_level_01 = QtGui.QPushButton(self.centralwidget)
        self.special_button_for_level_01.setGeometry(QtCore.QRect(380, 410, 221, 27))
        self.special_button_for_level_01.setObjectName(_fromUtf8("special_button_for_level_01"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1303, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "gungor.ner cleaner tool", None))
        self.label.setText(_translate("MainWindow", "Xoutput files", None))
        self.label_2.setText(_translate("MainWindow", "selected file contents", None))
        self.label_3.setText(_translate("MainWindow", "selected row", None))
        self.label_4.setText(_translate("MainWindow", "samples from train and dev files", None))
        self.label_5.setText(_translate("MainWindow", "golden morph. analysis", None))
        self.label_6.setText(_translate("MainWindow", "rule list", None))
        self.label_7.setText(_translate("MainWindow", "corrected morph. analysis", None))
        self.addRuleToTheListButton.setText(_translate("MainWindow", "Add this as a rule", None))
        self.sort_and_save_button.setText(_translate("MainWindow", "Sort and Save the rules", None))
        self.special_button_for_level_01.setText(_translate("MainWindow", "Add all rules for n_analysis=1", None))

