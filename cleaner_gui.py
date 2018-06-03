import os
import sys
import codecs

import subprocess

import collections

from PyQt4.QtCore import QStringList, QDir, QString

from PyQt4.QtGui import QTreeView, QFileSystemModel, QTableWidgetItem, QListWidgetItem

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True, choices=["gui"])
    # parser.add_argument("--gold_data", type=bool, default=False)
    # parser.add_argument("--output_dir", required=True)
    # parser.add_argument("--verbose", type=bool, default=False)

    args = parser.parse_args()

    from PyQt4 import QtGui
    import main_form

    class ExampleApp(QtGui.QMainWindow, main_form.Ui_MainWindow):
        def __init__(self):
            super(self.__class__, self).__init__()
            self.setupUi(self)  # This is defined in design.py file automatically

            # self.listWidget_Xoutput_files.addItem("deneme")

            cleaner_files_path = os.path.join(str(QDir.currentPath()), "cleaner_files")
            if not os.path.exists(cleaner_files_path):
                os.mkdir(cleaner_files_path)

            self.model = QFileSystemModel()
            self.model.setRootPath(cleaner_files_path)

            self.model.setNameFilters(QStringList(["Xoutput-n_analyses-*.txt"]))
            self.model.setNameFilterDisables(False)
            self.model.setFilter(QDir.Dirs | QDir.Files)


            self.treeView_Xoutput_files.setModel(self.model)

            self.treeView_Xoutput_files.setRootIndex(self.model.index(cleaner_files_path))

            self.treeView_Xoutput_files.setColumnWidth(0, 500)

            self.treeView_Xoutput_files.selectionModel().selectionChanged.connect(self.load_and_view_file_contents)

            self.rules_dict = {}

            self.special_button_for_level_01.setDisabled(True)

        def load_and_view_file_contents(self, current, previous):

            print current.indexes()
            model_index = current.indexes()[0]

            filename = self.model.data(model_index).toString()

            import re
            m = re.match(r"Xoutput-n_analyses-([0-9]+)", filename)
            if m:
                n_analyzes = int(m.group(1))
            else:
                n_analyzes = -1

            if n_analyzes == 1:
                self.special_button_for_level_01.setDisabled(False)
                self.special_button_for_level_01.clicked.connect(self.add_all_level_01_to_rule_dict)
            else:
                self.special_button_for_level_01.setDisabled(True)

            with codecs.open(filename, "r", encoding="utf8") as f:
                lines = f.readlines()
                # print lines
                self.listWidget_selected_file_contents.clear()
                self.listWidget_selected_file_contents.addItems(QStringList(lines))
                self.listWidget_selected_file_contents.selectionModel().selectionChanged.connect(self.load_and_view_samples_from_train_and_dev)

            self.load_and_view_rule_file_contents(n_analyzes)

        def load_and_view_rule_file_contents(self, n_analyzes):
            # load rules file
            self.rules_dict = {}
            rules_filename = "Xoutput-n_analyses-%02d.txt.rules" % n_analyzes
            try:
                with codecs.open(rules_filename, "r") as rules_f:
                    self.output_file_load_status.setText("%s loaded." % rules_filename)
                    self.output_file_load_status.setStyleSheet("QLabel { color : green; }")

                    rules = []

                    line = rules_f.readline().strip()
                    while line:
                        rules.append(line.split(" "))
                        self.rules_dict[int(line.split(" ")[0])] = line.split(" ")
                        line = rules_f.readline().strip()

                    self.update_tableWidgetxxxx(self.tableWidget_output_file_contents,
                                                sorted(self.rules_dict.items(), key=lambda x: x[0]),
                                                len(self.rules_dict.keys()),
                                                1 + 1 + n_analyzes + 1) # id + golden + FST analyzes + selected


            except IOError as e:
                # print "File not found"
                self.output_file_load_status.setText("File not found")
                self.output_file_load_status.setStyleSheet("QLabel { color : red; }")

                self.update_tableWidgetxxxx(self.tableWidget_output_file_contents,
                                            [],
                                            0,
                                            1)  # id + golden + FST analyzes + selected

        def update_tableWidgetxxxx(self, table_widget, rules, row_count, col_count):
            table_widget.clear()
            table_widget.setColumnCount(col_count)
            table_widget.setRowCount(row_count)

            if rules:
                for row in range(table_widget.rowCount()):
                    row_items = rules[row]
                    print row_items
                    item = self.listWidget_selected_file_contents.item(
                        int(row_items[0]) - 1)  # type: QListWidgetItem
                    item.setBackgroundColor(QtGui.QColor(255, 0, 0, 127))
                    for column in range(
                            table_widget.columnCount()):
                        if column < len(row_items[1]):
                            table_widget.setItem(row, column, QTableWidgetItem(row_items[1][column].decode("utf8")))

                # self.tableWidget_samples_from_train_and_dev.resizeColumnToContents()
                for column in range(table_widget.columnCount()):
                    table_widget.resizeColumnToContents(column)

        def update_corrected_morph_analysis(self, current, previous):

            model_index = current.indexes()[0]

            self.textEdit_2.setPlainText(self.listWidget_selected_row.model().data(model_index).toString())

        def add_all_level_01_to_rule_dict(self):

            self.rules_dict = {}

            for idx in range(self.listWidget_selected_file_contents.count()):
                row_items = unicode(self.listWidget_selected_file_contents.item(idx).text()).strip().split(" ")

                rules_item = [x.encode("utf8") for x in [row_items[0],
                                                         row_items[4],
                                                         row_items[-1],
                                                         row_items[-1]]]

                self.rules_dict[int(row_items[0])] = rules_item

            self.update_tableWidgetxxxx(self.tableWidget_output_file_contents,
                                        sorted(self.rules_dict.items(), key=lambda x: x[0]),
                                        len(self.rules_dict.keys()),
                                        1 + 1 + 1 + 1)  # id + golden + FST analyzes + selected


        def add_to_the_rule_dict(self, state):

            n_analyzes, entry_id = [int(x) for x in self.label_8.text().split(" ")]

            other_analyzes = [self.listWidget_selected_row.item(i) for i in range(self.listWidget_selected_row.count())] # type: list[QListWidgetItem]

            rules_item = [unicode(x).encode("utf8") for x in [entry_id,
                          self.textEdit_golden_morph_analysis.toPlainText()] + \
                          [x.text() for x in other_analyzes] + \
                         [self.textEdit_2.toPlainText()]]

            self.rules_dict[entry_id] = rules_item

            self.update_tableWidgetxxxx(self.tableWidget_output_file_contents,
                                        sorted(self.rules_dict.items(), key=lambda x: x[0]),
                                        len(self.rules_dict.keys()),
                                        1 + 1 + n_analyzes + 1)  # id + golden + FST analyzes + selected

        def load_and_view_samples_from_train_and_dev(self, current, previous):
            print current.indexes()

            model_index = current.indexes()[0]

            morph_analyzes = unicode(self.listWidget_selected_file_contents.model().data(
                model_index).toString()).strip().split(" ")
            # print morph_analyzes
            golden_morph_analysis = morph_analyzes[4]
            target = golden_morph_analysis[1:]

            other_morph_analyzes = morph_analyzes[5:]

            n_analyzes = len(other_morph_analyzes)

            self.label_3.setText("selected row id: %d" % int(morph_analyzes[0]))
            self.label_8.setText("%d %d" % (int(n_analyzes), int(morph_analyzes[0])))

            self.listWidget_selected_row.clear()
            self.listWidget_selected_row.addItems(QStringList(other_morph_analyzes))

            self.textEdit_golden_morph_analysis.setPlainText(golden_morph_analysis)

            # self.addRuleToTheListButton.clicked.connect(
            #     partial(self.save_to_file, n_analyzes=n_analyzes, entry_id=int(morph_analyzes[0])))

            # from functools import partial
            self.addRuleToTheListButton.clicked.connect(self.add_to_the_rule_dict)

            if len(other_morph_analyzes) == 1:
                self.textEdit_2.setPlainText(other_morph_analyzes[0])

            self.listWidget_selected_row.selectionModel().selectionChanged.connect(self.update_corrected_morph_analysis)

            print type(target)
            print target
            print target.encode("utf8")

            # target = target.replace("?", "\?")

            lines = subprocess.check_output(("grep -F -m 50 %s ./dataset/errors.gungor.ner.train_and_dev" % target).split(" "),
                                            shell=False)

            # print lines

            lines = [x.decode("utf8") for x in lines.split("\n")]

            print type(lines[0])
            print len(lines)

            self.tableWidget_samples_from_train_and_dev.clear()

            self.tableWidget_samples_from_train_and_dev.setColumnCount(n_analyzes + 1)
            self.tableWidget_samples_from_train_and_dev.setRowCount(len(lines)-1)

            for row in range(self.tableWidget_samples_from_train_and_dev.rowCount()):
                row_items = lines[row].split(" ")[2:]
                for column in range(self.tableWidget_samples_from_train_and_dev.columnCount()):
                    if column < len(row_items):
                        self.tableWidget_samples_from_train_and_dev.setItem(row, column, QTableWidgetItem(row_items[column]))

            # self.tableWidget_samples_from_train_and_dev.resizeColumnToContents()
            for column in range(self.tableWidget_samples_from_train_and_dev.columnCount()):
                self.tableWidget_samples_from_train_and_dev.resizeColumnToContents(column)

            self.sort_and_save_button.clicked.connect(self.sort_and_save)

        def sort_and_save(self):

            indexes = self.treeView_Xoutput_files.selectedIndexes()

            model_index = indexes[0]

            filename = self.model.data(model_index).toString()

            with open(filename+ ".rules", "w") as f:
                for row in range(self.tableWidget_output_file_contents.rowCount()):
                    row_content = []
                    for column in range(self.tableWidget_output_file_contents.columnCount()):
                        cell_content = self.tableWidget_output_file_contents.item(row, column).text() # type: QString
                        if cell_content:
                            row_content.append(unicode(cell_content).encode("utf8"))
                    if row != 0:
                        f.write("\n")
                    f.write(" ".join(row_content))


    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    form = ExampleApp()  # We set the form to be our ExampleApp (design)
    form.show()  # Show the form
    app.exec_()  # and execute the app