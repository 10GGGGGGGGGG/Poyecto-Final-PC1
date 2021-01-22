import sys
import time
import pickle
from data_preprocessing import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QProgressBar, QPushButton)
from sklearn.model_selection import cross_validate

TIME_LIMIT = 100


class CVTraining(QThread):
    countChanged = pyqtSignal(int)
    consoleChanged = pyqtSignal(str)
    endOfCV = pyqtSignal(pd.DataFrame)

    def __init__(self, modelos, dataFrame, X, y):
        super().__init__()
        self.modelos = modelos
        self.df = dataFrame
        self.X = X
        self.y = y
        self.cvDataset = pd.DataFrame(columns=[
                                      'modelo', 'acierto medio', 'precisión media', 'recall medio', 'f1 medio'])

    def run(self):
        count = 0
        if(len(self.modelos) == 0):
            self.consoleChanged.emit("No hay ningún modelo seleccionado.")
        else:
            increment = 100/len(self.modelos)
            for i in range(len(self.modelos)):
                self.consoleChanged.emit(
                    '\nRealizando validación cruzada del modelo ' + (self.df['modelo'][i]).upper())
                self.consoleChanged.emit(self.df['parámetros'][i])
                results = cross_validate(self.modelos[i], self.X, self.y, cv=5, scoring=[
                                         'accuracy', 'precision', 'recall', 'f1'])
                self.cvDataset = self.cvDataset.append(
                    {"modelo": self.df['modelo'][i],
                     "acierto medio": str(results['test_accuracy'].mean()),
                     "precisión media": str(results['test_precision'].mean()),
                     "recall medio": str(results['test_recall'].mean()),
                     "f1 medio": str(results['test_f1'].mean())},
                    ignore_index=True)
                self.consoleChanged.emit('Validación finalizada.')
                count += increment
                self.countChanged.emit(count)
            self.endOfCV.emit(self.cvDataset)


class entrenamiento_y_resultados():
    def initEntrenamiento(self):
        self.CVModels_button.clicked.connect(self.initCrossValidation)
        self.training_button.clicked.connect(self.training)
        self.save_model_button.clicked.connect(self.saveFile)
        self.modelos_comboBox.currentIndexChanged.connect(self.clearLabels)

    def onCountChanged(self, value):
        self.progress.setValue(value)

    def onConsoleChanged(self, value):
        if(self.console_text.toPlainText() == "" or self.console_text.toPlainText() == "No hay ningún modelo seleccionado."):
            self.console_text.setText(value)
        else:
            self.console_text.setText(
                self.console_text.toPlainText()+"\n"+value)

    def initCrossValidation(self):
        self.console_text.clear()
        self.calc = CVTraining(
            self.modelos, self.modelosDataframe, self.X, self.y)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.consoleChanged.connect(self.onConsoleChanged)
        self.calc.endOfCV.connect(self.onEndOfCV)
        self.calc.start()

    def onEndOfCV(self, df):
        proxyModel = QSortFilterProxyModel()
        modelt = PandasModel(df, header=True)
        proxyModel.setSourceModel(modelt)
        self.cv_tableView.setModel(proxyModel)
        self.cv_tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cv_tableView.resizeRowsToContents()
        self.cv_tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def training(self):
        print(self.modelos)
        print(self.modelos_comboBox.currentIndex())
        self.selected_model = self.modelos[self.modelos_comboBox.currentIndex(
        )]
        self.training_label.setStyleSheet(
            "QLabel {font-weight: normal;color : black;}")
        self.training_label.setText("Entrenando...")
        self.app.processEvents()
        self.selected_model.fit(self.X, self.y)
        self.training_label.setStyleSheet(
            "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
        self.training_label.setText("¡Entrenado!")
        print(self.selected_model)

    def clearLabels(self):
        self.training_label.clear()
        self.saved_label.clear()

    def saveFile(self):
        fileName = QFileDialog.getSaveFileName(
            self, "Save File", "/home/untitled.model", "modelo (*.model)")
        # print(fileName[0])
        if(fileName[0] != ""):
            model_tuple = (self.selected_model, self.cv)
            pickle.dump(model_tuple, open(fileName[0], 'wb'))
            self.saved_label.setStyleSheet(
                "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
            self.saved_label.setText("Guardado!")
