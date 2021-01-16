import sys
import time
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
        self.trainModels_button.clicked.connect(self.initCrossValidation)

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
