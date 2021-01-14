import sys
import time
from data_preprocessing import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QProgressBar, QPushButton)
from sklearn.model_selection import cross_validate

TIME_LIMIT = 100


class ExternalProgressBar(QThread):
    countChanged = pyqtSignal(int)
    consoleChanged = pyqtSignal(str)

    def __init__(self, modelos, dataFrame, X, y):
        super().__init__()
        self.modelos = modelos
        self.df = dataFrame
        self.X = X
        self.y = y

    def run(self):
        count = 0
        if(len(self.modelos) == 0):
            self.consoleChanged.emit("No hay ningún modelo seleccionado.")
        else:
            increment = 100/len(self.modelos)
            for i in range(len(self.modelos)):
                self.consoleChanged.emit(
                    '\nCROSS VALIDATION ' + self.df['modelo'][i])
                self.consoleChanged.emit(self.df['parámetros'][i])
                results = cross_validate(self.modelos[i], self.X, self.y, cv=5, scoring=[
                                         'accuracy', 'precision', 'recall', 'f1'])
                print(self.df['modelo'][i]+':')
                print('acierto medio: '+str(results['test_accuracy'].mean()))
                print('precisión media: ' +
                      str(results['test_precision'].mean()))
                print('recall medio: '+str(results['test_recall'].mean()))
                print('f1 medio(media harmónica de la precisión y el recall): ' +
                      str(results['test_f1'].mean()))

                count += increment
                self.countChanged.emit(count)


class entrenamiento_y_resultados():
    def initEntrenamiento(self):
        self.trainModels_button.clicked.connect(self.initProgressBar)

    def onCountChanged(self, value):
        self.progress.setValue(value)

    def onConsoleChanged(self, value):
        if(self.console_text.toPlainText() == "" or self.console_text.toPlainText() == "No hay ningún modelo seleccionado."):
            self.console_text.setText(value)
        else:
            self.console_text.setText(
                self.console_text.toPlainText()+"\n"+value)

    def initProgressBar(self):
        self.calc = ExternalProgressBar(
            self.modelos, self.modelosDataframe, self.X, self.y)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.consoleChanged.connect(self.onConsoleChanged)
        self.calc.start()
