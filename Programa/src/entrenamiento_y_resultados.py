import sys
import time
from data_preprocessing import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QProgressBar, QPushButton)

TIME_LIMIT = 100


class ExternalProgressBar(QThread):
    countChanged = pyqtSignal(int)
    consoleChanged = pyqtSignal(str)

    def __init__(self, modelos):
        super().__init__()
        self.modelos = modelos

    def run(self):
        count = 0
        if(len(self.modelos) == 0):
            self.consoleChanged.emit("No hay ningún modelo seleccionado.")
        else:
            increment = 100/len(self.modelos)
            for m in self.modelos:
                time.sleep(1)
                self.consoleChanged.emit(str(m))
                # tambien se puede imprimir el cross val report
                print(m)
                # entrenamos el modelo
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
        self.calc = ExternalProgressBar(self.modelos)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.consoleChanged.connect(self.onConsoleChanged)
        self.calc.start()
