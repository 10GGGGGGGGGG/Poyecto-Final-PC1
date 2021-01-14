from ventana_principal_ui import *
from seleccion_de_modelos import *
from entrenamiento_y_resultados import *
from data_preprocessing import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QSize, QThread
import time


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow, seleccion_de_modelos, entrenamiento_y_resultados):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.initModelos()
        self.initEntrenamiento()
        self.desp_button.clicked.connect(self.getfolderDesp)
        self.nodesp_button.clicked.connect(self.getfoldernoDesp)
        self.prepButton.clicked.connect(self.loadingPrep)
        self.matrixButton.clicked.connect(self.loadingMatrix)

    def getfolderDesp(self):
        dir = QFileDialog.getExistingDirectory(
            self, "Open Directory", "/home", QFileDialog.ShowDirsOnly)
        if dir != "":
            self.desp_line.setText(dir)

    def getfoldernoDesp(self):
        dir = QFileDialog.getExistingDirectory(
            self, "Open Directory", "/home", QFileDialog.ShowDirsOnly)
        if dir != "":
            self.nodesp_line.setText(dir)

    def loadingPrep(self):
        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: normal;color : black;}")
        self.labelPrep.setText("Procesando.")
        app.processEvents()

        textos, y = load_texts(self.desp_line.text(), 'despoblacion')

        self.labelPrep.setText("Procesando..")
        app.processEvents()

        textos, y = load_texts(self.nodesp_line.text(),
                               'no_despoblacion', textos, y)

        self.labelPrep.setText("Procesando...")
        app.processEvents()

        self.processed_texts = token_stop_stem_lower(textos, stopWords=self.stopW.isChecked(
        ), stemmer=self.stemming.isChecked(), minus=self.minusc.isChecked(), elim_num=self.elim_num.isChecked())
        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
        self.labelPrep.setText("¡Listo!")

    def loadingMatrix(self):
        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: normal;color : black;}")
        self.labelMatrix.setText("Procesando...")

        if(self.cv_occ.isChecked()):
            X = cv_ocurrences(self.processed_texts, maxdf=(
                self.maxdfSpin.value()/100), mindf=(self.mindfSpin.value()/100))
        elif(self.cv_tfidf.isChecked()):
            X = cv_TFIDF(self.processed_texts, maxdf=(
                self.maxdfSpin.value()/100), mindf=(self.mindfSpin.value()/100))

        model = PandasModel(X, header=True)
        self.tableViewMatriz.setModel(model)
        self.tableViewMatriz.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableViewMatriz.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
        self.labelMatrix.setText("¡Listo!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
