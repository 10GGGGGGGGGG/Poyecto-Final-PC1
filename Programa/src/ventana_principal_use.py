from ventana_principal_ui import *
from seleccion_de_modelos import *
from entrenamiento_y_resultados import *
from nueva_prediccion import *
from data_preprocessing import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QSize, QThread
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow, seleccion_de_modelos, entrenamiento_y_resultados, nueva_prediccion):
    def __init__(self, app, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.initModelos()
        self.initEntrenamiento()
        self.initPrediccion()
        self.desp_button.clicked.connect(self.getfolderDesp)
        self.nodesp_button.clicked.connect(self.getfoldernoDesp)
        self.prepButton.clicked.connect(self.loadingPrep)
        self.matrixButton.clicked.connect(self.loadingMatrix)
        self.app = app

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

        textos, self.y = load_texts(
            self.desp_line.text(), 0)  # 0 = despoblación

        self.labelPrep.setText("Procesando..")
        app.processEvents()

        textos, self.y = load_texts(
            self.nodesp_line.text(), 1, textos, self.y)  # 1 = no despoblación

        self.labelPrep.setText("Procesando...")
        app.processEvents()

        self.processed_texts = token_stop_stem_lower(textos, stopWords=self.stopW.isChecked(
        ), stemmer=self.stemming.isChecked(), minus=self.minusc.isChecked(), elim_num=self.elim_num.isChecked())
        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
        self.labelPrep.setText("¡Listo!")

    def loadingMatrix(self):
        if((not self.cv_occ.isChecked()) and (not self.cv_tfidf.isChecked())):
            self.labelMatrix.setText(
                "Seleccione uno de los dos tipos de matriz de términos.")
        else:
            self.labelPrep.setStyleSheet(
                "QLabel {font-weight: normal;color : black;}")
            self.labelMatrix.setText("Procesando...")

            if(self.cv_occ.isChecked()):
                self.cv = CountVectorizer(
                    max_df=(self.maxdfSpin.value()/100), min_df=(self.mindfSpin.value()/100))
                equis = self.cv.fit_transform(self.processed_texts).toarray()
                self.X = pd.DataFrame(
                    equis, columns=self.cv.get_feature_names())

            elif(self.cv_tfidf.isChecked()):
                self.cv = TfidfVectorizer(
                    max_df=(self.maxdfSpin.value()/100), min_df=(self.mindfSpin.value()/100))
                X_tfidf = self.cv.fit_transform(
                    self.processed_texts).toarray()
                self.X = pd.DataFrame(
                    X_tfidf, columns=self.cv.get_feature_names())

            model = PandasModel(self.X, header=True)
            self.tableViewMatriz.setModel(model)
            self.tableViewMatriz.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tableViewMatriz.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

            self.labelPrep.setStyleSheet(
                "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
            self.labelMatrix.setText("¡Listo!")


if __name__ == "__main__":
    global app
    app = QtWidgets.QApplication([])
    window = MainWindow(app)
    window.show()
    app.exec_()
