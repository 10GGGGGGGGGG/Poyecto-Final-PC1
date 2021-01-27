from PyQt5.QtWidgets import QFileDialog
import os
from data_preprocessing import *
import pickle


class nueva_prediccion():
    def initPrediccion(self):
        self.folder_pred_button.clicked.connect(self.getfolderPred)
        self.model_pred_button.clicked.connect(self.getModelPred)
        self.prep_prediction_button.clicked.connect(
            self.prediction_Preprocessing)
        self.pred_button.clicked.connect(self.make_prediction)
        self.export_button.clicked.connect(self.save_to_Excel)

    def getfolderPred(self):
        self.pred_prep_label.clear()
        dir = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio", "/home", QFileDialog.ShowDirsOnly)
        if dir != "":
            self.pred_prep_label.clear()
            self.pred_line.setText(dir)

    def getModelPred(self):
        fname = QFileDialog.getOpenFileName(self, 'Seleccionar modelo',
                                            '/home', "modelo (*.model)")
        if fname[0] != "":
            self.pred_prep_label.clear()
            self.model_line.setText(fname[0])

    def prediction_Preprocessing(self):
        if(self.model_line.text() == ""):
            self.pred_prep_label.setText(
                "Seleccione el modelo a utilizar.")
        else:
            self.loaded_tuple = pickle.load(open(self.model_line.text(), 'rb'))
            self.cv = self.loaded_tuple[1]
            self.pred_prep_label.setStyleSheet(
                "QLabel {font-weight: normal;color : black;}")
            self.pred_prep_label.setText("Procesando...")
            self.app.processEvents()

            textos = []
            self.newText_names = os.listdir(self.pred_line.text())
            for n in self.newText_names:
                file = open(self.pred_line.text()+'/'+n, "r",
                            encoding='utf-8', errors='ignore')
                textos.append(file.read())

            self.proc_texts_prediction = token_stop_stem_lower(textos, stopWords=self.stopW.isChecked(
            ), stemmer=self.stemming.isChecked(), minus=self.minusc.isChecked(), elim_num=self.elim_num.isChecked())

            X_val = self.cv.transform(self.proc_texts_prediction).toarray()
            self.X_new_pred = pd.DataFrame(
                X_val, columns=self.cv.get_feature_names())

            self.pred_prep_label.setStyleSheet(
                "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
            self.pred_prep_label.setText("¡Listo!")
            # print(self.X_pred)

    def make_prediction(self):
        loaded_model = self.loaded_tuple[0]
        model_name = type(loaded_model).__name__
        if(model_name != "LinearSVC" and model_name != "SVC" and model_name != "SGDClassifier"):
            result = loaded_model.predict_proba(self.X_new_pred)
            self.prediction_dataset = pd.DataFrame(
                columns=['nombre de archivo', 'predicción', 'confianza (despoblación)', 'confianza (no despoblación)'])
            for i in range(len(self.newText_names)):
                self.prediction_dataset = self.prediction_dataset.append(
                    {"nombre de archivo": self.newText_names[i],
                     "predicción": 'despoblación' if result[i][0] >= 0.5 else 'no despoblación',
                     "confianza (despoblación)": str(result[i][0]),
                     "confianza (no despoblación)": str(result[i][1])},
                    ignore_index=True)
        else:
            result = loaded_model.predict(self.X_new_pred)
            self.prediction_dataset = pd.DataFrame(
                columns=['nombre de archivo', 'predicción'])
            for i in range(len(self.newText_names)):
                self.prediction_dataset = self.prediction_dataset.append(
                    {"nombre de archivo": self.newText_names[i],
                     "predicción": 'despoblación' if result[i] == 0 else 'no despoblación'},
                    ignore_index=True)

        proxyModel = QSortFilterProxyModel()
        modelt = PandasModel(self.prediction_dataset, header=True)
        proxyModel.setSourceModel(modelt)
        self.pred_tableView.setModel(proxyModel)
        self.pred_tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pred_tableView.resizeRowsToContents()
        self.pred_tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def save_to_Excel(self):
        fileName = QFileDialog.getSaveFileName(
            self, "Save File", "/home/untitled.xlsx", "Excel file (*.xlsx)")
        if(fileName[0] != ""):
            self.prediction_dataset.to_excel(fileName[0], index=True)
