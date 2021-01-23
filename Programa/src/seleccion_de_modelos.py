import pandas as pd
from data_preprocessing import *
from PyQt5 import QtWidgets
from sklearn import tree, naive_bayes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier


class seleccion_de_modelos():
    def initModelos(self):
        self.modelos = []
        self.modelosDataframe = pd.DataFrame(columns=['modelo', 'parámetros'])
        self.add_layer_button.clicked.connect(self.addLayer)
        self.remove_layer_button.clicked.connect(self.removeLayer)
        self.cargarModelos_button.clicked.connect(self.cargarModelos)

    def addLayer(self):
        columnas = self.tableWidgetNeurons.columnCount()
        self.tableWidgetNeurons.setColumnCount(columnas+1)
        item = QtWidgets.QTableWidgetItem()
        item.setText("capa oculta "+str(columnas+1))
        self.tableWidgetNeurons.setHorizontalHeaderItem(columnas, item)

    def removeLayer(self):
        columnas = self.tableWidgetNeurons.columnCount()
        self.tableWidgetNeurons.setColumnCount(columnas-1)

    def cargarModelos(self):
        self.modelos.clear()
        self.console_text.clear()
        self.modelosDataframe = self.modelosDataframe.iloc[0:0]

        if(self.dTree_check.isChecked()):
            self.loadTree()

        if(self.bayes_check.isChecked()):
            self.loadNaiveBayes()

        if(self.RForest_check.isChecked()):
            self.loadRandomForest()

        if(self.knn_check.isChecked()):
            self.loadKNN()

        if(self.GBoosting_check.isChecked()):
            self.loadGBoosting()

        if(self.mlp_check.isChecked()):
            self.loadMLP()

        if(self.svc_check.isChecked()):
            self.loadLinearSVC()

        if(self.svm_check.isChecked()):
            self.loadSVM()

        if(self.sgd_check.isChecked()):
            self.loadSGD()

        modelt = PandasModel(self.modelosDataframe, header=True)
        print(self.modelos)
        print(self.modelosDataframe)
        self.tableView_modelos.setModel(modelt)
        self.tableView_modelos.resizeRowsToContents()
        self.modelos_comboBox.clear()
        for index, row in self.modelosDataframe.iterrows():
            self.modelos_comboBox.addItem(row['modelo'])
        self.modelos_comboBox.setSizeAdjustPolicy(0)

    def loadTree(self):
        depth = None if self.dTree_spinBox.value() == 0 else self.dTree_spinBox.value()
        dTree_switcher = {
            0: "auto",
            1: "sqrt",
            2: "log2",
            3: None
        }
        dTree_criterion = self.dTree_comboBox1.currentText()
        dtree_features = dTree_switcher.get(
            self.dTree_comboBox2.currentIndex(), "switch out of bounds")

        self.modelos.append(tree.DecisionTreeClassifier(random_state=0,
                                                        max_depth=depth,
                                                        criterion=dTree_criterion,
                                                        max_features=dtree_features))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Árbol de decisión",
             "parámetros": "profundidad máxima: "+("None" if depth is None else str(depth)) +
             "\ncriterio de calidad: "+dTree_criterion +
             "\nmáximas características: "+("None" if dtree_features is None else dtree_features)},
            ignore_index=True)

    def loadNaiveBayes(self):
        bayes_switcher = {
            0: naive_bayes.GaussianNB(),
            1: naive_bayes.MultinomialNB(),
            2: naive_bayes.ComplementNB(),
            3: naive_bayes.BernoulliNB(),
            4: naive_bayes.CategoricalNB(),
        }
        self.modelos.append(bayes_switcher.get(
            self.bayes_comboBox.currentIndex(), "switch out of bounds"))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Naive Bayes",
             "parámetros": "clasificador: "+self.bayes_comboBox.currentText()},
            ignore_index=True)

    def loadRandomForest(self):
        depth = None if self.RForest_spinBox_depth.value(
        ) == 0 else self.RForest_spinBox_depth.value()
        RForest_criterion = self.RForest_comboBox1.currentText()
        RForest_switcher = {
            0: "auto",
            1: "sqrt",
            2: "log2",
            3: None
        }
        RForest_features = RForest_switcher.get(
            self.RForest_comboBox2.currentIndex(), "switch out of bounds")

        self.modelos.append(RandomForestClassifier(random_state=0,
                                                   n_estimators=self.RForest_spinBox_trees.value(),
                                                   max_depth=depth,
                                                   criterion=RForest_criterion,
                                                   max_features=RForest_features))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Random Forest",
             "parámetros": "número de árboles: "+str(self.RForest_spinBox_trees.value()) +
             "\nprofundidad máxima: "+("None" if depth is None else str(depth)) +
             "\ncriterio de calidad: "+RForest_criterion +
             "\nmáximas características: "+("None" if RForest_features is None else RForest_features)},
            ignore_index=True)

    def loadKNN(self):
        knn_switcher = {
            0: "auto",
            1: "brute",
            2: "kd_tree",
            3: "ball_tree"
        }
        self.modelos.append(KNeighborsClassifier(n_neighbors=self.knn_spinBox.value(),
                                                 algorithm=knn_switcher.get(self.knn_comboBox.currentIndex(), "switch out of bounds")))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "K Vecinos",
             "parámetros": "número de vecinos: "+str(self.knn_spinBox.value()) +
             "\nalgoritmo: "+self.knn_comboBox.currentText()},
            ignore_index=True)

    def loadGBoosting(self):
        GBoosting_switcher_loss = {
            0: "deviance",
            1: "exponential"
        }
        GBoosting_switcher_f = {
            0: "auto",
            1: "sqrt",
            2: "log2",
            3: None
        }
        GBoosting_features = GBoosting_switcher_f.get(
            self.GBoosting_comboBox2.currentIndex(), "switch out of bounds")
        self.modelos.append(GradientBoostingClassifier(random_state=0,
                                                       n_estimators=self.GBoosting_spinBox_stages.value(),
                                                       learning_rate=self.GBoosting_doubleSpinBox_rate.value(),
                                                       loss=GBoosting_switcher_loss.get(
                                                           self.GBoosting_comboBox1.currentIndex(), "switch out of bounds"),
                                                       max_depth=self.GBoosting_spinBox_depth.value(),
                                                       max_features=GBoosting_features))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Gradient Boosting",
             "parámetros": "número de etapas: "+str(self.GBoosting_spinBox_stages.value()) +
             "\nritmo de aprendizaje: "+str(self.GBoosting_doubleSpinBox_rate.value()) +
             "\nfunción de pérdida: "+self.GBoosting_comboBox1.currentText() +
             "\nprofundidad máxima: "+str(self.GBoosting_spinBox_depth.value()) +
             "\nmáximas características: "+("None" if GBoosting_features is None else GBoosting_features)},
            ignore_index=True)

    def loadMLP(self):
        layers = []
        for col in range(self.tableWidgetNeurons.columnCount()):
            layers.append(int(self.tableWidgetNeurons.item(0, col).text()))
        layers = tuple(layers)
        MLP_switcher_activation = {
            0: "relu",
            1: "tanh",
            2: "logistic",
            3: "identity"
        }
        MLP_switcher_rate = {
            0: "constant",
            1: "invscaling",
            2: "adaptive"
        }
        self.modelos.append(MLPClassifier(random_state=0,
                                          hidden_layer_sizes=layers,
                                          activation=MLP_switcher_activation.get(
                                              self.mlp_comboBox1.currentIndex(), "switch out of bounds"),
                                          learning_rate=MLP_switcher_rate.get(
                                              self.mlp_comboBox2.currentIndex(), "switch out of bounds"),
                                          max_iter=self.mlp_spinBox.value()))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Perceptrón Multicapa",
             "parámetros": "capas ocultas: "+str(layers) +
             "\nfunción de activación: "+self.mlp_comboBox1.currentText() +
             "\nritmo de aprendizaje: "+self.mlp_comboBox2.currentText() +
             "\nmáximo número de épocas: "+str(self.mlp_spinBox.value())},
            ignore_index=True)

    def loadLinearSVC(self):
        svc_switcher = {
            0: "squared_hinge",
            1: "hinge"
        }
        self.modelos.append(LinearSVC(random_state=0,
                                      loss=svc_switcher.get(
                                          self.svc_comboBox.currentIndex(), "switch out of bounds"),
                                      dual=self.dual_svc_checkBox.isChecked(),
                                      max_iter=self.svc_spinBox.value()))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Support Vector Lineal",
             "parámetros": "función de pérdida: "+self.svc_comboBox.currentText() +
             "\noptimización dual: "+str(self.dual_svc_checkBox.isChecked()) +
             "\nmáximas iteraciones: "+str(self.svc_spinBox.value())},
            ignore_index=True)

    def loadSVM(self):
        self.modelos.append(SVC(random_state=0,
                                kernel=self.svm_comboBox1.currentText(),
                                gamma='scale' if self.svm_comboBox2.currentIndex() == 0 else 'auto'))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Support Vector Machine",
             "parámetros": "tipo de kernel: "+self.svm_comboBox1.currentText() +
             "\ncoeficiente gamma: "+self.svm_comboBox2.currentText()},
            ignore_index=True)

    def loadSGD(self):
        sgd_switcher = {
            0: "optimal",
            1: "constant",
            2: "invscaling",
            3: "adaptive"
        }
        self.modelos.append(SGDClassifier(random_state=0,
                                          max_iter=self.sgd_spinBox.value(),
                                          learning_rate=sgd_switcher.get(
                                              self.sgd_comboBox.currentIndex(), "switch out of bounds"),
                                          eta0=self.sgd_doubleSpinBox_rate.value()))
        self.modelosDataframe = self.modelosDataframe.append(
            {"modelo": "Stochastic Gradient Descent",
             "parámetros": "máximas iteraciones: "+str(self.sgd_spinBox.value()) +
             "\nritmo de aprendizaje: "+self.sgd_comboBox.currentText() +
             "\nritmo de aprendizaje inicial (eta0): "+str(self.sgd_doubleSpinBox_rate.value())},
            ignore_index=True)
