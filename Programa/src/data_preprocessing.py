# PREPROCESAMIENTO DE LOS DATOS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from IPython.display import display
import matplotlib.pyplot as plt
from statistics import mean
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def load_texts(directory, category, textos=[], y=[]):
    names = os.listdir(directory)
    for n in names:
        file = open(directory+'/'+n, "r", encoding='utf-8', errors='ignore')
        textos.append(file.read())
        y.append(category)
    return textos, y


def token_stop_stem_lower(texts, stopWords=True, stemmer=True, minus=True, elim_num=True):
    spanish_stemmer = SnowballStemmer('spanish')
    esp_stops = set(stopwords.words('spanish'))
    processed_texts = []
    for x in texts:
        # tokenizo cada texto por palabras
        word_tokens = word_tokenize(x)
        word_stopw = []
        word_stemmed = []

        if(stopWords):
            # filtro el texto quitando los stopwords
            for token in word_tokens:
                if token not in esp_stops:
                    word_stopw.append(token)
        else:
            for token in word_tokens:
                word_stopw.append(token)

        if(stemmer):
            for token in word_stopw:
                # stemizo cada palabra suelta
                word_stemmed.append(spanish_stemmer.stem(token))
        else:
            for token in word_stopw:
                word_stemmed.append(token)

        # junto todas las palabras stemizadas separadas por espacios
        text_stemmed = ' '.join(word_stemmed)
        if(minus):
            # lo pongo todo en minúsculas
            text_stemmed = text_stemmed.lower()
        # eliminar numeros y caracteres especiales
        if(elim_num):
            # sustituyo todo lo que no sean palabras por espacios
            text_stemmed = re.sub("[^\wáéíóúÁÉÍÓÚñÑ]|[0-9]", " ", text_stemmed)

        # añado el texto procesado a mi array processed_texts
        processed_texts.append(text_stemmed)
    return processed_texts


def cv_ocurrences(processed_texts, maxdf=0.95, mindf=0.05):
    # COUNT VECTORIZER (BAG OF WORDS)
    # creo el countvectorizer con ocurrencia máxima del 95% y mínima del 5%
    cv = CountVectorizer(max_df=maxdf, min_df=mindf)
    # creo mi array X con la información del countvectorizer dentro y los processed_texts como datos de entrada
    equis = cv.fit_transform(processed_texts).toarray()
    # genero un dataframe para ver los datos generados en la matriz
    return pd.DataFrame(equis, columns=cv.get_feature_names())


def cv_TFIDF(processed_texts, maxdf=0.95, mindf=0.05):
    # TF-IDF VECTORIZER
    # creo el TF-IDF vectorizer con ocurrencia máxima del 95% y mínima del 5%
    cv_tfidf = TfidfVectorizer(max_df=maxdf, min_df=mindf)
    # creo mi array X_tfidf con la información del TF-IDF vectorizer dentro y los processed_texts como datos de entrada
    X_tfidf = cv_tfidf.fit_transform(processed_texts).toarray()
    # genero un dataframe para ver los datos generados en la matriz
    return pd.DataFrame(X_tfidf, columns=cv_tfidf.get_feature_names())


# Función para validar por el método de cross validation (10 folds) el modelo pasado
def crossValidationTest(clf, name, X_train, y_train, folds):
    y_train_pred_cv = cross_val_predict(clf, X_train, y_train, cv=folds)
    target_names = ['despoblación', 'no despoblación']
    print('\nCROSS VALIDATION ', name)
    print(classification_report(y_train, y_train_pred_cv, target_names=target_names))
    print(confusion_matrix(y_train, y_train_pred_cv))
    accuracy = accuracy_score(y_train, y_train_pred_cv)
    print('accuracy score: '+str(accuracy))
    return accuracy

# Función para entrenar el modelo con todo el train set y ver los resultados de predicciones del train set y el test set


def predictTrainAndTest(clf, name, X_train, y_train, X_test, y_test):
    target_names = ['despoblación', 'no despoblación']
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    print('\n', name, ' TRAIN SET')
    print(classification_report(y_train, y_train_pred, target_names=target_names))
    print(confusion_matrix(y_train, y_train_pred))
    accTrain = accuracy_score(y_train, y_train_pred)
    print('accuracy score: '+str(accTrain))

    y_test_pred = clf.predict(X_test)
    print('\n', name, ' TEST SET')
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_test_pred))
    accTest = accuracy_score(y_test, y_test_pred)
    print('accuracy score: '+str(accTest))
    return accTrain, accTest

# Punción para imprimir los resultados de un entrenamiento de modelos utilizando Grid Search cross validation


def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):
    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results)
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(
            param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()
    return scores_df

# Función para aplicar al modelo pasado una validación cruzada en un grid con los diferentes parámetros introducidos


def gridSearchCrossValidation(clf, parameters, param_name, X_train, y_train, folds):
    grid = GridSearchCV(clf, parameters, cv=folds)
    grid.fit(X_train, y_train)
    print(grid.cv_results_)
    return GridSearch_table_plot(grid, param_name, negative=False)


def saveResults(model, accCV, accTrain, accTest):
    f = open("../results.csv", "a")
    if(os.stat("../results.csv").st_size == 0):
        f.write(
            "Model;Cross Validation Accuracy;Train Set Accuracy;Test Set Accuracy;Mean\n")
    row = str(model)+";"+str(accCV)+";"+str(accTrain)+";" + \
        str(accTest)+";"+str(mean([accCV, accTrain, accTest]))+"\n"
    row = re.sub("\.", ",", row)
    f.write(row)
    f.close()


class PandasModel(QtCore.QAbstractTableModel):
    # Class to populate a table view with a pandas dataframe
    def __init__(self, data, header, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.header = header
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole and self.header:
            return self._data.columns[col]
        return QtCore.QAbstractTableModel.headerData(self, col, orientation, role)
