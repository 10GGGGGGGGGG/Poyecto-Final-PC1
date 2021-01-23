# PREPROCESAMIENTO DE LOS DATOS
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
import os
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
