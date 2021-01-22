self.processed_texts = token_stop_stem_lower(textos, stopWords=self.stopW.isChecked(
        ), stemmer=self.stemming.isChecked(), minus=self.minusc.isChecked(), elim_num=self.elim_num.isChecked())
        self.labelPrep.setStyleSheet(
            "QLabel {font-weight: bold;color: rgb(0, 221, 0);}")
        self.labelPrep.setText("¡Listo!")