import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class SF_PD:
    def __init__(self, filename):
        self.filename = filename
        self.dataset = None
        self.selected_features = []
        self.feature_scores = None

    def load_data(self):
        self.dataset = pd.read_csv(self.filename).drop_duplicates()
        self.dataset["diagnosis"] = self.dataset["diagnosis"].map({'B': 0, 'M': 1})

    def select_features(self, threshold=0.2):
        X = self.dataset.drop(columns=["diagnosis"])
        y = self.dataset["diagnosis"]
        X = X.apply(pd.to_numeric, errors='coerce')

        selector = SelectKBest(mutual_info_classif, k='all')
        selector.fit(X, y)

        self.feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
        self.feature_scores = self.feature_scores.sort_values(by='Score', ascending=False)

        self.selected_features = self.feature_scores[self.feature_scores['Score'] > threshold]

        if self.selected_features.empty:
            print("Tidak ada fitur dengan probabilitas lebih dari threshold!")


    def get_feature_probabilities(self):
        return self.feature_scores

    def get_selected_features(self):
        return self.selected_features
