import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI untuk server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class C45:
    def __init__(self, filename):
        self.filename = filename
        self.dataset = None
        self.k = 10
        self.best_fold = -1
        self.best_accuracy = 0
        self.best_cm = None
        self.best_metrics = {}
        self.results = []
        self.best_model = None
        self.rules = []

    def load_data(self):
        """Load dataset from the provided CSV file."""
        self.dataset = pd.read_csv(self.filename)

    def kfold_validation(self):
        X = self.dataset.drop(columns=["stunting"])
        y = self.dataset["stunting"]

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = DecisionTreeClassifier(criterion='entropy', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()  # Make sure the label order matches your dataset

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            self.results.append({
                'fold': fold,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_fold = fold
                self.best_cm = cm
                self.best_metrics = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "Akurasi": acc,
                                     "Presisi": precision, "Recall": recall, "F1-Score": f1}
                self.best_model = model

        self.save_confusion_matrix()
        self.extract_rules()

    def save_confusion_matrix(self):
        """Save the best confusion matrix as an image."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak Stunting", "Stunting"], yticklabels=["Tidak Stunting", "Stunting"])
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title(f"Confusion Matrix - Fold {self.best_fold}")
        plt.savefig("static/img/cm_c45.png")
        plt.close()

    def get_results(self):
        """Return results and best metrics."""
        return self.results, self.best_fold, self.best_metrics, "static/img/cm_c45.png"
    def get_kfold_aggregates(self):
        avg_accuracy = np.mean([result['accuracy'] for result in self.results])
        avg_precision = np.mean([result['precision'] for result in self.results])
        avg_recall = np.mean([result['recall'] for result in self.results])
        avg_f1 = np.mean([result['f1'] for result in self.results])

        return {
            'Rata-rata Akurasi': avg_accuracy,
            'Rata-rata Presisi': avg_precision,
            'Rata-rata Recall': avg_recall,
            'Rata-rata F1-Score': avg_f1
        }

    def predict(self, gender, age, weight, height):
        """Predict status gizi berdasarkan input pengguna."""
        if height < 43:
            return 1  # Langsung return 1 jika tinggi di bawah 43 cm

        if self.best_model is None:
            raise Exception("Model belum dilatih. Jalankan kfold_validation() terlebih dahulu.")

        input_data = np.array([[age, gender, weight, height]])
        prediction = self.best_model.predict(input_data)
        return prediction[0]

    def get_best_accuracy(self):
        """Return the best accuracy in percentage."""
        return round(self.best_accuracy * 100, 2)  # Convert to percentage
    
    def extract_rules(self):
        """Extract rules from the trained decision tree model."""
        self.rules = []
        if self.best_model:
            # Get the tree structure
            tree_ = self.best_model.tree_
            feature_names = self.dataset.drop(columns=["stunting"]).columns
            self._extract_tree_rules(tree_, feature_names)

    def _extract_tree_rules(self, tree, feature_names, node_id=0, rule=""):
        """Recursively extract rules from the decision tree."""
        # If it's a leaf node, append the rule
        if tree.feature[node_id] == _tree.TREE_UNDEFINED:
            class_name = tree.value[node_id].argmax()
            self.rules.append(f"{rule} -> {class_name}")
        else:
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            left_rule = f"{rule} AND {feature_name} <= {threshold}"
            right_rule = f"{rule} AND {feature_name} > {threshold}"

            self._extract_tree_rules(tree, feature_names, tree.children_left[node_id], left_rule)
            self._extract_tree_rules(tree, feature_names, tree.children_right[node_id], right_rule)

    def get_rules(self):
        """Return the decision tree rules."""
        return self.rules
