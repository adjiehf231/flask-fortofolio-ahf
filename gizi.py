import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI untuk server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GIZI:
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


    def load_data(self):
        """Load dataset from the provided CSV file."""
        self.dataset = pd.read_csv(self.filename)
        
    def kfold_validation(self):
        """Perform k-fold cross-validation and store the best model and metrics."""
        X = self.dataset.drop(columns=["status"])
        y = self.dataset["status"]

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = KNeighborsClassifier(n_neighbors=10, weights="distance", algorithm="ball_tree", metric="euclidean")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            self.results.append({
                'fold': fold,
                'confusion_matrix': cm.tolist(),  # simpan cm jika ingin
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            if acc > self.best_accuracy:
                self.best_accuracy = acc
                self.best_fold = fold
                self.best_cm = cm
                self.best_metrics = {
                    "Akurasi": acc,
                    "Presisi": precision,
                    "Recall": recall,
                    "F1-Score": f1
                }
                self.best_model = model

        self.save_confusion_matrix()

    def save_confusion_matrix(self):
        """Save the best confusion matrix as an image."""
        plt.figure(figsize=(8, 6))
        labels = ['Gizi Buruk', 'Gizi Kurang', 'Gizi Normal', 'Gizi Baik', 'Gizi Sangat Baik']
        sns.heatmap(self.best_cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title(f"Confusion Matrix - Fold {self.best_fold}")
        plt.savefig("static/img/cm_gizi.png")
        plt.close()

    def get_results(self):
        """Return results and best metrics."""
        return self.results, self.best_fold, self.best_metrics, "static/img/cm_gizi.png"
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

    def predict(self, age, gender, weight, height):
        """Predict status gizi berdasarkan input pengguna."""
        if self.best_model is None:
            raise Exception("Model belum dilatih. Jalankan kfold_validation() terlebih dahulu.")

        input_data = np.array([[age, gender, weight, height]])
        prediction = self.best_model.predict(input_data)
        return prediction[0]



    def get_best_accuracy(self):
        """Return the best accuracy in percentage."""
        return round(self.best_accuracy * 100, 2)  # Convert to percentage
