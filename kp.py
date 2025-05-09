import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI untuk server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class KP:
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
        self.selected_features = [
            "perimeter_worst", "area_worst", "radius_worst", "concave points_mean", 
            "concave points_worst", "perimeter_mean", "concavity_mean", "radius_mean", 
            "area_mean", "area_se", "concavity_worst", "perimeter_se", 
            "radius_se", "compactness_worst", "compactness_mean"
        ]

    def load_data(self):
        """Load dataset from the provided CSV file."""
        self.dataset = pd.read_csv(self.filename)

    def kfold_validation(self):
        """Perform k-fold cross-validation and store the best model and metrics."""
        X = self.dataset[self.selected_features]
        y = self.dataset["diagnosis"]

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)

        for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = SVC(kernel="linear", random_state=42)
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

    def save_confusion_matrix(self):
        """Save the best confusion matrix as an image."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title(f"Confusion Matrix - Fold {self.best_fold}")
        plt.savefig("static/img/cm_kp.png")
        plt.close()

    def get_results(self):
        """Return results and best metrics."""
        return self.results, self.best_fold, self.best_metrics, "static/img/cm_kp.png"
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

    def predict(self, input_data):
        """Receive input data as a dictionary, make prediction using the best model."""
        input_df = pd.DataFrame([input_data])  # Convert input to DataFrame
        
        # Get minimum values from benign class for each feature
        benign_min = self.dataset[self.dataset["diagnosis"] == "0"][self.selected_features].min()

        # Count how many features are below the benign minimum values
        count_below_min = sum(input_df.iloc[0] < benign_min)
        
        # If all features are below benign min, classify as Normal
        if count_below_min == len(self.selected_features):
            return "Normal"
        
        # If more than half the features are below benign min, classify as Normal
        if count_below_min >= len(self.selected_features) / 2:
            return "Normal"
        
        # Otherwise, use the best model to predict
        prediction = self.best_model.predict(input_df)[0]  
        return "B (Benign/Jinak)" if prediction == 0 else "M (Malignant/Ganas)"  # Convert prediction result

    def get_best_accuracy(self):
        """Return the best accuracy in percentage."""
        return round(self.best_accuracy * 100, 2)  # Convert to percentage
