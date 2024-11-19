from sklearn.metrics import (
    f1_score, recall_score, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()