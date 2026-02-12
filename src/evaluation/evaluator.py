import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def get_predictions(model, test_generator):
    print("Generating predictions...")
    test_generator.reset()
    
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    
    return y_true, y_pred_classes, y_pred_probs

def plot_classification_report(y_true, y_pred, class_names):
    print("\n--- Classification Report ---")
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="viridis", fmt=".2f")
    plt.title("Classification Report (Heatmap)")
    plt.show()
    
    print(classification_report(y_true, y_pred, target_names=class_names))

def plot_confusion_matrix(y_true, y_pred, class_names):
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curves(y_true, y_pred_probs, class_names):
    print("\n--- ROC Curves & AUC ---")
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], lw=2, 
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def evaluate_model(model, test_generator, emotion_dict):
    print("\n" + "="*40)
    print("      STARTING COMPREHENSIVE EVALUATION      ")
    print("="*40)
    
    class_names = [k for k, v in sorted(emotion_dict.items(), key=lambda item: item[1])]
    y_true, y_pred_classes, y_pred_probs = get_predictions(model, test_generator)
    
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    plot_classification_report(y_true, y_pred_classes, class_names)
    plot_roc_curves(y_true, y_pred_probs, class_names)
    
    print("\nEvaluation Complete.")