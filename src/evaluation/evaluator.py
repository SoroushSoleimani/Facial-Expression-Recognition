import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_generator, emotion_dict):
    print("\n--- Starting Evaluation ---")
    
    class_names = [k for k, v in sorted(emotion_dict.items(), key=lambda item: item[1])]
    
    test_generator.reset()
    
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()