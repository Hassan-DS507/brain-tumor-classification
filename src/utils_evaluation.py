import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_data, steps):
    """Evaluate model on test dataset and print accuracy and loss."""
    test_loss, test_acc = model.evaluate(test_data, steps=steps)
    print(f"\n Test Accuracy: {test_acc:.4f}")
    print(f" Test Loss: {test_loss:.4f}")
    return test_loss, test_acc


def full_classification_report(model, test_data, steps, class_names=None, show_matrix=True):
    """Print classification report and plot confusion matrix."""
    y_true = []
    y_pred = []

    for images, labels in test_data.take(steps):
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    if show_matrix:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()
