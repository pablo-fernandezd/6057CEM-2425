import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, \
    roc_curve


class AdvancedAnalytics:
    def __init__(self, class_names):
        self.class_names = class_names

    def full_analysis(self, model, X_test, y_test):
        """Evaluate the model comprehensively."""
        # Predict probabilities and classes
        y_probs = model.predict(X_test)
        y_pred = y_probs.argmax(axis=1)

        # Print classification report
        print("🔬 Comprehensive Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        # Calculate additional metrics
        print(f"\n📈 ROC AUC (OvR): {roc_auc_score(y_test, y_probs, multi_class='ovr'):.4f}")

        # Visualize confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)

        # Visualize ROC curves
        self._plot_roc_curves(y_test, y_probs)

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Visualize confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def _plot_roc_curves(self, y_true, y_probs):
        """Visualize ROC curves for multi-class classification."""
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(self.class_names)))

        plt.figure(figsize=(10, 8))
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC={roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.show()
