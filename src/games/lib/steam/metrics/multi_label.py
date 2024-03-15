import torch
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


def create_multi_label_metrics(
    positive_label_threshold: float = 1.0, prediction_threshold: float = 0.5
):
    """
    A function that returns another function based on the meta parameters that calculates multiple metrics for a multi-label
    classification problem based on given predictions and labels.

    Parameters:
    - positive_label_threshold: In the case of label smoothing, positive examples might not be exactly 1.0. This parameter lets us customize when we consider an expected label to be 1
    - prediction_threshold: When do we consider a _predicted_ label to be 1

    Returns:
    A dictionary containing the following metrics:
    - f1_micro: Micro-averaged F1 score
    - f1_macro: Macro-averaged F1 score
    - f1_weighted: Weighted F1 score
    - roc_auc: ROC AUC score
    - accuracy: Accuracy score
    """

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def calculate(predictions: np.ndarray, labels: np.ndarray):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))

        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= prediction_threshold)] = 1

        # finally, compute metrics
        y_true = np.zeros(labels.shape)
        y_true[np.where(labels >= positive_label_threshold)] = 1

        f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        roc_auc = roc_auc_score(y_true, y_pred, average="micro")
        accuracy = accuracy_score(y_true, y_pred)

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
        }

    return calculate
