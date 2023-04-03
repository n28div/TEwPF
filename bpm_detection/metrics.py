from typing import List
import numpy as np

def accuracy(X: List[float], y: List[float], accuracy_type: str = "1") -> bool:
    """
    Compute the accuracy of a prediction. A prediction is considered as 
    correct if it is within the 4% of the original label.
    Accuracy multiples can be taken into account by setting the accuracy_type
    parameter: "1" does not take into account any BPM multiple while "2" computes
    the accuracy by also taking into account 1/3, 1/2, 2, 3 times the prediction
    (binary and ternary multiples).


    Args:
        X (List[float]): BPM Prediction
        y (List[float]): True BPM
        accuracy_type (str, optional): Accuracy type. Either "1" or "2" (default).

    Returns:
        bool: True if the prediction is correct, False otherwise
    """
    assert accuracy_type in ["1", "2"]
    alphas = [1] if accuracy_type == "1" else [1/3, 1/2, 1, 2, 3]

    return np.mean([
        np.any([(a * truth * 0.96) < pred < (a * truth * 1.04) for a in alphas])
        for pred, truth in zip(X, y)
    ])