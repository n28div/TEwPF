from typing import List
import numpy as np

def accuracy(y: List[float], y_pred: List[float], accuracy_type: str = "1") -> bool:
    """
    Compute the accuracy of a prediction. A prediction is considered as 
    correct if it is within the 4% of the original label.
    Accuracy multiples can be taken into account by setting the accuracy_type
    parameter: "1" does not take into account any BPM multiple while "2" computes
    the accuracy by also taking into account 1/3, 1/2, 2, 3 times the prediction
    (binary and ternary multiples).


    Args:
        y (List[float]): True BPM
        y_pred (List[float]): BPM Prediction
        accuracy_type (str, optional): Accuracy type. Either "1" or "2" (default).

    Returns:
        bool: True if the prediction is correct, False otherwise
    """
    assert accuracy_type in ["1", "2"]
    alphas = [1] if accuracy_type == "1" else [1/3, 1/2, 1, 2, 3]

    return np.mean([
        np.any([(a * pred * 0.96) < truth < (a * pred * 1.04) for a in alphas])
        for pred, truth in zip(y_pred, y)
    ])


def octave_error(y: List[float], y_pred: List[float], oe_type: str = "1") -> bool:
    """
    Compute the octave error of a prediction. It is computed as log2(y/y_pred).
    Accuracy multiples can be taken into account by setting the oe_type parameter: 
    "1" does not take into account any BPM multiple while "2" computes
    the accuracy by also taking into account 1/3, 1/2, 2, 3 times the prediction
    (binary and ternary multiples).


    Args:
        y (List[float]): True BPM
        y_pred (List[float]): BPM Prediction
        accuracy_type (str, optional): Accuracy type. Either "1" or "2" (default).

    Returns:
        bool: True if the prediction is correct, False otherwise
    """
    assert oe_type in ["1", "2"]
    alphas = [1] if oe_type == "1" else [1/3, 1/2, 1, 2, 3]

    if oe_type == "1":
        oe = [np.log2(pred / y) for pred, truth in zip(y_pred, y)]
    else:
        oe = [min([np.log2((y_pred * a) / y) for a in alphas]) 
              for pred, truth in zip(y_pred, y)]

    return oe


def absolute_octave_error(y: List[float], y_pred: List[float], oe_type: str = "1") -> bool:
    """
    Compute the basolute octave error of a prediction as |OE(y, y_pred)|.
    Accuracy multiples can be taken into account by setting the oe_type parameter: 
    "1" does not take into account any BPM multiple while "2" computes
    the accuracy by also taking into account 1/3, 1/2, 2, 3 times the prediction
    (binary and ternary multiples).


    Args:
        y (List[float]): True BPM
        y_pred (List[float]): BPM Prediction
        accuracy_type (str, optional): Accuracy type. Either "1" or "2" (default).

    Returns:
        bool: True if the prediction is correct, False otherwise
    """
    assert oe_type in ["1", "2"]
    oe = np.abs(octave_error(y, y_pred, oe_type=oe_type))
    return oe