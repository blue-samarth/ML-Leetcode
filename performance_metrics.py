
def performance_metrics(actual : list[int] , predicted : list[int]) -> tuple :
    """
    This function calculates the performance metrics of a classification model.
    like Confusion_Matrix , Accuracy , Precision , Recall , F1 Score , Specificity , Negative Predictive Value .
    Args:
        actual : list[int] : The actual target values.
        predicted : list[int] : The predicted target values.
    Returns:
        tuple : A tuple containing the performance metrics.
    """
    n : int = len(actual)
    m : int = len(predicted)

    if n != m: return ([[0 , 0],[0 , 0]],0,0,0,0)

    TP : int = 0
    TN : int = 0
    FP : int = 0
    FN : int = 0

    for i in range(n):
        if actual[i] == 1 and predicted[i] == 1:
            TP += 1
        elif actual[i] == 0 and predicted[i] == 0:
            TN += 1
        elif actual[i] == 0 and predicted[i] == 1:
            FP += 1
        elif actual[i] == 1 and predicted[i] == 0:
            FN += 1
    
    confusion_matrix : list[list[int]] = [[TP, FN], [FP, TN]]
    accuracy : float = (TP + TN) / n
    precision : float = TP / (TP + FP) if TP + FP != 0 else 0
    recall : float = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score : float = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    specificity : float = TN / (TN + FP) if TN + FP != 0 else 0
    negative_predictive_value : float = TN / (TN + FN) if TN + FN != 0 else 0

    return confusion_matrix, round(accuracy, 3), round(f1_score, 3), round(specificity, 3), round(negative_predictive_value, 3)