import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix

def precision(y_pred, y_true):
    tp = np.sum(y_pred[y_true == 1])
    fp = np.sum(y_pred[y_true == 0])
    
    if (tp + fp) == 0:
        return 0
    else:
        return tp / (tp + fp)

def recall(y_pred, y_true):
    tp = np.sum(y_pred[y_true == 1])
    fn = np.sum(y_pred[y_true == 1] == 0)
    if (tp + fn) == 0:
        return 0
    else:
        return tp / (tp + fn)

def dice(y_pred, y_true):
    precision_ = precision(y_pred, y_true)
    recall_ = recall(y_pred, y_true)
    if (precision_ + recall_) == 0:
        return 0
    else:
        return 2*precision_*recall_/(precision_ + recall_)
    
def dice_score(out_file, label):
    
    pred_image = nib.load(out_file)
    true_image = nib.load(label)
    
    y_pred = pred_image.get_fdata()
    y_true = true_image.get_fdata()
    
    score = dice(y_pred, y_true)
    
    print(score) #!
    
    return score