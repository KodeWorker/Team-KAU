import cv2
import numpy as np

def dsc(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

if __name__ == "__main__":
     
     msk_path = r"D:\code\AIdea\data\temp\valid\masks\BIK2X7OK_63.png"
     
     msk = cv2.imread(msk_path)
     msk[msk==255] = 1.0
     
     print(dsc(msk, msk))
     