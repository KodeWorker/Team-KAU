import cv2
import numpy as np

def dsc(y_pred, y_true, smooth=1):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return (np.sum((y_pred * y_true)) * 2.0 + smooth) / (np.sum(y_pred) + np.sum(y_true)+ smooth)

if __name__ == "__main__":
     
     msk_path = r"D:\code\AIdea\data\temp\valid\masks\BIK2X7OK_63.png"
     
     msk = cv2.imread(msk_path)
     msk = msk / 255
     
     print(dsc(msk, msk))
     