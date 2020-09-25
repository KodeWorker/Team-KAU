from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import dump
import random
import os

if __name__ == "__main__":

    n_estimators = 1000
    max_depth = 3
    random_state = 777
    model_path = "rf.model"
    feature_dir = "rf_features"
    
    X_positive = np.load(os.path.join(feature_dir, "positive.npy"))
    X_negative = np.load(os.path.join(feature_dir, "negative.npy"))
    
    y_positive = np.ones(len(X_positive))
    y_negative = np.ones(len(X_negative))
    
    X, y = np.vstack((X_positive, X_negative)), np.append(y_positive, y_negative)
    
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X, y)
    
    dump(rf, model_path)