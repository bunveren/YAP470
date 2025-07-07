import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
from sklearn.decomposition import PCA

def train_svm(X_train_scaled, y_train, C=1.0, gamma='scale', kernel='rbf'):
    if X_train_scaled is None or y_train is None or X_train_scaled.shape[0] == 0:
         print("svm train edilemedi! training data bos")
         return None
    svm_model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
    start_time = time.time()
    svm_model.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"svm model training suresi: {end_time - start_time:.2f} saniye")

    return svm_model

def evaluate_model(model, X_test_scaled, y_test, model_name="Model"):
    if model is None or X_test_scaled is None or y_test is None or X_test_scaled.shape[0] == 0:
         print(f"{model_name} eval edilemedi! data bos")
         return
    y_pred = model.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (Sensitivity): {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix ({model_name}):")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nclass report ({model_name}):")
    print(classification_report(y_test, y_pred))

def tune_svm_hyperparameters(X_train_scaled, y_train, param_grid):
    if X_train_scaled is None or y_train is None or X_train_scaled.shape[0] == 0:
         return None

    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='f1', 
        n_jobs=-1,
        verbose=2
    )

    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"\nGridSearchCV {end_time - start_time:.2f} saniye surdu")
    print("\nen iyi parametreler:")
    print(grid_search.best_params_)
    print("\nen iyi cv f1 skoru:")
    print(grid_search.best_score_)

    return grid_search

def main():
    dataset_choice = 'kaggle' 
    try:
        config = get_config(dataset_choice)
    except ValueError:
        return 
    
    feature_params = get_feature_params()

    processed_data_dict = load_and_preprocess_data(config)
    if processed_data_dict is None or processed_data_dict.get('labels', []).shape[0] == 0:
        print("data load sorunu")
        return 
    
    features_array, labels_array = extract_features(processed_data_dict, feature_params)
    if features_array.shape[0] == 0:
         print("feat exc sorunu")
         return 

    X_train, X_test, y_train, y_test = split_data(features_array, labels_array)
    if X_train is None:
         print("split sorunu")
         return
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    if X_train_scaled is None:
         print("scaling sorunu")
         return
    initial_svm_model = train_svm(X_train_scaled, y_train)
    if initial_svm_model:
        evaluate_model(initial_svm_model, X_test_scaled, y_test, model_name="SVM")
    else:
         print("svm train sorunu")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    tuned_grid_search = tune_svm_hyperparameters(X_train_scaled, y_train, param_grid)

    if tuned_grid_search:
        best_svm_model = tuned_grid_search.best_estimator_
        evaluate_model(best_svm_model, X_test_scaled, y_test, model_name="Tuned SVM (GridSearch)")
    else:
         print("hyper tuning sorunu")

if __name__ == "__main__":
    main()    
    