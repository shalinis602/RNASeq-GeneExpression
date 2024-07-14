import os
import sys, datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Generate a timestamp for this run
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/random_forest_{timestamp}.out"

# Redirect stdout and stderr to the log file
sys.stdout = open(f'{log_file}', 'a')
sys.stderr = open(f'{log_file}', 'a')

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_serialize_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def main():
    # Deserialize the input
    input_dir = 'data/processed'
    X_train_pca = pickle_deserialize_object(os.path.join(input_dir, 'X_train_pca.pkl'))
    y_train_resampled = pickle_deserialize_object(os.path.join(input_dir, 'y_train_resampled.pkl'))    

    input_dir2 = 'data/processed/transformed'
    X_val_pca = pickle_deserialize_object(os.path.join(input_dir2, 'X_val_pca.pkl'))
    X_test_pca = pickle_deserialize_object(os.path.join(input_dir2, 'X_test_pca.pkl'))

    input_dir3 = 'data/processed/split_data'
    y_val = pickle_deserialize_object(os.path.join(input_dir3, 'y_val.pkl'))
    y_test = pickle_deserialize_object(os.path.join(input_dir3, 'y_test.pkl'))

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [2, 4]
    }

    # Initialize and fit RandomForestClassifier with GridSearchCV
    rf = RandomForestClassifier(random_state=1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_pca, y_train_resampled)

    # Get the best estimator
    best_rf = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_classification_report = classification_report(y_val, y_val_pred)
    val_confusion_matrix = confusion_matrix(y_val, y_val_pred)

    # Evaluate on test set
    y_test_pred = best_rf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_classification_report = classification_report(y_test, y_test_pred)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)

    # Write results to a file
    output_filename = 'results/random_forest.txt'
    with open(output_filename, 'w') as f:
        f.write(f"Validation Accuracy: {val_accuracy:.2f}\n")
        f.write("Validation Classification Report:\n")
        f.write(val_classification_report + '\n')
        f.write("Validation Confusion Matrix:\n")
        f.write(str(val_confusion_matrix) + '\n\n')
        
        f.write(f"Test Accuracy: {test_accuracy:.2f}\n")
        f.write("Test Classification Report:\n")
        f.write(test_classification_report + '\n')
        f.write("Test Confusion Matrix:\n")
        f.write(str(test_confusion_matrix) + '\n')

if __name__ == "__main__":
    main()