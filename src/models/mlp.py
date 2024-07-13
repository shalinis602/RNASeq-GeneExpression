import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_serialize_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def main():
    print("Starting main function")
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

    # Initialise anc configure the MLPClassifier
    mlp = MLPClassifier(random_state=1)

    # Train the MLPClassifier on Training Data
    mlp.fit(X_train_pca, y_train_resampled)

    # Evaluate the model on Validation Data
    y_val_pred = mlp.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_classification_report = classification_report(y_val, y_val_pred)
    val_confusion_matrix = confusion_matrix(y_val, y_val_pred)

    # Tune Hyperparamters with GridSearchCV (Optional)
    param_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }

    grid_search = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
    grid_search.fit(X_train_pca, y_train_resampled)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the model with the best parameters
    mlp = grid_search.best_estimator_

    # Evaluate on validation data
    y_val_pred_tuned = mlp.predict(X_val_pca)
    val_accuracy_tuned = accuracy_score(y_val, y_val_pred_tuned)
    val_classification_report_tuned = classification_report(y_val, y_val_pred_tuned)
    val_confusion_matrix_tuned = confusion_matrix(y_val, y_val_pred_tuned)

    # Write results to a file
    output_filename = 'results/mlp.txt'
    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write(f"Validation Accuracy: {val_accuracy:.2f}\n")
            f.write("Validation Classification Report:\n")
            f.write(val_classification_report + '\n')
            f.write("Validation Confusion Matrix:\n")
            f.write(str(val_confusion_matrix) + '\n\n')

            f.write(f"Best Parameters: {best_params}\n")
            f.write(f"Validation Accuracy after tuning: {val_accuracy_tuned:.2f}\n")
            f.write("Validation Classification Report after tuning:\n")
            f.write(val_classification_report_tuned + '\n')
            f.write("Validation Confusion Matrix after tuning:\n")
            f.write(str(val_confusion_matrix_tuned) + '\n\n')
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    main()