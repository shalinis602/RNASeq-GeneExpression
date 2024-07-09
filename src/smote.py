import os
import pickle
import numpy as np  
from imblearn.over_sampling import SMOTE

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_serialize_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)    
        
def main():
    # Deserialize the serialized data
    input_dir = "data/processed/split_data"

    X_train = pickle_deserialize_object(os.path.join(input_dir, 'X_train.pkl'))
    y_train = pickle_deserialize_object(os.path.join(input_dir, 'y_train.pkl'))

    # Initialize SMOTE
    smote = SMOTE(random_state=1)

    # Resample X and y using SMOTE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Save the resampled datasets
    output_dir = 'data/processed'

    pickle_serialize_object(os.path.join(output_dir, 'X_train_resampled.pkl'), X_train_resampled)
    pickle_serialize_object(os.path.join(output_dir, 'y_train_resampled.pkl'), y_train_resampled)

    # Check the class distribution after resampling
    print("Resampled class distribution:")
    print(np.unique(y_train_resampled, return_counts=True))

if __name__ == "__main__":
    main()
