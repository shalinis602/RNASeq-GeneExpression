import os
import pickle
from sklearn.model_selection import train_test_split

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
       
def pickle_serialize_object(filename, obj):
	with open(filename, 'wb') as f:
          pickle.dump(obj, f)
    
def main():
    # Deserialize the serialized data
    input_dir = 'data/processed'

    X = pickle_deserialize_object(os.path.join(input_dir, 'rna_seq_X.pkl'))
    y = pickle_deserialize_object(os.path.join(input_dir, 'rna_seq_y.pkl'))

    # Split data into training, valid and test sets (80% training, 10% test, 10% valid)
    X_train, X_test_temp, y_train, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test_temp, y_test_temp, test_size=0.5, random_state=1)
    
    # Save the split datasets using pickle
    output_dir = 'data/processed/split_data'
    os.makedirs(output_dir, exist_ok=True)
    
    pickle_serialize_object(os.path.join(output_dir, 'X_train.pkl'), X_train)
    pickle_serialize_object(os.path.join(output_dir, 'X_test.pkl'), X_test)
    pickle_serialize_object(os.path.join(output_dir, 'X_val.pkl'), X_val)
    pickle_serialize_object(os.path.join(output_dir, 'y_train.pkl'), y_train)
    pickle_serialize_object(os.path.join(output_dir, 'y_test.pkl'), y_test)
    pickle_serialize_object(os.path.join(output_dir, 'y_val.pkl'), y_val)
    
    # Check the shapes of the resulting datasets
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

if __name__ == "__main__":
    main()
