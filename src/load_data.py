import os
import pandas as pd
import pickle

# Define serialization functions
def pickle_serialize_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # Step 1: Load the CSV files
    data_path = 'data/raw/data.csv'
    label_path = 'data/raw/labels.csv'

    data = pd.read_csv(data_path)
    label = pd.read_csv(label_path)

    # Step 2: Prepare data for serialization
    X = data.drop(columns=['Unnamed: 0'])  # Drop the 'Unnamed: 0' column if it's not needed

    # Drop the 'Unnamed: 0' column from label if present
    if 'Unnamed: 0' in label.columns:
        label = label.drop(columns=['Unnamed: 0'])

    y = label['Class']  # Use the 'Class' column as the target

    # Step 3: Inspect the data
    print(data.info())
    print(label.info())

    # Serialize the features and target
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    pickle_serialize_object(os.path.join(output_dir, 'rna_seq_X.pkl'), X)
    pickle_serialize_object(os.path.join(output_dir, 'rna_seq_y.pkl'), y)

    # Step 4: Deserialize the data
    X = pickle_deserialize_object(os.path.join(output_dir, 'rna_seq_X.pkl'))
    y = pickle_deserialize_object(os.path.join(output_dir, 'rna_seq_y.pkl'))

    # Verify the deserialized data
    print("Deserialized X:")
    print(X.info())

    print("\nDeserialized y:")
    print(y.head())

if __name__ == "__main__":
    main()
