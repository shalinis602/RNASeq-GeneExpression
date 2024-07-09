import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def pickle_deserialize_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickle_serialize_object(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def main():
    # Deserialize the input
    input_dir = 'data/processed'
    X_train_resampled = pickle_deserialize_object(os.path.join(input_dir, 'X_train_resampled.pkl'))    
    
    input_dir2 = 'data/processed/split_data'
    X_val = pickle_deserialize_object(os.path.join(input_dir2, 'X_val.pkl'))
    X_test = pickle_deserialize_object(os.path.join(input_dir2, 'X_test.pkl'))

    if X_train_resampled is None:
        print("Failed to load resampled datasets.")
        return

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)

    # Apply PCA with 7 components
    pca = PCA(n_components=7)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Transform validation and test sets
    X_val_numeric = X_val.values
    X_val_pca = pca.transform(scaler.transform(X_val_numeric))

    X_test_numeric = X_test.values
    X_test_pca = pca.transform(scaler.transform(X_test_numeric))

    # Save X_train_pca dataset
    output_dir = 'data/processed/transformed'
    os.makedirs(output_dir, exist_ok=True)
    pickle_serialize_object(os.path.join(output_dir, 'X_train_pca.pkl'), X_train_pca)
    pickle_serialize_object(os.path.join(output_dir, 'X_val_pca.pkl'), X_val_pca)
    pickle_serialize_object(os.path.join(output_dir, 'X_test_pca.pkl'), X_test_pca)

    # Plot explained variance for the selected number of components
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend(loc='best')
    plt.grid(True)

    plot_filename = os.path.join(output_dir, 'explained_variance_plot.png')
    plt.savefig(plot_filename)
    plt.show()
    
    # Print explained variance values for the selected components
    print("Explained variance ratios for each component:")
    for i, var in enumerate(explained_variance, start=1):
        print(f"Principal Component {i}: {var:.4f}")

    print("\nCumulative explained variance ratios:")
    for i, cum_var in enumerate(cumulative_explained_variance, start=1):
        print(f"Principal Component {i}: {cum_var:.4f}")

if __name__ == "__main__":
    main()