import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, median_absolute_error
import torch.nn.functional as F
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder
from train_sync_offset_detector import SimilarityMatrixClassifier as _SimilarityMatrixClassifier

# This avoids running the argument parser
SimilarityMatrixClassifier = _SimilarityMatrixClassifier

# Function to calculate the baseline sync offset using the median approach


def calculate_median_offset(softmaxed_sim_12):
    predict = softmaxed_sim_12.argmax(axis=1)
    ground = np.arange(softmaxed_sim_12.shape[0])

    frames = predict - ground
    median_frames = np.median(frames)
    return median_frames


def pad_matrices(matrices, target_size):
    padded_matrices = []
    for matrix in matrices:
        original_height, original_width = matrix.shape
        pad_height = max(0, target_size - original_height)
        pad_width = max(0, target_size - original_width)

        # Pad and truncate as necessary
        padded_matrix = np.pad(
            matrix, ((0, pad_height), (0, pad_width)), mode='constant')
        padded_matrix = padded_matrix[:target_size,
                                      :target_size]  # Truncate if necessary

        # Print shapes for debugging
        # print(
        #     f'Original matrix shape: {matrix.shape}, Padded matrix shape: {padded_matrix.shape}')

        # Flatten for logistic regression
        padded_matrices.append(padded_matrix.flatten())
        # print(f'Flattened padded matrix shape: {padded_matrices[-1].shape}')

    return np.vstack(padded_matrices)


if __name__ == "__main__":
    # Add argument parsing
    parser = argparse.ArgumentParser(
        description='Evaluate sync offset detector')
    parser.add_argument('--model_prefix', type=str, required=True,
                        help='Prefix for loading the trained model (e.g., "ntu")')
    parser.add_argument('--data_prefix', type=str, required=True,
                        help='Prefix for loading validation data (e.g., "h36m")')
    parser.add_argument('--models', type=str, nargs='+', choices=['log_reg', 'mlp', 'cnn'],
                        default=['log_reg', 'mlp', 'cnn'],
                        help='Models to evaluate (choices: log_reg, mlp, cnn)')
    args = parser.parse_args()

    # Load prepared data using data_prefix
    X = np.load(f'{args.data_prefix}_val_softmaxed_sim_12.npy',
                allow_pickle=True)
    y = np.load(
        f'{args.data_prefix}_val_softmaxed_sim_12_labels.npy', allow_pickle=True)

    print(f'Shape of X: {X.shape}, Shape of y: {y.shape}')

    # Pad matrices to target_size*target_size
    X_padded = pad_matrices(X, target_size=256)
    print(f'Shape of X_padded: {X_padded.shape}')

    X_val = X_padded
    y_val = y

    # Load only the specified models
    models = {}
    if 'log_reg' in args.models:
        with open(f'{args.model_prefix}_log_reg_model.pkl', 'rb') as file:
            models['log_reg'] = pickle.load(file)

    if 'mlp' in args.models:
        with open(f'{args.model_prefix}_mlp_model.pkl', 'rb') as file:
            models['mlp'] = pickle.load(file)

    if 'cnn' in args.models:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models['cnn'] = SimilarityMatrixClassifier()
        models['cnn'].load_state_dict(torch.load(
            f'{args.model_prefix}_cnn_model.pt'))
        models['cnn'].to(device)
        models['cnn'].eval()

    # Define label encoder with same parameters as training
    min_offset = -30
    max_offset = 30
    all_possible_classes = np.arange(min_offset, max_offset + 1)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_possible_classes + 30)

    # Evaluate only the specified models
    if 'log_reg' in models:
        y_pred_log_reg = models['log_reg'].predict(X_val)
        mae_log_reg = mean_absolute_error(y_val, y_pred_log_reg)
        medae_log_reg = median_absolute_error(y_val, y_pred_log_reg)
        print(
            f'Logistic Regression - MAE: {mae_log_reg}, MedAE: {medae_log_reg}')

    if 'mlp' in models:
        y_val_shifted = y_val + 30
        y_val_encoded = label_encoder.transform(y_val_shifted)
        y_pred_mlp = models['mlp'].predict(X_val)
        y_pred_mlp = y_pred_mlp - 30  # Convert back to original scale
        mae_mlp = mean_absolute_error(y_val, y_pred_mlp)
        medae_mlp = median_absolute_error(y_val, y_pred_mlp)
        print(f'MLP - MAE: {mae_mlp}, MedAE: {medae_mlp}')

    if 'cnn' in models:
        # Reshape data for CNN (add channel dimension)
        X_val_cnn = X_padded.reshape(-1, 1, 256, 256)
        X_val_tensor = torch.tensor(X_val_cnn, dtype=torch.float32).to(device)

        # Evaluate in batches to prevent memory issues
        batch_size = 32
        y_pred_cnn = []

        with torch.no_grad():
            for i in range(0, len(X_val_tensor), batch_size):
                batch = X_val_tensor[i:i+batch_size]
                outputs = models['cnn'](batch)
                _, predicted = torch.max(outputs, 1)
                y_pred_cnn.extend(predicted.cpu().numpy())

        # Convert back to original scale
        y_pred_cnn = np.array(y_pred_cnn) - 30
        mae_cnn = mean_absolute_error(y_val, y_pred_cnn)
        medae_cnn = median_absolute_error(y_val, y_pred_cnn)
        print(f'CNN - MAE: {mae_cnn}, MedAE: {medae_cnn}')

    # Calculate baseline using median approach
    baseline_offsets = []
    for i, softmaxed_sim_12 in enumerate(X):
        # print(f'Shape of softmaxed_sim_12: {softmaxed_sim_12.shape}')
        median_offset = calculate_median_offset(softmaxed_sim_12)
        baseline_offsets.append(median_offset)

    baseline_mae = mean_absolute_error(y, baseline_offsets)
    baseline_medae = median_absolute_error(y, baseline_offsets)

    print(f'Baseline (Median) - MAE: {baseline_mae}, MedAE: {baseline_medae}')
