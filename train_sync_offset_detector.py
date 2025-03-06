import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, median_absolute_error
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from evaluation.utils import calculate_margin_of_error

# Function to calculate the baseline sync offset using the median approach


def calculate_median_offset(softmaxed_sim_12):
    predict = softmaxed_sim_12.argmax(axis=1)
    ground = np.arange(softmaxed_sim_12.shape[0])

    frames = predict - ground
    median_frames = np.floor(np.median(frames))
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


# ----------------------------
# 1) Dataset Class
# ----------------------------
class SimilarityMatrixDataset(Dataset):
    """
    A PyTorch Dataset for similarity matrices of shape (256, 256).
    Expects data of shape:
        X: (num_samples, 1, 256, 256)  -- single-channel "image"
        y: (num_samples,)              -- integer labels in [0..(num_classes-1)]
    """

    def __init__(self, X, y):
        """
        X: 4D NumPy array, shape [num_samples, 1, 256, 256]
        y: 1D NumPy array, integer class labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # shape: (1, 256, 256)
        x_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor


# ----------------------------
# 2) Model Definition
# ----------------------------
class SimilarityMatrixClassifier(nn.Module):
    """
    A CNN classifier for similarity matrices, outputting logits
    for 'num_classes' distinct categories.
    """

    def __init__(self, num_classes=61):
        """
        Example: if labels are -30..30 => 61 classes
        """
        super(SimilarityMatrixClassifier, self).__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output: (32, 128, 128)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output: (64, 64, 64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # output: (128, 32, 32)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)   # output: (256, 16, 16)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(512, num_classes)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)                   # (batch_size, 256, 16, 16)
        # flatten => (batch_size, 256*16*16)
        x = x.view(x.size(0), -1)
        # => (batch_size, num_classes) logits
        x = self.classifier(x)
        return x


# ----------------------------
# 3) Training Function
# ----------------------------
def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    """
    Standard PyTorch training loop for multi-class classification.

    model:        PyTorch model (SimilarityMatrixClassifier)
    train_loader: DataLoader for training set
    val_loader:   DataLoader for validation set
    num_epochs:   how many epochs
    lr:           initial learning rate
    device:       'cuda' or 'cpu'
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # Forward
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += data.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for val_data, val_targets in val_loader:
                val_data, val_targets = val_data.to(
                    device), val_targets.to(device)
                val_outputs = model(val_data)
                loss = criterion(val_outputs, val_targets)

                val_loss += loss.item() * val_data.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                correct_val += (val_predicted == val_targets).sum().item()
                total_val += val_data.size(0)

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train sync offset detector')
    parser.add_argument('--prefix', type=str, required=True,
                        help='Prefix for input/output files (e.g., "ntu")')
    parser.add_argument('--sync_methods', nargs='+', choices=['log_reg', 'svm', 'mlp', 'cnn'],
                        required=True, help='Sync methods to train (log_reg, svm, mlp, cnn)')
    args = parser.parse_args()

    prefix = args.prefix

    # Load prepared data
    X_train = np.load(
        f'{prefix}_train_softmaxed_sim_12.npy', allow_pickle=True)
    y_train = np.load(
        f'{prefix}_train_softmaxed_sim_12_labels.npy', allow_pickle=True)
    X_val = np.load(f'{prefix}_val_softmaxed_sim_12.npy', allow_pickle=True)
    y_val = np.load(
        f'{prefix}_val_softmaxed_sim_12_labels.npy', allow_pickle=True)

    print(
        f'Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}')
    print(f'Shape of X_val: {X_val.shape}, Shape of y_val: {y_val.shape}')

    # Pad matrices to target_size*target_size
    X_train_padded = pad_matrices(X_train, target_size=256)
    X_val_padded = pad_matrices(X_val, target_size=256)

    print('after padding X matrices')
    print(
        f'Shape of X_train_padded: {X_train_padded.shape}, Shape of X_val_padded: {X_val_padded.shape}')
    print(f'Shape of y_train: {y_train.shape}, Shape of y_val: {y_val.shape}')

    # Define the range of classes explicitly
    min_offset = -30
    max_offset = 30
    n_classes = max_offset - min_offset + 1  # 61 classes
    all_possible_classes = np.arange(min_offset, max_offset + 1)

    # Initialize label encoder with all possible classes
    label_encoder = LabelEncoder()
    # Fit with shifted values (0 to 60)
    label_encoder.fit(all_possible_classes + 30)

    # Convert y values to class labels (0 to 60)
    y_train_shifted = y_train + 30
    y_val_shifted = y_val + 30
    y_train_encoded = label_encoder.transform(y_train_shifted)
    y_val_encoded = label_encoder.transform(y_val_shifted)

    config = {
        'train_log_reg': 'log_reg' in args.sync_methods,
        'train_svm': 'svm' in args.sync_methods,
        'train_mlp': 'mlp' in args.sync_methods,
        'train_cnn': 'cnn' in args.sync_methods,
        'mlp_config': {
            'hidden_layer_sizes': (2048, 1024, 512),
            'max_iter': 1000,
            'activation': 'relu',
            'solver': 'adam',
            'random_state': 42,
            'learning_rate_init': 0.001,
            'batch_size': 'auto',
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 20,
            'tol': 1e-4,
        },
        'cnn_config': {
            'num_epochs': 150,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_classes': 61,  # For range -30 to 30
        }
    }

    # Print configuration and prefix for logging
    print("\nRunning with configuration:")
    print(f"Prefix: {prefix}")
    print(json.dumps(config, indent=4))
    print("\n")

    # Prepare data for CNN if needed
    if config['train_cnn']:
        # Reshape data for CNN (add channel dimension)
        # Assuming 256x256 from your padding
        X_train_cnn = X_train_padded.reshape(-1, 1, 256, 256)
        X_val_cnn = X_val_padded.reshape(-1, 1, 256, 256)

        # Create datasets and dataloaders
        train_dataset = SimilarityMatrixDataset(X_train_cnn, y_train_encoded)
        val_dataset = SimilarityMatrixDataset(X_val_cnn, y_val_encoded)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['cnn_config']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['cnn_config']['batch_size'],
            shuffle=False
        )

    # Initialize models based on config
    models = {}
    if config['train_log_reg']:
        models['log_reg'] = LogisticRegression(
            n_jobs=-1, verbose=True, max_iter=1000)

    if config['train_svm']:
        models['svm'] = SVR()

    if config['train_mlp']:
        models['mlp'] = MLPClassifier(verbose=True, **config['mlp_config'])

    if config['train_cnn']:
        models['cnn'] = SimilarityMatrixClassifier(
            num_classes=config['cnn_config']['num_classes']
        )

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")

        if name == 'cnn':
            # Use PyTorch training loop
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['cnn_config']['num_epochs'],
                lr=config['cnn_config']['learning_rate'],
                device=device
            )

            # Evaluate CNN model
            model.eval()
            y_pred = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    y_pred.extend(predicted.cpu().numpy())

            y_pred = np.array(y_pred) - 30  # Convert back to original scale

        elif name == 'mlp':
            # Existing MLP training code
            model.fit(X_train_padded, y_train_encoded)
            y_pred = model.predict(X_val_padded)
            y_pred = y_pred - 30
        else:
            # Existing training code for other models
            model.fit(X_train_padded, y_train)
            y_pred = model.predict(X_val_padded)

        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        medae = median_absolute_error(y_val, y_pred)

        # Calculate absolute errors for margin calculation
        absolute_errors = np.abs(y_val - y_pred)
        margin = calculate_margin_of_error(absolute_errors)

        results[name] = {
            'mae': mae,
            'medae': medae,
            'mae_margin': margin
        }

        print(f'{name.upper()} - MAE: {mae:.4f} ± {margin:.4f}, MedAE: {medae:.4f}')

        # Save model
        if name == 'cnn':
            torch.save(model.state_dict(), f'{prefix}_{name}_model.pt')
        else:
            with open(f'{prefix}_{name}_model.pkl', 'wb') as file:
                pickle.dump(model, file)

    # For baseline results
    baseline_offsets = []
    for i, softmaxed_sim_12 in enumerate(X_val):
        median_offset = calculate_median_offset(softmaxed_sim_12)
        baseline_offsets.append(median_offset)

    baseline_mae = mean_absolute_error(y_val, baseline_offsets)
    baseline_medae = median_absolute_error(y_val, baseline_offsets)
    baseline_absolute_errors = np.abs(y_val - baseline_offsets)
    baseline_margin = calculate_margin_of_error(baseline_absolute_errors)

    print(
        f'Baseline (Median) - MAE: {baseline_mae:.4f} ± {baseline_margin:.4f}, MedAE: {baseline_medae:.4f}')
