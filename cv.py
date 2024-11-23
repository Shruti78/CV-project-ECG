# -*- coding: utf-8 -*-

import wfdb
import numpy as np
import pickle
import os
from tqdm import tqdm
import requests
import logging
from scipy import signal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MITBIHDataPreparator:
    def __init__(self, output_dir='ECG_code'):
        """
        Initialize the data preparator
        Args:
            output_dir: Directory where the processed data will be saved
        """
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'raw_data')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # List of records to download
        self.records = ['100', '101', '102', '103', '104', '105', '106', '107',
                       '108', '109', '111', '112', '113', '114', '115', '116',
                       '117', '118', '119', '121', '122', '123', '124', '200',
                       '201', '202', '203', '205', '207', '208', '209', '210',
                       '212', '213', '214', '215', '217', '219', '220', '221',
                       '222', '223', '228', '230', '231', '232', '233', '234']

    def download_record(self, record):
        """Download a single record from PhysioNet"""
        base_url = "https://physionet.org/files/mitdb/1.0.0/"
        for ext in ['.dat', '.hea', '.atr']:
            file_url = f"{base_url}{record}{ext}"
            local_file = os.path.join(self.data_dir, f"{record}{ext}")

            if not os.path.exists(local_file):
                response = requests.get(file_url)
                response.raise_for_status()
                with open(local_file, 'wb') as f:
                    f.write(response.content)

    def preprocess_signal(self, sig):
        """Apply basic preprocessing to the ECG signal"""
        # Remove baseline wander (high-pass filter)
        fs = 360.0  # Sampling frequency
        nyq = fs/2
        cutoff = 0.5  # Cutoff frequency
        order = 5
        b, a = signal.butter(order, cutoff/nyq, btype='high')
        sig = signal.filtfilt(b, a, sig)

        # Normalize
        sig = (sig - np.mean(sig)) / np.std(sig)
        return sig

    def extract_beats(self, signal, r_peaks, window_size=480):
        """Extract beats centered around R-peaks"""
        half_window = window_size // 2
        beats = []

        for peak in r_peaks:
            if peak - half_window >= 0 and peak + half_window <= len(signal):
                beat = signal[peak - half_window:peak + half_window]
                beats.append(beat)

        return np.array(beats)

    def process_data(self):
        """Process all records and create the final dataset"""
        all_beats = []
        all_labels = []

        logger.info("Downloading and processing records...")
        for record in tqdm(self.records):
            try:
                # Download record if not exists
                self.download_record(record)

                # Read record
                record_path = os.path.join(self.data_dir, record)
                signals, fields = wfdb.rdsamp(record_path)
                annotations = wfdb.rdann(record_path, 'atr')

                # Get ECG signal (lead II)
                ecg_signal = signals[:, 0]  # First channel

                # Preprocess signal
                processed_signal = self.preprocess_signal(ecg_signal)

                # Extract beats
                beats = self.extract_beats(processed_signal, annotations.sample)

                # Process annotations
                for beat, label in zip(beats, annotations.symbol):
                    if label in ['N', 'L', 'R', 'A', 'V']:  # Normal, LBBB, RBBB, APC, PVC
                        beat_label = {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'V': 4}[label]
                        all_beats.append(beat)
                        all_labels.append(beat_label)

            except Exception as e:
                logger.error(f"Error processing record {record}: {str(e)}")
                continue

        # Convert to numpy arrays
        X = np.array(all_beats)
        y = np.array(all_labels).reshape(-1, 1)

        # Combine features and labels
        dataset = np.hstack((X, y))

        # Save processed data
        output_file = os.path.join(self.output_dir, 'MIT_BIH_data_5class.p')
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)

        logger.info(f"Dataset saved to {output_file}")
        logger.info(f"Dataset shape: {dataset.shape}")
        logger.info(f"Number of samples per class:")
        for i, label in enumerate(['Normal', 'LBBB', 'RBBB', 'APC', 'PVC']):
            count = np.sum(y == i)
            logger.info(f"{label}: {count}")

def main():
    try:
        preparator = MITBIHDataPreparator()
        preparator.process_data()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix  # Import confusion_matrix
from torch.nn.parameter import Parameter
import time
import pickle
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for better visualization of confusion matrix

def one_hot(y_, maxvalue=None):
    if maxvalue is None:
        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
    else:
        n_values = maxvalue
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

class ECGDataset:
    def _init_(self, data_path):
        # Load and preprocess data
        self.data = pickle.load(open(data_path, "rb"), encoding='latin1')
        self.data = self.data[:, :-1]  # Remove last column if it's unnecessary
        self.n_features = self.data.shape[-1] - 1

        # Normalize features
        feature_all = self.data[:, 0:self.n_features]
        label_all = self.data[:, self.n_features:self.n_features+1]

        # Clean and validate labels
        valid_mask = (label_all >= 0) & (label_all < 5)  # Assuming 5 classes (0-4)
        valid_mask = valid_mask.reshape(-1)

        # Filter out invalid data
        feature_all = feature_all[valid_mask]
        label_all = label_all[valid_mask]

        print(f"Original samples: {len(valid_mask)}")
        print(f"Valid samples: {np.sum(valid_mask)}")
        print(f"Removed {len(valid_mask) - np.sum(valid_mask)} invalid samples")

        # Check unique labels
        unique_labels = np.unique(label_all)
        print(f"Unique labels in dataset: {unique_labels}")

        self.feature_normalized = preprocessing.scale(feature_all)
        self.all_data = np.hstack((self.feature_normalized, label_all))

        # Shuffle data
        np.random.shuffle(self.all_data)

    def prepare_data(self, device, train_ratio=0.8):
        n_samples = self.all_data.shape[0]
        self.all_data = torch.Tensor(self.all_data).to(device)

        # Split train and test
        train_size = int(train_ratio * n_samples)
        train_data = self.all_data[:train_size]
        test_data = self.all_data[train_size:]

        # Prepare features for CNN (pad if necessary)
        def prepare_features(data):
            features = data[:, :-1]
            labels = data[:, -1].long()

            # Verify label range
            assert torch.all(labels >= 0) and torch.all(labels < 5), "Invalid labels found in dataset"

            # Calculate padding needed
            total_features = features.shape[1]
            target_size = 480  # 2 * 240
            if total_features < target_size:
                padding = torch.zeros(features.shape[0], target_size - total_features).to(device)
                features = torch.cat([features, padding], dim=1)
            else:
                features = features[:, :target_size]

            # Reshape to [batch, channels, length]
            features = features.reshape(-1, 2, 240)
            # Add channel dimension for CNN
            features = features.unsqueeze(1)

            return features, labels

        train_x, train_y = prepare_features(train_data)
        test_x, test_y = prepare_features(test_data)

        # Print label distribution
        for dataset_name, labels in [("Training", train_y), ("Test", test_y)]:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            print(f"\n{dataset_name} set label distribution:")
            for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
                print(f"Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")

        return train_x, train_y, test_x, test_y

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ImprovedCNN, self).__init__()

        # First convolution block - only pool in time dimension
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))  # Only pool in time dimension
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))  # Only pool in time dimension
        )

        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(2, 3), padding=(0, 1)),  # Reduce spatial dimension
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))  # Only pool in time dimension
        )

        # Calculate the flattened size
        self._to_linear = None
        x = torch.zeros(1, 1, 2, 240)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self._to_linear = int(np.prod(x.shape[1:]))

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        # Learnable weight for signal attention
        self.weight = Parameter(torch.ones(1, 240))

    def forward(self, x):
        # For debugging
        batch_size = x.size(0)

        # Apply attention weight
        x = x.view(batch_size, 1, 2, 240)
        x = x * self.weight.unsqueeze(0).unsqueeze(0)

        # Debug prints
        # print(f"Input shape: {x.shape}")

        # Convolutional layers
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")

        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")

        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")

        # Flatten
        x = x.view(batch_size, -1)
        # print(f"After flatten: {x.shape}")

        # Fully connected layers
        features = F.relu(self.fc1(x))
        x = self.dropout(features)
        output = self.fc2(x)

        return output, features

def train_model(model, train_loader, test_x, test_y, epochs, device, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    loss_func = nn.CrossEntropyLoss()

    best_auc = 0
    train_losses = []  # To store training loss
    test_accuracies = []  # To store test accuracy

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0  # Initialize loss for the epoch

        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = model(batch_x)[0]
            loss = loss_func(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate the loss

            if step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    test_output = model(test_x)[0]
                    test_loss = loss_func(test_output, test_y)

                    test_y_score = one_hot(test_y.cpu().numpy())
                    pred_score = F.softmax(test_output, dim=1).cpu().numpy()
                    auc_score = roc_auc_score(test_y_score, pred_score, multi_class='ovr')

                    pred_y = torch.max(test_output, 1)[1].cpu().numpy()
                    accuracy = (pred_y == test_y.cpu().numpy()).mean()

                    print(f'Epoch: {epoch} | Step: {step} | Test Loss: {test_loss:.4f} | '
                          f'Accuracy: {accuracy:.4f} | AUC: {auc_score:.4f}')

                    if auc_score > best_auc:
                        best_auc = auc_score
                        torch.save(model.state_dict(), 'best_model.pt')
                        print(f'Model saved with AUC: {best_auc:.4f}')

                model.train()

        # Calculate average loss for the epoch and record accuracy
        train_losses.append(epoch_loss / len(train_loader))
        test_accuracies.append(accuracy)

        # Adjust learning rate
        scheduler.step(auc_score)

    # Plotting the results
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Test Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()  # Display the plots

    # Compute confusion matrix after training
    model.eval()
    with torch.no_grad():
        test_output = model(test_x)[0]
        pred_y = torch.max(test_output, 1)[1].cpu().numpy()  # Get predicted labels
        test_y_np = test_y.cpu().numpy()  # Get true labels

        # Generate confusion matrix
        cm = confusion_matrix(test_y_np, pred_y)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=np.arange(5), yticklabels=np.arange(5))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()  # Display the confusion matrix

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load and prepare data
    dataset = ECGDataset('/content/ECG_code/MIT_BIH_data_5class.p')
    train_x, train_y, test_x, test_y = dataset.prepare_data(device)

    print(f"Training data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")

    # Create data loader
    batch_size = int(len(train_x) * 0.05)  # 5% of training data
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize and train model
    model = ImprovedCNN().to(device)

    # Test forward pass
    with torch.no_grad():
        dummy_input = train_x[:2]
        dummy_output = model(dummy_input)
        print("Forward pass successful!")

    train_model(model, train_loader, test_x, test_y, epochs=10, device=device)

    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        test_output = model(test_x)[0]
        pred_score = F.softmax(test_output, dim=1).cpu().numpy()
        pred_y = np.argmax(pred_score, axis=1)

        test_y_np = test_y.cpu().numpy()
        test_y_score = one_hot(test_y_np)
        auc_score = roc_auc_score(test_y_score, pred_score, multi_class='ovr')

        print('Final Results:')
        print(classification_report(test_y_np, pred_y, digits=4))
        print(f"Final AUC: {auc_score:.4f}")

if __name__ == "__main__":
    main()