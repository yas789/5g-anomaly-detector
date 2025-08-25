import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import pickle
import os

class Autoencoder(nn.Module):
    """
    Autoencoder neural network for 5G network anomaly detection
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, hidden_dim: int = 64):
        """
        Initialize the autoencoder
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Dimension of the encoding layer (bottleneck)
            hidden_dim (int): Dimension of hidden layers
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming normalized input data
        )
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input data"""
        return self.encoder(x)

class AnomalyDetector:
    """
    5G Network Anomaly Detector using Autoencoder
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, hidden_dim: int = 64, 
                 device: Optional[str] = None):
        """
        Initialize the anomaly detector
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Dimension of the encoding layer
            hidden_dim (int): Dimension of hidden layers
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, encoding_dim, hidden_dim).to(self.device)
        self.scaler = StandardScaler()
        self.threshold = None
        self.input_dim = input_dim
        self.feature_names = None
        
        print(f"Using device: {self.device}")
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: list, 
                    test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): List of feature column names
            test_size (float): Fraction of data to use for testing
            
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders
        """
        self.feature_names = feature_columns
        
        # Extract features and filter out anomalies for training
        normal_data = df[df['is_anomaly'] == False][feature_columns]
        
        # Normalize the data
        X_normalized = self.scaler.fit_transform(normal_data)
        
        # Split into train/validation
        X_train, X_val = train_test_split(X_normalized, test_size=test_size, random_state=42)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 100, learning_rate: float = 0.001, 
              patience: int = 10) -> dict:
        """
        Train the autoencoder model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for optimizer
            patience (int): Early stopping patience
            
        Returns:
            dict: Training history
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_data, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data, batch_targets in val_loader:
                    outputs = self.model(batch_data)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_autoencoder.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': epoch + 1
        }
    
    def calculate_threshold(self, df: pd.DataFrame, feature_columns: list, 
                          percentile: float = 95) -> float:
        """
        Calculate anomaly threshold based on reconstruction errors of normal data
        
        Args:
            df (pd.DataFrame): Training dataframe
            feature_columns (list): List of feature column names
            percentile (float): Percentile for threshold calculation
            
        Returns:
            float: Anomaly threshold
        """
        normal_data = df[df['is_anomaly'] == False][feature_columns]
        X_normalized = self.scaler.transform(normal_data)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        self.threshold = np.percentile(reconstruction_errors, percentile)
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        
        return self.threshold
    
    def detect_anomalies(self, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        """
        Detect anomalies in the given data
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): List of feature column names
            
        Returns:
            np.ndarray: Array of reconstruction errors
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Please call calculate_threshold() first.")
        
        X = df[feature_columns]
        X_normalized = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        return reconstruction_errors
    
    def predict_anomalies(self, df: pd.DataFrame, feature_columns: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies and return both errors and binary predictions
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): List of feature column names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (reconstruction_errors, anomaly_predictions)
        """
        reconstruction_errors = self.detect_anomalies(df, feature_columns)
        anomaly_predictions = reconstruction_errors > self.threshold
        
        return reconstruction_errors, anomaly_predictions
    
    def evaluate_model(self, df: pd.DataFrame, feature_columns: list) -> dict:
        """
        Evaluate model performance on labeled data
        
        Args:
            df (pd.DataFrame): Test dataframe with true labels
            feature_columns (list): List of feature column names
            
        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        reconstruction_errors, predictions = self.predict_anomalies(df, feature_columns)
        true_labels = df['is_anomaly'].values
        
        metrics = {
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions),
            'auc_score': roc_auc_score(true_labels, reconstruction_errors),
            'threshold': self.threshold
        }
        
        print(f"Evaluation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the complete model and scaler"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load the complete model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.input_dim = model_data['input_dim']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from: {filepath}")

def plot_training_history(history: dict):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reconstruction_errors(errors: np.ndarray, true_labels: np.ndarray, threshold: float):
    """Plot reconstruction errors distribution"""
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Histogram of reconstruction errors
    plt.subplot(1, 2, 1)
    normal_errors = errors[true_labels == False]
    anomaly_errors = errors[true_labels == True]
    
    plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    
    # Plot 2: Time series of errors
    plt.subplot(1, 2, 2)
    plt.plot(errors, alpha=0.7, label='Reconstruction Error')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
    anomaly_indices = np.where(true_labels == True)[0]
    plt.scatter(anomaly_indices, errors[anomaly_indices], color='red', alpha=0.8, label='True Anomalies')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()