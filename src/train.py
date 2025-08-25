#!/usr/bin/env python3
"""
Training script for 5G Network Anomaly Detection using Autoencoder
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector, plot_training_history, plot_reconstruction_errors
from data.generate_synthetic_data import generate_synthetic_5g_data

def load_or_generate_data(data_path: str, regenerate: bool = False, num_samples: int = 10000):
    """
    Load existing data or generate new synthetic data
    
    Args:
        data_path (str): Path to the CSV data file
        regenerate (bool): Force regeneration of data
        num_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Loaded or generated data
    """
    if os.path.exists(data_path) and not regenerate:
        print(f"Loading existing data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")
    else:
        print(f"Generating new synthetic data...")
        df = generate_synthetic_5g_data(num_samples=num_samples, anomaly_rate=0.1)
        df.to_csv(data_path, index=False)
        print(f"Generated and saved {len(df)} samples to: {data_path}")
    
    return df

def prepare_feature_columns():
    """
    Define the feature columns for training
    
    Returns:
        list: List of feature column names
    """
    return [
        'prb_utilization',
        'active_ue_count', 
        'throughput_mbps',
        'latency_ms',
        'handover_success_rate',
        'snr_db',
        'packet_loss_rate'
    ]

def train_model(df: pd.DataFrame, feature_columns: list, config: dict):
    """
    Train the autoencoder model
    
    Args:
        df (pd.DataFrame): Training data
        feature_columns (list): List of feature columns
        config (dict): Training configuration
        
    Returns:
        AnomalyDetector: Trained model
        dict: Training history
    """
    print("Initializing anomaly detector...")
    
    # Initialize the detector
    detector = AnomalyDetector(
        input_dim=len(feature_columns),
        encoding_dim=config['encoding_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    print(f"Model architecture:")
    print(f"- Input dimension: {len(feature_columns)}")
    print(f"- Hidden dimension: {config['hidden_dim']}")
    print(f"- Encoding dimension: {config['encoding_dim']}")
    
    # Prepare data
    print("Preparing training data...")
    train_loader, val_loader = detector.prepare_data(
        df, 
        feature_columns, 
        test_size=config['test_size']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Train the model
    print("Starting training...")
    history = detector.train(
        train_loader,
        val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience']
    )
    
    print(f"Training completed after {history['epochs_trained']} epochs")
    
    return detector, history

def evaluate_model(detector: AnomalyDetector, df: pd.DataFrame, feature_columns: list):
    """
    Evaluate the trained model
    
    Args:
        detector (AnomalyDetector): Trained model
        df (pd.DataFrame): Evaluation data
        feature_columns (list): List of feature columns
        
    Returns:
        dict: Evaluation metrics
        np.ndarray: Reconstruction errors
    """
    print("Calculating anomaly threshold...")
    threshold = detector.calculate_threshold(df, feature_columns, percentile=95)
    
    print("Evaluating model performance...")
    metrics = detector.evaluate_model(df, feature_columns)
    
    # Get reconstruction errors for visualization
    reconstruction_errors = detector.detect_anomalies(df, feature_columns)
    
    return metrics, reconstruction_errors

def save_training_artifacts(detector: AnomalyDetector, history: dict, metrics: dict, 
                          model_dir: str, timestamp: str):
    """
    Save all training artifacts
    
    Args:
        detector (AnomalyDetector): Trained model
        history (dict): Training history
        metrics (dict): Evaluation metrics
        model_dir (str): Directory to save artifacts
        timestamp (str): Timestamp for file naming
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, f'anomaly_detector_{timestamp}.pkl')
    detector.save_model(model_path)
    
    # Save training history
    history_path = os.path.join(model_dir, f'training_history_{timestamp}.txt')
    with open(history_path, 'w') as f:
        f.write(f"Training History - {timestamp}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Epochs trained: {history['epochs_trained']}\n")
        f.write(f"Final training loss: {history['train_losses'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history['val_losses'][-1]:.6f}\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write("-"*20 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"Training artifacts saved to: {model_dir}")

def create_visualizations(history: dict, errors: np.ndarray, true_labels: np.ndarray, 
                         threshold: float, output_dir: str, timestamp: str):
    """
    Create and save visualization plots
    
    Args:
        history (dict): Training history
        errors (np.ndarray): Reconstruction errors
        true_labels (np.ndarray): True anomaly labels
        threshold (float): Anomaly threshold
        output_dir (str): Output directory for plots
        timestamp (str): Timestamp for file naming
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'training_history_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot reconstruction errors
    plot_reconstruction_errors(errors, true_labels, threshold)
    plt.savefig(os.path.join(output_dir, f'reconstruction_errors_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train 5G Network Anomaly Detection Model')
    parser.add_argument('--data-path', type=str, default='data/synthetic_5g_metrics.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--regenerate-data', action='store_true',
                        help='Force regeneration of synthetic data')
    parser.add_argument('--num-samples', type=int, default=10000,
                        help='Number of samples to generate (if regenerating)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--encoding-dim', type=int, default=32,
                        help='Encoding layer dimension')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for validation')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for saved models')
    parser.add_argument('--plot-dir', type=str, default='plots',
                        help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("5G NETWORK ANOMALY DETECTION - TRAINING")
    print("="*60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run timestamp: {timestamp}")
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'encoding_dim': args.encoding_dim,
        'test_size': args.test_size,
        'patience': args.patience
    }
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"- {key}: {value}")
    
    try:
        # Load or generate data
        df = load_or_generate_data(args.data_path, args.regenerate_data, args.num_samples)
        
        # Prepare feature columns
        feature_columns = prepare_feature_columns()
        print(f"\nFeature columns: {feature_columns}")
        
        # Check data quality
        print(f"\nData Summary:")
        print(f"- Total samples: {len(df)}")
        print(f"- Normal samples: {len(df[df['is_anomaly'] == False])}")
        print(f"- Anomalous samples: {len(df[df['is_anomaly'] == True])}")
        print(f"- Anomaly rate: {df['is_anomaly'].mean():.2%}")
        
        # Train the model
        detector, history = train_model(df, feature_columns, config)
        
        # Evaluate the model
        metrics, reconstruction_errors = evaluate_model(detector, df, feature_columns)
        
        # Save training artifacts
        save_training_artifacts(detector, history, metrics, args.output_dir, timestamp)
        
        # Create visualizations
        if not args.no_plots:
            create_visualizations(
                history, 
                reconstruction_errors, 
                df['is_anomaly'].values,
                detector.threshold,
                args.plot_dir,
                timestamp
            )
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"✓ Model trained successfully in {history['epochs_trained']} epochs")
        print(f"✓ Final validation loss: {history['val_losses'][-1]:.6f}")
        print(f"✓ Anomaly threshold: {detector.threshold:.6f}")
        print(f"✓ F1-Score: {metrics['f1_score']:.4f}")
        print(f"✓ AUC Score: {metrics['auc_score']:.4f}")
        print(f"✓ Model saved to: {args.output_dir}")
        
        if not args.no_plots:
            print(f"✓ Plots saved to: {args.plot_dir}")
        
        print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()