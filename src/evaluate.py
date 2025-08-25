#!/usr/bin/env python3
"""
Evaluation script for 5G Network Anomaly Detection Model
Calculates comprehensive performance metrics and creates visualizations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from data.generate_synthetic_data import generate_synthetic_5g_data

class ModelEvaluator:
    """
    Comprehensive evaluation class for anomaly detection models
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the evaluator with a trained model
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.detector = None
        self.feature_columns = [
            'prb_utilization',
            'active_ue_count', 
            'throughput_mbps',
            'latency_ms',
            'handover_success_rate',
            'snr_db',
            'packet_loss_rate'
        ]
        self.results = {}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained anomaly detection model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.detector = AnomalyDetector(input_dim=len(self.feature_columns))
        self.detector.load_model(self.model_path)
        print("✅ Model loaded successfully")
    
    def evaluate_model(self, test_data: pd.DataFrame, save_plots: bool = True, 
                      output_dir: str = "evaluation_results") -> dict:
        """
        Comprehensive model evaluation with multiple metrics
        
        Args:
            test_data (pd.DataFrame): Test dataset with ground truth labels
            save_plots (bool): Whether to save visualization plots
            output_dir (str): Directory to save results
            
        Returns:
            dict: Complete evaluation results
        """
        print("🔍 Starting comprehensive model evaluation...")
        
        # Create output directory
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get predictions
        reconstruction_errors = self.detector.detect_anomalies(test_data, self.feature_columns)
        predictions = reconstruction_errors > self.detector.threshold
        true_labels = test_data['is_anomaly'].values
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(true_labels, predictions, reconstruction_errors)
        
        # Create visualizations
        if save_plots:
            self._create_visualizations(
                true_labels, predictions, reconstruction_errors, 
                output_dir, test_data
            )
        
        # Detailed analysis
        detailed_analysis = self._detailed_analysis(
            true_labels, predictions, reconstruction_errors, test_data
        )
        
        # Combine results
        self.results = {
            'metrics': metrics,
            'detailed_analysis': detailed_analysis,
            'model_info': {
                'model_path': self.model_path,
                'threshold': float(self.detector.threshold),
                'feature_count': len(self.feature_columns),
                'test_samples': len(test_data)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("✅ Evaluation completed successfully")
        return self.results
    
    def _calculate_metrics(self, true_labels: np.ndarray, predictions: np.ndarray, 
                         reconstruction_errors: np.ndarray) -> dict:
        """Calculate comprehensive performance metrics"""
        print("📊 Calculating performance metrics...")
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve and Average Precision
        pr_precision, pr_recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
        avg_precision = average_precision_score(true_labels, reconstruction_errors)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom != 0 else 0
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'npv': float(npv),
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'mcc': float(mcc),
            'false_positive_rate': float(fpr_rate),
            'false_negative_rate': float(fnr_rate),
            'confusion_matrix': {
                'true_positive': int(tp),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_negative': int(tn)
            },
            'reconstruction_error_stats': {
                'mean': float(reconstruction_errors.mean()),
                'std': float(reconstruction_errors.std()),
                'min': float(reconstruction_errors.min()),
                'max': float(reconstruction_errors.max()),
                'median': float(np.median(reconstruction_errors))
            }
        }
        
        return metrics
    
    def _detailed_analysis(self, true_labels: np.ndarray, predictions: np.ndarray,
                          reconstruction_errors: np.ndarray, test_data: pd.DataFrame) -> dict:
        """Perform detailed analysis of model performance"""
        print("🔬 Performing detailed analysis...")
        
        # Anomaly distribution analysis
        true_anomaly_indices = np.where(true_labels == True)[0]
        pred_anomaly_indices = np.where(predictions == True)[0]
        
        # Error analysis by feature
        feature_analysis = {}
        if len(true_anomaly_indices) > 0:
            true_anomaly_data = test_data.iloc[true_anomaly_indices]
            normal_data = test_data[test_data['is_anomaly'] == False]
            
            for feature in self.feature_columns:
                anomaly_values = true_anomaly_data[feature]
                normal_values = normal_data[feature]
                
                feature_analysis[feature] = {
                    'anomaly_mean': float(anomaly_values.mean()),
                    'anomaly_std': float(anomaly_values.std()),
                    'normal_mean': float(normal_values.mean()),
                    'normal_std': float(normal_values.std()),
                    'separation_score': abs(anomaly_values.mean() - normal_values.mean()) / 
                                      (anomaly_values.std() + normal_values.std() + 1e-8)
                }
        
        # Reconstruction error distribution analysis
        normal_errors = reconstruction_errors[true_labels == False]
        anomaly_errors = reconstruction_errors[true_labels == True]
        
        error_analysis = {
            'normal_errors': {
                'mean': float(normal_errors.mean()),
                'std': float(normal_errors.std()),
                'percentiles': {
                    '25': float(np.percentile(normal_errors, 25)),
                    '50': float(np.percentile(normal_errors, 50)),
                    '75': float(np.percentile(normal_errors, 75)),
                    '95': float(np.percentile(normal_errors, 95)),
                    '99': float(np.percentile(normal_errors, 99))
                }
            },
            'anomaly_errors': {
                'mean': float(anomaly_errors.mean()) if len(anomaly_errors) > 0 else 0,
                'std': float(anomaly_errors.std()) if len(anomaly_errors) > 0 else 0,
                'percentiles': {
                    '25': float(np.percentile(anomaly_errors, 25)) if len(anomaly_errors) > 0 else 0,
                    '50': float(np.percentile(anomaly_errors, 50)) if len(anomaly_errors) > 0 else 0,
                    '75': float(np.percentile(anomaly_errors, 75)) if len(anomaly_errors) > 0 else 0,
                    '95': float(np.percentile(anomaly_errors, 95)) if len(anomaly_errors) > 0 else 0,
                    '99': float(np.percentile(anomaly_errors, 99)) if len(anomaly_errors) > 0 else 0
                }
            }
        }
        
        # Threshold analysis
        threshold_analysis = self._analyze_thresholds(true_labels, reconstruction_errors)
        
        return {
            'feature_analysis': feature_analysis,
            'error_analysis': error_analysis,
            'threshold_analysis': threshold_analysis,
            'sample_distribution': {
                'total_samples': len(test_data),
                'true_anomalies': int(true_labels.sum()),
                'predicted_anomalies': int(predictions.sum()),
                'anomaly_rate_true': float(true_labels.mean()),
                'anomaly_rate_predicted': float(predictions.mean())
            }
        }
    
    def _analyze_thresholds(self, true_labels: np.ndarray, 
                           reconstruction_errors: np.ndarray) -> dict:
        """Analyze performance at different threshold values"""
        thresholds = np.percentile(reconstruction_errors, [90, 95, 97, 98, 99, 99.5])
        threshold_results = {}
        
        for percentile, threshold in zip([90, 95, 97, 98, 99, 99.5], thresholds):
            predictions = reconstruction_errors > threshold
            
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            threshold_results[f'p{percentile}'] = {
                'threshold': float(threshold),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'predicted_anomalies': int(predictions.sum())
            }
        
        return threshold_results
    
    def _create_visualizations(self, true_labels: np.ndarray, predictions: np.ndarray,
                             reconstruction_errors: np.ndarray, output_dir: str,
                             test_data: pd.DataFrame):
        """Create comprehensive visualization plots"""
        print("📈 Creating visualization plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(true_labels, predictions, output_dir)
        
        # 2. ROC Curve
        self._plot_roc_curve(true_labels, reconstruction_errors, output_dir)
        
        # 3. Precision-Recall Curve
        self._plot_precision_recall_curve(true_labels, reconstruction_errors, output_dir)
        
        # 4. Reconstruction Error Distribution
        self._plot_error_distribution(true_labels, reconstruction_errors, output_dir)
        
        # 5. Feature Analysis
        self._plot_feature_analysis(test_data, output_dir)
        
        # 6. Threshold Analysis
        self._plot_threshold_analysis(true_labels, reconstruction_errors, output_dir)
        
        # 7. Performance Summary
        self._plot_performance_summary(output_dir)
    
    def _plot_confusion_matrix(self, true_labels: np.ndarray, predictions: np.ndarray,
                             output_dir: str):
        """Plot confusion matrix with detailed annotations"""
        plt.figure(figsize=(10, 8))
        
        cm = confusion_matrix(true_labels, predictions)
        
        # Create heatmap with annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   square=True, linewidths=0.5)
        
        plt.title('Confusion Matrix for Anomaly Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=14)
        plt.ylabel('True Labels', fontsize=14)
        plt.xticks([0.5, 1.5], ['Normal', 'Anomaly'])
        plt.yticks([0.5, 1.5], ['Normal', 'Anomaly'])
        
        # Add performance metrics as text
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Precision: {precision:.3f}\\nRecall: {recall:.3f}\\nF1-Score: {f1:.3f}'
        plt.text(2.2, 1, metrics_text, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                              facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, true_labels: np.ndarray, reconstruction_errors: np.ndarray,
                       output_dir: str):
        """Plot ROC curve with AUC score"""
        plt.figure(figsize=(10, 8))
        
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8,
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, true_labels: np.ndarray, 
                                   reconstruction_errors: np.ndarray, output_dir: str):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(10, 8))
        
        precision, recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
        avg_precision = average_precision_score(true_labels, reconstruction_errors)
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = true_labels.mean()
        plt.axhline(y=baseline, color='red', linestyle='--', alpha=0.8,
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, true_labels: np.ndarray, 
                               reconstruction_errors: np.ndarray, output_dir: str):
        """Plot reconstruction error distribution for normal vs anomaly samples"""
        plt.figure(figsize=(12, 8))
        
        normal_errors = reconstruction_errors[true_labels == False]
        anomaly_errors = reconstruction_errors[true_labels == True]
        
        # Plot histograms
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True, 
                color='blue', edgecolor='black')
        if len(anomaly_errors) > 0:
            plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', density=True,
                    color='red', edgecolor='black')
        
        # Add threshold line
        plt.axvline(self.detector.threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold = {self.detector.threshold:.6f}')
        
        plt.xlabel('Reconstruction Error', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Distribution of Reconstruction Errors', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_analysis(self, test_data: pd.DataFrame, output_dir: str):
        """Plot feature analysis showing normal vs anomalous patterns"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.ravel()
        
        normal_data = test_data[test_data['is_anomaly'] == False]
        anomaly_data = test_data[test_data['is_anomaly'] == True]
        
        for i, feature in enumerate(self.feature_columns):
            ax = axes[i]
            
            # Plot distributions
            ax.hist(normal_data[feature], bins=30, alpha=0.7, label='Normal', 
                   density=True, color='blue')
            if len(anomaly_data) > 0:
                ax.hist(anomaly_data[feature], bins=30, alpha=0.7, label='Anomaly',
                       density=True, color='red')
            
            ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(len(self.feature_columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Feature Distribution Analysis: Normal vs Anomalous Samples', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_threshold_analysis(self, true_labels: np.ndarray, 
                               reconstruction_errors: np.ndarray, output_dir: str):
        """Plot performance metrics vs different threshold values"""
        thresholds = np.percentile(reconstruction_errors, np.linspace(80, 99.9, 50))
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            predictions = reconstruction_errors > threshold
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
        
        # Mark current threshold
        current_idx = np.argmin(np.abs(thresholds - self.detector.threshold))
        plt.axvline(self.detector.threshold, color='black', linestyle='--', alpha=0.8,
                   label=f'Current Threshold = {self.detector.threshold:.6f}')
        
        plt.xlabel('Threshold Value', fontsize=14)
        plt.ylabel('Metric Score', fontsize=14)
        plt.title('Performance Metrics vs Threshold Values', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self, output_dir: str):
        """Create a summary plot of key performance metrics"""
        if not self.results or 'metrics' not in self.results:
            return
        
        metrics = self.results['metrics']
        
        # Key metrics to display
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'ROC-AUC', 'Specificity']
        metric_values = [
            metrics['precision'], metrics['recall'], metrics['f1_score'],
            metrics['accuracy'], metrics['roc_auc'], metrics['specificity']
        ]
        
        # Color code based on performance
        colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in metric_values]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Score', fontsize=14)
        plt.title('Model Performance Summary', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal lines for reference
        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.8)')
        plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.6)')
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, output_dir: str, filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file"""
        if not self.results:
            print("No results to save. Run evaluation first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to: {filepath}")
    
    def print_summary(self):
        """Print a formatted summary of evaluation results"""
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        metrics = self.results['metrics']
        
        print("\\n" + "="*60)
        print("📊 MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\\n🎯 Classification Metrics:")
        print(f"   Accuracy:     {metrics['accuracy']:.4f}")
        print(f"   Precision:    {metrics['precision']:.4f}")
        print(f"   Recall:       {metrics['recall']:.4f}")
        print(f"   F1-Score:     {metrics['f1_score']:.4f}")
        print(f"   Specificity:  {metrics['specificity']:.4f}")
        
        print(f"\\n📈 Advanced Metrics:")
        print(f"   ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {metrics['average_precision']:.4f}")
        print(f"   MCC:          {metrics['mcc']:.4f}")
        
        print(f"\\n🎭 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   True Positives:  {cm['true_positive']}")
        print(f"   False Positives: {cm['false_positive']}")
        print(f"   False Negatives: {cm['false_negative']}")
        print(f"   True Negatives:  {cm['true_negative']}")
        
        print(f"\\n🔧 Model Configuration:")
        model_info = self.results['model_info']
        print(f"   Threshold:    {model_info['threshold']:.6f}")
        print(f"   Features:     {model_info['feature_count']}")
        print(f"   Test Samples: {model_info['test_samples']}")
        
        print("\\n" + "="*60)

def find_latest_model(models_dir: str = "models") -> str:
    """Find the latest trained model"""
    import glob
    model_pattern = os.path.join(models_dir, 'anomaly_detector_*.pkl')
    model_files = glob.glob(model_pattern)
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {models_dir}")
    return max(model_files, key=os.path.getmtime)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate 5G Network Anomaly Detection Model')
    parser.add_argument('--model-path', type=str,
                        help='Path to trained model (if not provided, finds latest)')
    parser.add_argument('--test-data', type=str,
                        help='Path to test data CSV file')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic test data')
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of samples for synthetic data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results and plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 5G NETWORK ANOMALY DETECTION - MODEL EVALUATION")
    print("="*60)
    
    try:
        # Find or use specified model
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = find_latest_model()
        
        print(f"Using model: {model_path}")
        
        # Load or generate test data
        if args.test_data:
            print(f"Loading test data from: {args.test_data}")
            test_data = pd.read_csv(args.test_data)
        elif args.generate_data:
            print(f"Generating synthetic test data ({args.num_samples} samples)...")
            test_data = generate_synthetic_5g_data(
                num_samples=args.num_samples, 
                anomaly_rate=0.15
            )
        else:
            # Try to load default data file
            default_path = "data/synthetic_5g_metrics.csv"
            if os.path.exists(default_path):
                print(f"Loading default test data from: {default_path}")
                test_data = pd.read_csv(default_path)
            else:
                print("No test data found. Generating synthetic data...")
                test_data = generate_synthetic_5g_data(
                    num_samples=args.num_samples, 
                    anomaly_rate=0.15
                )
        
        print(f"Test data: {len(test_data)} samples, "
              f"{test_data['is_anomaly'].sum()} anomalies "
              f"({test_data['is_anomaly'].mean():.2%})")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)
        
        # Run evaluation
        results = evaluator.evaluate_model(
            test_data, 
            save_plots=not args.no_plots,
            output_dir=args.output_dir
        )
        
        # Print summary
        evaluator.print_summary()
        
        # Save results
        evaluator.save_results(args.output_dir)
        
        if not args.no_plots:
            print(f"\\n📈 Visualization plots saved to: {args.output_dir}")
        
        print(f"\\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()