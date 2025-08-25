#!/usr/bin/env python3
"""
Inference script for 5G Network Anomaly Detection
Loads trained model, detects anomalies, and generates LLM reports
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json
import glob

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from models.llm_reporter import NetworkAnomalyReporter
from data.generate_synthetic_data import generate_synthetic_5g_data

class NetworkAnomalyInference:
    """
    Main inference class for 5G network anomaly detection and reporting
    """
    
    def __init__(self, model_path: str, llm_model: str = "google/flan-t5-small"):
        """
        Initialize the inference system
        
        Args:
            model_path (str): Path to the trained model
            llm_model (str): Name of the LLM model for reporting
        """
        self.model_path = model_path
        self.detector = None
        self.reporter = None
        self.feature_columns = [
            'prb_utilization',
            'active_ue_count', 
            'throughput_mbps',
            'latency_ms',
            'handover_success_rate',
            'snr_db',
            'packet_loss_rate'
        ]
        
        # Load the trained model
        self._load_model()
        
        # Initialize the LLM reporter
        print("Initializing LLM reporter...")
        self.reporter = NetworkAnomalyReporter(model_name=llm_model)
        
    def _load_model(self):
        """Load the trained anomaly detection model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading trained model from: {self.model_path}")
        
        # Initialize detector with dummy dimensions (will be overridden by load_model)
        self.detector = AnomalyDetector(input_dim=len(self.feature_columns))
        
        # Load the trained model
        self.detector.load_model(self.model_path)
        
        print("✓ Model loaded successfully")
        print(f"  - Input features: {len(self.detector.feature_names)}")
        print(f"  - Anomaly threshold: {self.detector.threshold:.6f}")
    
    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input data format and columns
        
        Args:
            df (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if valid, raises exception if not
        """
        # Check required columns
        missing_columns = set(self.feature_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty data
        if len(df) == 0:
            raise ValueError("Input data is empty")
        
        # Check for null values
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            print("Warning: Found null values in data:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"  - {col}: {count} null values")
        
        # Check data types
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Warning: Column '{col}' is not numeric, attempting conversion...")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Data validation passed: {len(df)} samples, {len(self.feature_columns)} features")
        return True
    
    def detect_anomalies(self, df: pd.DataFrame) -> tuple:
        """
        Detect anomalies in the input data
        
        Args:
            df (pd.DataFrame): Input network metrics data
            
        Returns:
            tuple: (reconstruction_errors, anomaly_predictions, anomaly_indices)
        """
        print("Detecting anomalies...")
        
        # Validate input data
        self.validate_input_data(df)
        
        # Detect anomalies
        reconstruction_errors, anomaly_predictions = self.detector.predict_anomalies(df, self.feature_columns)
        
        # Get indices of anomalous samples
        anomaly_indices = np.where(anomaly_predictions)[0]
        
        print(f"✓ Anomaly detection completed")
        print(f"  - Total samples analyzed: {len(df)}")
        print(f"  - Anomalies detected: {len(anomaly_indices)} ({len(anomaly_indices)/len(df)*100:.1f}%)")
        print(f"  - Average reconstruction error: {reconstruction_errors.mean():.6f}")
        print(f"  - Max reconstruction error: {reconstruction_errors.max():.6f}")
        
        return reconstruction_errors, anomaly_predictions, anomaly_indices
    
    def generate_reports(self, df: pd.DataFrame, anomaly_indices: np.ndarray, 
                        timestamp: str = None) -> dict:
        """
        Generate comprehensive anomaly reports using LLM
        
        Args:
            df (pd.DataFrame): Input data
            anomaly_indices (np.ndarray): Indices of anomalous samples
            timestamp (str): Optional timestamp for the report
            
        Returns:
            dict: Complete anomaly report
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("Generating LLM reports...")
        
        # Generate comprehensive report
        report = self.reporter.create_detailed_report(df, anomaly_indices, timestamp)
        
        # Add technical metrics
        if len(anomaly_indices) > 0:
            anomaly_data = df.iloc[anomaly_indices]
            report['anomaly_details'] = {}
            
            for col in self.feature_columns:
                anomaly_values = anomaly_data[col]
                report['anomaly_details'][col] = {
                    'mean': float(anomaly_values.mean()),
                    'std': float(anomaly_values.std()),
                    'min': float(anomaly_values.min()),
                    'max': float(anomaly_values.max()),
                    'affected_samples': len(anomaly_values)
                }
        
        print("✓ Reports generated successfully")
        return report
    
    def generate_real_time_alerts(self, df: pd.DataFrame, anomaly_indices: np.ndarray) -> list:
        """
        Generate real-time alerts for critical anomalies
        
        Args:
            df (pd.DataFrame): Input data
            anomaly_indices (np.ndarray): Indices of anomalous samples
            
        Returns:
            list: List of alert messages
        """
        alerts = []
        
        if len(anomaly_indices) == 0:
            return alerts
        
        print("Generating real-time alerts...")
        
        # Define critical thresholds for immediate alerts
        critical_thresholds = {
            'prb_utilization': (95, 'Network congestion detected'),
            'latency_ms': (50, 'High latency affecting user experience'),
            'packet_loss_rate': (0.1, 'Significant packet loss detected'),
            'handover_success_rate': (0.5, 'Handover failures detected'),
            'snr_db': (5, 'Poor signal quality detected')
        }
        
        anomaly_data = df.iloc[anomaly_indices]
        
        for metric, (threshold, description) in critical_thresholds.items():
            if metric in anomaly_data.columns:
                critical_values = anomaly_data[metric]
                
                if metric in ['prb_utilization', 'latency_ms', 'packet_loss_rate']:
                    critical_samples = critical_values[critical_values > threshold]
                elif metric in ['handover_success_rate', 'snr_db']:
                    critical_samples = critical_values[critical_values < threshold]
                else:
                    continue
                
                if len(critical_samples) > 0:
                    alert_msg = self.reporter.generate_real_time_alert(
                        metric,
                        critical_samples.mean(),
                        self.reporter.metric_info[metric]['normal_range'],
                        'high'
                    )
                    alerts.append({
                        'metric': metric,
                        'severity': 'high',
                        'affected_samples': len(critical_samples),
                        'message': alert_msg,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        print(f"✓ Generated {len(alerts)} real-time alerts")
        return alerts
    
    def save_results(self, df: pd.DataFrame, reconstruction_errors: np.ndarray, 
                    anomaly_predictions: np.ndarray, report: dict, alerts: list,
                    output_dir: str, timestamp: str) -> dict:
        """
        Save all inference results to files
        
        Args:
            df (pd.DataFrame): Original input data
            reconstruction_errors (np.ndarray): Reconstruction errors
            anomaly_predictions (np.ndarray): Binary anomaly predictions
            report (dict): Generated report
            alerts (list): Real-time alerts
            output_dir (str): Output directory
            timestamp (str): Timestamp for file naming
            
        Returns:
            dict: Paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save detailed results CSV
        results_df = df.copy()
        results_df['reconstruction_error'] = reconstruction_errors
        results_df['is_anomaly_predicted'] = anomaly_predictions
        results_df['timestamp'] = pd.to_datetime(results_df.get('timestamp', datetime.now()))
        
        results_path = os.path.join(output_dir, f'inference_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        saved_files['results'] = results_path
        
        # Save comprehensive report
        report_json_path = os.path.join(output_dir, f'anomaly_report_{timestamp}.json')
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=2)
        saved_files['report_json'] = report_json_path
        
        # Save human-readable report
        report_txt_path = os.path.join(output_dir, f'anomaly_report_{timestamp}.txt')
        self.reporter.export_report(report, report_txt_path, format='txt')
        saved_files['report_txt'] = report_txt_path
        
        # Save alerts
        if alerts:
            alerts_path = os.path.join(output_dir, f'alerts_{timestamp}.json')
            with open(alerts_path, 'w') as f:
                json.dump(alerts, f, indent=2)
            saved_files['alerts'] = alerts_path
        
        # Save summary statistics
        summary = {
            'timestamp': timestamp,
            'total_samples': len(df),
            'anomalies_detected': int(anomaly_predictions.sum()),
            'anomaly_rate': float(anomaly_predictions.mean()),
            'avg_reconstruction_error': float(reconstruction_errors.mean()),
            'max_reconstruction_error': float(reconstruction_errors.max()),
            'model_threshold': float(self.detector.threshold),
            'alerts_generated': len(alerts)
        }
        
        summary_path = os.path.join(output_dir, f'summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = summary_path
        
        return saved_files

def find_latest_model(models_dir: str) -> str:
    """
    Find the latest trained model in the models directory
    
    Args:
        models_dir (str): Directory containing trained models
        
    Returns:
        str: Path to the latest model file
    """
    model_pattern = os.path.join(models_dir, 'anomaly_detector_*.pkl')
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {models_dir}")
    
    # Sort by modification time, get the latest
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def load_sample_data(data_path: str, num_samples: int = 1000) -> pd.DataFrame:
    """
    Load or generate sample data for inference
    
    Args:
        data_path (str): Path to data file
        num_samples (int): Number of samples to generate if file doesn't exist
        
    Returns:
        pd.DataFrame: Sample data
    """
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        if 'is_anomaly' in df.columns:
            # Remove ground truth labels for real inference
            df = df.drop('is_anomaly', axis=1)
    else:
        print(f"Generating sample data for inference...")
        df = generate_synthetic_5g_data(num_samples=num_samples, anomaly_rate=0.15)
        # Remove ground truth labels for real inference
        if 'is_anomaly' in df.columns:
            df = df.drop('is_anomaly', axis=1)
    
    return df

def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='5G Network Anomaly Detection Inference')
    parser.add_argument('--model-path', type=str, 
                        help='Path to trained model (if not provided, finds latest in models/)')
    parser.add_argument('--data-path', type=str, 
                        help='Path to input data CSV file')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to generate if no data file provided')
    parser.add_argument('--llm-model', type=str, default='google/flan-t5-small',
                        help='LLM model for report generation')
    parser.add_argument('--no-reports', action='store_true',
                        help='Skip LLM report generation')
    parser.add_argument('--alerts-only', action='store_true',
                        help='Generate only critical alerts, skip detailed reports')
    
    args = parser.parse_args()
    
    # Create timestamp for this inference run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("5G NETWORK ANOMALY DETECTION - INFERENCE")
    print("="*60)
    print(f"Inference started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Find model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = find_latest_model(args.models_dir)
        
        print(f"Using model: {model_path}")
        
        # Load input data
        if args.data_path:
            df = load_sample_data(args.data_path)
        else:
            df = load_sample_data('', args.num_samples)
        
        print(f"Loaded {len(df)} samples for inference")
        
        # Initialize inference system
        inference_system = NetworkAnomalyInference(
            model_path=model_path,
            llm_model=args.llm_model if not args.no_reports else None
        )
        
        # Detect anomalies
        reconstruction_errors, anomaly_predictions, anomaly_indices = inference_system.detect_anomalies(df)
        
        # Generate reports and alerts
        report = None
        alerts = []
        
        if len(anomaly_indices) > 0:
            if not args.no_reports and not args.alerts_only:
                report = inference_system.generate_reports(df, anomaly_indices, timestamp)
            
            if not args.no_reports:
                alerts = inference_system.generate_real_time_alerts(df, anomaly_indices)
        else:
            print("No anomalies detected - system operating normally")
            if not args.no_reports and not args.alerts_only:
                report = inference_system.generate_reports(df, anomaly_indices, timestamp)
        
        # Save results
        saved_files = inference_system.save_results(
            df, reconstruction_errors, anomaly_predictions, 
            report, alerts, args.output_dir, timestamp
        )
        
        # Print summary
        print("\n" + "="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"✓ Processed {len(df)} network metric samples")
        print(f"✓ Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(df)*100:.1f}%)")
        
        if alerts:
            print(f"✓ Generated {len(alerts)} critical alerts")
        
        if report:
            print(f"✓ Generated comprehensive LLM report")
        
        print(f"✓ Results saved to: {args.output_dir}")
        
        # Show critical alerts
        if alerts:
            print("\nCRITICAL ALERTS:")
            print("-" * 30)
            for alert in alerts:
                print(f"⚠️  {alert['metric']}: {alert['message'][:100]}...")
        
        print(f"\nInference completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print file paths
        print("\nGenerated files:")
        for file_type, path in saved_files.items():
            print(f"  - {file_type}: {path}")
        
    except Exception as e:
        print(f"\n❌ Inference failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()