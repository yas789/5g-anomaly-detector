from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
import json

class NetworkAnomalyReporter:
    """
    Natural language reporter for 5G network anomalies using Hugging Face transformers
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", device: Optional[str] = None):
        """
        Initialize the anomaly reporter
        
        Args:
            model_name (str): Name of the Hugging Face model to use
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        try:
            # Initialize the text generation pipeline
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == 'cuda' else -1,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=-1,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        # Define metric descriptions and thresholds
        self.metric_info = {
            'prb_utilization': {
                'name': 'Physical Resource Block Utilization',
                'unit': '%',
                'normal_range': (20, 80),
                'description': 'measures how much of the available spectrum is being used'
            },
            'active_ue_count': {
                'name': 'Active User Equipment Count',
                'unit': 'devices',
                'normal_range': (50, 500),
                'description': 'indicates the number of connected devices'
            },
            'throughput_mbps': {
                'name': 'Network Throughput',
                'unit': 'Mbps',
                'normal_range': (200, 600),
                'description': 'measures data transfer speed'
            },
            'latency_ms': {
                'name': 'Network Latency',
                'unit': 'ms',
                'normal_range': (2, 20),
                'description': 'measures response time delay'
            },
            'handover_success_rate': {
                'name': 'Handover Success Rate',
                'unit': '%',
                'normal_range': (0.85, 0.98),
                'description': 'measures successful device transitions between cells'
            },
            'snr_db': {
                'name': 'Signal-to-Noise Ratio',
                'unit': 'dB',
                'normal_range': (10, 30),
                'description': 'measures signal quality'
            },
            'packet_loss_rate': {
                'name': 'Packet Loss Rate',
                'unit': '%',
                'normal_range': (0.001, 0.05),
                'description': 'measures percentage of lost data packets'
            }
        }
    
    def analyze_anomaly_patterns(self, data: pd.DataFrame, anomaly_indices: np.ndarray) -> Dict:
        """
        Analyze patterns in anomalous data
        
        Args:
            data (pd.DataFrame): Network metrics data
            anomaly_indices (np.ndarray): Indices of anomalous samples
            
        Returns:
            Dict: Analysis results
        """
        if len(anomaly_indices) == 0:
            return {"anomaly_count": 0, "patterns": []}
        
        anomaly_data = data.iloc[anomaly_indices]
        normal_data = data[~data.index.isin(anomaly_indices)]
        
        patterns = []
        
        for metric, info in self.metric_info.items():
            if metric not in data.columns:
                continue
                
            anomaly_values = anomaly_data[metric]
            normal_mean = normal_data[metric].mean()
            normal_std = normal_data[metric].std()
            
            anomaly_mean = anomaly_values.mean()
            anomaly_std = anomaly_values.std()
            
            # Determine anomaly type
            min_normal, max_normal = info['normal_range']
            
            if anomaly_mean < min_normal:
                anomaly_type = "below normal"
                severity = "high" if anomaly_mean < min_normal * 0.5 else "medium"
            elif anomaly_mean > max_normal:
                anomaly_type = "above normal"
                severity = "high" if anomaly_mean > max_normal * 1.5 else "medium"
            else:
                anomaly_type = "within range but irregular"
                severity = "low"
            
            patterns.append({
                'metric': metric,
                'metric_name': info['name'],
                'unit': info['unit'],
                'anomaly_mean': round(anomaly_mean, 3),
                'normal_mean': round(normal_mean, 3),
                'anomaly_type': anomaly_type,
                'severity': severity,
                'affected_samples': len(anomaly_values),
                'description': info['description']
            })
        
        return {
            "anomaly_count": len(anomaly_indices),
            "total_samples": len(data),
            "anomaly_percentage": round(len(anomaly_indices) / len(data) * 100, 2),
            "patterns": patterns
        }
    
    def create_anomaly_prompt(self, analysis: Dict, timestamp: Optional[str] = None) -> str:
        """
        Create a structured prompt for the LLM
        
        Args:
            analysis (Dict): Anomaly analysis results
            timestamp (str): Optional timestamp for the report
            
        Returns:
            str: Formatted prompt
        """
        if analysis["anomaly_count"] == 0:
            return "Generate a brief report: No network anomalies detected. All 5G network metrics are operating within normal parameters."
        
        timestamp_str = f" at {timestamp}" if timestamp else ""
        
        prompt = f"""Generate a professional network anomaly report{timestamp_str}:

Detected {analysis['anomaly_count']} anomalies out of {analysis['total_samples']} samples ({analysis['anomaly_percentage']}%).

Key findings:"""
        
        for pattern in analysis['patterns']:
            if pattern['affected_samples'] > 0:
                prompt += f"""
- {pattern['metric_name']}: {pattern['anomaly_mean']} {pattern['unit']} (normal: ~{pattern['normal_mean']} {pattern['unit']})
  Status: {pattern['anomaly_type']} - {pattern['severity']} severity
  Affected samples: {pattern['affected_samples']}"""
        
        prompt += "\n\nProvide a concise technical summary with potential causes and recommended actions:"
        
        return prompt
    
    def generate_report(self, prompt: str, max_length: int = 300) -> str:
        """
        Generate natural language report using the LLM
        
        Args:
            prompt (str): Input prompt for the model
            max_length (int): Maximum length of generated text
            
        Returns:
            str: Generated report
        """
        try:
            # Generate response using the pipeline
            response = self.generator(
                prompt,
                max_length=max_length,
                min_length=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            if isinstance(response, list) and len(response) > 0:
                return response[0]['generated_text'].strip()
            else:
                return "Error: Unable to generate report."
                
        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def create_detailed_report(self, data: pd.DataFrame, anomaly_indices: np.ndarray, 
                            timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        Create a comprehensive anomaly report
        
        Args:
            data (pd.DataFrame): Network metrics data
            anomaly_indices (np.ndarray): Indices of anomalous samples
            timestamp (str): Optional timestamp for the report
            
        Returns:
            Dict[str, str]: Complete report with sections
        """
        # Analyze anomaly patterns
        analysis = self.analyze_anomaly_patterns(data, anomaly_indices)
        
        # Generate executive summary
        summary_prompt = self.create_anomaly_prompt(analysis, timestamp)
        executive_summary = self.generate_report(summary_prompt, max_length=200)
        
        # Generate technical details
        if analysis["anomaly_count"] > 0:
            tech_prompt = f"""Provide technical analysis for 5G network anomalies:

Metrics affected: {', '.join([p['metric_name'] for p in analysis['patterns'] if p['affected_samples'] > 0])}

Explain potential root causes, network impact, and troubleshooting steps:"""
            
            technical_details = self.generate_report(tech_prompt, max_length=300)
            
            # Generate recommendations
            rec_prompt = f"""Based on these 5G network anomalies, provide specific recommendations:

Critical issues: {', '.join([p['metric_name'] for p in analysis['patterns'] if p['severity'] == 'high'])}

List actionable steps for network operators:"""
            
            recommendations = self.generate_report(rec_prompt, max_length=250)
        else:
            technical_details = "No technical issues detected. All network metrics are operating within expected parameters."
            recommendations = "Continue monitoring network performance. No immediate action required."
        
        return {
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "executive_summary": executive_summary,
            "technical_details": technical_details,
            "recommendations": recommendations,
            "metrics_analyzed": list(self.metric_info.keys()),
            "anomaly_statistics": {
                "total_anomalies": analysis["anomaly_count"],
                "total_samples": analysis["total_samples"],
                "anomaly_rate": f"{analysis['anomaly_percentage']}%"
            }
        }
    
    def generate_real_time_alert(self, metric_name: str, current_value: float, 
                               normal_range: tuple, severity: str = "medium") -> str:
        """
        Generate a real-time alert message
        
        Args:
            metric_name (str): Name of the affected metric
            current_value (float): Current anomalous value
            normal_range (tuple): Normal range for the metric
            severity (str): Severity level
            
        Returns:
            str: Alert message
        """
        metric_info = self.metric_info.get(metric_name, {})
        display_name = metric_info.get('name', metric_name)
        unit = metric_info.get('unit', '')
        
        prompt = f"""Generate a brief network alert:

ALERT: {display_name} anomaly detected
Current value: {current_value} {unit}
Normal range: {normal_range[0]} - {normal_range[1]} {unit}
Severity: {severity}

Provide a concise alert message for network operators:"""
        
        return self.generate_report(prompt, max_length=100)
    
    def export_report(self, report: Dict[str, str], filepath: str, format: str = "json"):
        """
        Export report to file
        
        Args:
            report (Dict): Report data
            filepath (str): Output file path
            format (str): Export format ('json', 'txt')
        """
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        elif format.lower() == "txt":
            with open(filepath, 'w') as f:
                f.write(f"5G Network Anomaly Report\n")
                f.write(f"Generated: {report['timestamp']}\n")
                f.write(f"{'='*50}\n\n")
                
                f.write(f"EXECUTIVE SUMMARY\n")
                f.write(f"{'-'*17}\n")
                f.write(f"{report['executive_summary']}\n\n")
                
                f.write(f"TECHNICAL DETAILS\n")
                f.write(f"{'-'*17}\n")
                f.write(f"{report['technical_details']}\n\n")
                
                f.write(f"RECOMMENDATIONS\n")
                f.write(f"{'-'*15}\n")
                f.write(f"{report['recommendations']}\n\n")
                
                f.write(f"STATISTICS\n")
                f.write(f"{'-'*10}\n")
                for key, value in report['anomaly_statistics'].items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        print(f"Report exported to: {filepath}")

def demo_reporter():
    """Demonstrate the anomaly reporter functionality"""
    print("Initializing Network Anomaly Reporter...")
    
    # Initialize reporter
    reporter = NetworkAnomalyReporter()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'prb_utilization': [45.2, 98.5, 42.1, 95.8],  # 2 anomalies
        'active_ue_count': [200, 150, 980, 220],       # 1 anomaly
        'throughput_mbps': [350, 80, 400, 390],        # 1 anomaly
        'latency_ms': [8.5, 150.2, 9.1, 8.8],         # 1 anomaly
        'handover_success_rate': [0.92, 0.94, 0.15, 0.91],  # 1 anomaly
        'snr_db': [22.1, 21.8, 45.5, 20.9],           # 1 anomaly
        'packet_loss_rate': [0.02, 0.01, 0.35, 0.018] # 1 anomaly
    })
    
    # Simulate anomaly detection results
    anomaly_indices = np.array([1, 2])  # Indices of anomalous samples
    
    # Generate comprehensive report
    report = reporter.create_detailed_report(
        sample_data, 
        anomaly_indices,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    print("\n" + "="*60)
    print("GENERATED ANOMALY REPORT")
    print("="*60)
    
    for section, content in report.items():
        if section not in ['metrics_analyzed', 'anomaly_statistics']:
            print(f"\n{section.upper().replace('_', ' ')}:")
            print("-" * len(section))
            print(content)
    
    # Generate real-time alert
    print("\n" + "="*60)
    print("REAL-TIME ALERT EXAMPLE")
    print("="*60)
    
    alert = reporter.generate_real_time_alert(
        'latency_ms', 
        150.2, 
        (2, 20), 
        'high'
    )
    print(alert)

if __name__ == "__main__":
    demo_reporter()