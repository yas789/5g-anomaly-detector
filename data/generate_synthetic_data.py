import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_5g_data(num_samples=10000, anomaly_rate=0.1):
    """
    Generate synthetic 5G network metrics with anomalies
    
    Args:
        num_samples (int): Number of data points to generate
        anomaly_rate (float): Percentage of anomalies to inject (0.1 = 10%)
    
    Returns:
        pandas.DataFrame: Generated synthetic data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=7)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]
    
    # Initialize data dictionary
    data = {
        'timestamp': timestamps,
        'prb_utilization': [],
        'active_ue_count': [],
        'throughput_mbps': [],
        'latency_ms': [],
        'handover_success_rate': [],
        'snr_db': [],
        'packet_loss_rate': [],
        'is_anomaly': []
    }
    
    # Determine which samples will be anomalies
    num_anomalies = int(num_samples * anomaly_rate)
    anomaly_indices = set(random.sample(range(num_samples), num_anomalies))
    
    for i in range(num_samples):
        is_anomaly = i in anomaly_indices
        
        if is_anomaly:
            # Generate anomalous values
            prb_util = np.random.choice([
                np.random.uniform(0, 10),  # Very low utilization
                np.random.uniform(90, 100)  # Very high utilization
            ])
            ue_count = np.random.choice([
                np.random.uniform(0, 5),    # Very few users
                np.random.uniform(950, 1000) # Network overload
            ])
            throughput = np.random.choice([
                np.random.uniform(0, 50),   # Very low throughput
                np.random.uniform(800, 1000) # Unusually high
            ])
            latency = np.random.choice([
                np.random.uniform(0.1, 0.5), # Unusually low
                np.random.uniform(80, 200)   # Very high latency
            ])
            handover_rate = np.random.choice([
                np.random.uniform(0, 0.3),  # Very low success rate
                np.random.uniform(0.99, 1.0) # Suspiciously perfect
            ])
            snr = np.random.choice([
                np.random.uniform(-10, 0),  # Poor signal
                np.random.uniform(35, 50)   # Unusually strong
            ])
            packet_loss = np.random.choice([
                np.random.uniform(0, 0.001), # No loss (suspicious)
                np.random.uniform(0.15, 0.5)  # High packet loss
            ])
        else:
            # Generate normal values with realistic distributions
            prb_util = np.random.normal(45, 15)  # Mean 45%, std 15%
            prb_util = np.clip(prb_util, 10, 85)  # Reasonable bounds
            
            ue_count = np.random.poisson(200)  # Average 200 active UEs
            ue_count = np.clip(ue_count, 50, 800)
            
            throughput = np.random.normal(300, 80)  # Mean 300 Mbps
            throughput = np.clip(throughput, 100, 600)
            
            latency = np.random.exponential(8) + 2  # Exponential with shift
            latency = np.clip(latency, 2, 50)
            
            handover_rate = np.random.beta(8, 2)  # Skewed towards high success
            handover_rate = np.clip(handover_rate, 0.7, 0.98)
            
            snr = np.random.normal(20, 8)  # Mean 20 dB
            snr = np.clip(snr, 5, 35)
            
            packet_loss = np.random.exponential(0.02)  # Low packet loss
            packet_loss = np.clip(packet_loss, 0.001, 0.1)
        
        # Add values to data dictionary
        data['prb_utilization'].append(round(prb_util, 2))
        data['active_ue_count'].append(int(ue_count))
        data['throughput_mbps'].append(round(throughput, 2))
        data['latency_ms'].append(round(latency, 2))
        data['handover_success_rate'].append(round(handover_rate, 4))
        data['snr_db'].append(round(snr, 2))
        data['packet_loss_rate'].append(round(packet_loss, 6))
        data['is_anomaly'].append(is_anomaly)
    
    return pd.DataFrame(data)

def main():
    """Generate and save synthetic 5G network data"""
    print("Generating synthetic 5G network metrics...")
    
    # Generate the data
    df = generate_synthetic_5g_data(num_samples=10000, anomaly_rate=0.1)
    
    # Save to CSV
    output_path = 'synthetic_5g_metrics.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples with {df['is_anomaly'].sum()} anomalies ({df['is_anomaly'].mean():.1%})")
    print(f"Data saved to: {output_path}")
    
    # Display basic statistics
    print("\nData Summary:")
    print(df.describe())
    
    print(f"\nAnomalies by metric (mean values):")
    normal_data = df[df['is_anomaly'] == False]
    anomaly_data = df[df['is_anomaly'] == True]
    
    metrics = ['prb_utilization', 'active_ue_count', 'throughput_mbps', 
               'latency_ms', 'handover_success_rate', 'snr_db', 'packet_loss_rate']
    
    for metric in metrics:
        normal_mean = normal_data[metric].mean()
        anomaly_mean = anomaly_data[metric].mean()
        print(f"{metric}: Normal={normal_mean:.3f}, Anomaly={anomaly_mean:.3f}")

if __name__ == "__main__":
    main()