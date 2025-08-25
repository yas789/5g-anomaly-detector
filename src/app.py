#!/usr/bin/env python3
"""
FastAPI application for 5G Network Anomaly Detection Service
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import traceback
import glob
import psutil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import uvicorn

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from models.llm_reporter import NetworkAnomalyReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class NetworkMetrics(BaseModel):
    """Input model for network metrics"""
    prb_utilization: float = Field(..., ge=0, le=100, description="PRB utilization percentage")
    active_ue_count: int = Field(..., ge=0, description="Number of active user equipment")
    throughput_mbps: float = Field(..., ge=0, description="Network throughput in Mbps")
    latency_ms: float = Field(..., ge=0, description="Network latency in milliseconds")
    handover_success_rate: float = Field(..., ge=0, le=1, description="Handover success rate (0-1)")
    snr_db: float = Field(..., description="Signal-to-noise ratio in dB")
    packet_loss_rate: float = Field(..., ge=0, le=1, description="Packet loss rate (0-1)")
    timestamp: Optional[str] = Field(None, description="Optional timestamp")
    
    @validator('handover_success_rate', 'packet_loss_rate')
    def validate_rates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Rate values must be between 0 and 1')
        return v

class NetworkMetricsBatch(BaseModel):
    """Input model for batch network metrics"""
    metrics: List[NetworkMetrics] = Field(..., min_items=1, max_items=1000, 
                                        description="List of network metrics (max 1000)")

class AnomalyResult(BaseModel):
    """Output model for anomaly detection result"""
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: float
    timestamp: str
    metrics_analyzed: Dict[str, float]

class AnomalyReport(BaseModel):
    """Output model for anomaly report"""
    timestamp: str
    total_samples: int
    anomalies_detected: int
    anomaly_rate: float
    executive_summary: str
    technical_details: str
    recommendations: str
    alerts: List[Dict[str, Any]]

class BatchPredictionResult(BaseModel):
    """Output model for batch prediction results"""
    total_samples: int
    anomalies_detected: int
    anomaly_rate: float
    results: List[AnomalyResult]
    report: Optional[AnomalyReport] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    model_loaded: bool
    llm_loaded: bool
    uptime_seconds: float

class MetricsResponse(BaseModel):
    """System metrics response model"""
    system: Dict[str, Any]
    api: Dict[str, Any]
    model: Dict[str, Any]

# Global application state
class AppState:
    def __init__(self):
        self.detector: Optional[AnomalyDetector] = None
        self.reporter: Optional[NetworkAnomalyReporter] = None
        self.start_time = time.time()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.feature_columns = [
            'prb_utilization', 'active_ue_count', 'throughput_mbps',
            'latency_ms', 'handover_success_rate', 'snr_db', 'packet_loss_rate'
        ]

app_state = AppState()

# Initialize FastAPI app
app = FastAPI(
    title="5G Network Anomaly Detection API",
    description="Real-time anomaly detection and reporting for 5G network metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_latest_model(models_dir: str = "models") -> str:
    """Find the latest trained model"""
    try:
        model_pattern = os.path.join(models_dir, 'anomaly_detector_*.pkl')
        model_files = glob.glob(model_pattern)
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {models_dir}")
        return max(model_files, key=os.path.getmtime)
    except Exception as e:
        logger.error(f"Error finding model: {e}")
        raise

def load_model():
    """Load the trained anomaly detection model and LLM reporter"""
    try:
        logger.info("Loading anomaly detection model...")
        
        # Find and load the latest model
        model_path = find_latest_model()
        logger.info(f"Found model: {model_path}")
        
        app_state.detector = AnomalyDetector(input_dim=len(app_state.feature_columns))
        app_state.detector.load_model(model_path)
        
        logger.info("Loading LLM reporter...")
        app_state.reporter = NetworkAnomalyReporter(model_name="google/flan-t5-small")
        
        logger.info("✓ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    try:
        logger.info("Starting 5G Network Anomaly Detection API...")
        load_model()
        logger.info("API ready to serve requests")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue without models for health checks
        pass

@app.middleware("http")
async def request_middleware(request, call_next):
    """Middleware to track requests and handle errors"""
    start_time = time.time()
    app_state.request_count += 1
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Request failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )
    finally:
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        uptime = time.time() - app_state.start_time
        
        return HealthResponse(
            status="healthy" if app_state.detector and app_state.reporter else "degraded",
            timestamp=datetime.now().isoformat(),
            model_loaded=app_state.detector is not None,
            llm_loaded=app_state.reporter is not None,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system and API metrics"""
    try:
        uptime = time.time() - app_state.start_time
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        system_metrics = {
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "uptime_seconds": uptime
        }
        
        # API metrics
        api_metrics = {
            "total_requests": app_state.request_count,
            "total_predictions": app_state.prediction_count,
            "total_errors": app_state.error_count,
            "error_rate": app_state.error_count / max(app_state.request_count, 1),
            "requests_per_second": app_state.request_count / max(uptime, 1)
        }
        
        # Model metrics
        model_metrics = {
            "model_loaded": app_state.detector is not None,
            "llm_loaded": app_state.reporter is not None,
            "threshold": float(app_state.detector.threshold) if app_state.detector else None,
            "feature_count": len(app_state.feature_columns)
        }
        
        return MetricsResponse(
            system=system_metrics,
            api=api_metrics,
            model=model_metrics
        )
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=AnomalyResult)
async def predict_single(metrics: NetworkMetrics):
    """Predict anomaly for a single set of network metrics"""
    try:
        if not app_state.detector:
            raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")
        
        app_state.prediction_count += 1
        
        # Convert to DataFrame
        data = {
            'prb_utilization': [metrics.prb_utilization],
            'active_ue_count': [metrics.active_ue_count],
            'throughput_mbps': [metrics.throughput_mbps],
            'latency_ms': [metrics.latency_ms],
            'handover_success_rate': [metrics.handover_success_rate],
            'snr_db': [metrics.snr_db],
            'packet_loss_rate': [metrics.packet_loss_rate]
        }
        
        df = pd.DataFrame(data)
        
        # Predict anomaly
        reconstruction_errors, anomaly_predictions = app_state.detector.predict_anomalies(
            df, app_state.feature_columns
        )
        
        reconstruction_error = float(reconstruction_errors[0])
        is_anomaly = bool(anomaly_predictions[0])
        
        # Calculate confidence (inverse of distance from threshold)
        threshold = app_state.detector.threshold
        if is_anomaly:
            confidence = min(0.99, (reconstruction_error - threshold) / threshold)
        else:
            confidence = min(0.99, (threshold - reconstruction_error) / threshold)
        confidence = max(0.01, confidence)
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            reconstruction_error=reconstruction_error,
            threshold=threshold,
            confidence=confidence,
            timestamp=metrics.timestamp or datetime.now().isoformat(),
            metrics_analyzed={
                col: float(df[col].iloc[0]) for col in app_state.feature_columns
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(batch: NetworkMetricsBatch, generate_report: bool = True, 
                       background_tasks: BackgroundTasks = None):
    """Predict anomalies for a batch of network metrics"""
    try:
        if not app_state.detector:
            raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")
        
        if not app_state.reporter and generate_report:
            raise HTTPException(status_code=503, detail="LLM reporter not loaded")
        
        app_state.prediction_count += len(batch.metrics)
        
        # Convert to DataFrame
        data = []
        for metric in batch.metrics:
            data.append({
                'prb_utilization': metric.prb_utilization,
                'active_ue_count': metric.active_ue_count,
                'throughput_mbps': metric.throughput_mbps,
                'latency_ms': metric.latency_ms,
                'handover_success_rate': metric.handover_success_rate,
                'snr_db': metric.snr_db,
                'packet_loss_rate': metric.packet_loss_rate,
                'timestamp': metric.timestamp or datetime.now().isoformat()
            })
        
        df = pd.DataFrame(data)
        
        # Predict anomalies
        reconstruction_errors, anomaly_predictions = app_state.detector.predict_anomalies(
            df, app_state.feature_columns
        )
        
        # Create individual results
        results = []
        threshold = app_state.detector.threshold
        
        for i, (error, is_anomaly) in enumerate(zip(reconstruction_errors, anomaly_predictions)):
            # Calculate confidence
            if is_anomaly:
                confidence = min(0.99, (error - threshold) / threshold)
            else:
                confidence = min(0.99, (threshold - error) / threshold)
            confidence = max(0.01, confidence)
            
            results.append(AnomalyResult(
                is_anomaly=bool(is_anomaly),
                reconstruction_error=float(error),
                threshold=threshold,
                confidence=confidence,
                timestamp=data[i]['timestamp'],
                metrics_analyzed={
                    col: float(df[col].iloc[i]) for col in app_state.feature_columns
                }
            ))
        
        # Generate report if requested
        report = None
        if generate_report:
            try:
                anomaly_indices = np.where(anomaly_predictions)[0]
                
                if len(anomaly_indices) > 0:
                    # Generate comprehensive report
                    llm_report = app_state.reporter.create_detailed_report(
                        df, anomaly_indices, datetime.now().isoformat()
                    )
                    
                    # Generate alerts
                    alerts = app_state.reporter.generate_real_time_alerts(df, anomaly_indices)
                    
                    report = AnomalyReport(
                        timestamp=llm_report['timestamp'],
                        total_samples=len(df),
                        anomalies_detected=len(anomaly_indices),
                        anomaly_rate=len(anomaly_indices) / len(df),
                        executive_summary=llm_report['executive_summary'],
                        technical_details=llm_report['technical_details'],
                        recommendations=llm_report['recommendations'],
                        alerts=alerts
                    )
                else:
                    report = AnomalyReport(
                        timestamp=datetime.now().isoformat(),
                        total_samples=len(df),
                        anomalies_detected=0,
                        anomaly_rate=0.0,
                        executive_summary="No anomalies detected. All network metrics are operating within normal parameters.",
                        technical_details="All metrics are within expected ranges. System is performing optimally.",
                        recommendations="Continue monitoring network performance. No immediate action required.",
                        alerts=[]
                    )
                    
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")
                # Continue without report
                
        return BatchPredictionResult(
            total_samples=len(batch.metrics),
            anomalies_detected=int(anomaly_predictions.sum()),
            anomaly_rate=float(anomaly_predictions.mean()),
            results=results,
            report=report
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_state.error_count += 1
        logger.error(f"Batch prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload the anomaly detection model (admin endpoint)"""
    try:
        logger.info("Reloading models...")
        load_model()
        return {"status": "success", "message": "Models reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": str(exc)}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    """Handle file not found errors"""
    return JSONResponse(
        status_code=503,
        content={"error": "Service unavailable", "detail": "Required model files not found"}
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="5G Network Anomaly Detection API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        access_log=True
    )